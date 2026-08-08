"""Human-vs-machine text detector implemented without scikit-learn.

Only NumPy, Pandas and SciPy are used.  The script implements:
  * word n-gram TF-IDF vectorisation from scratch;
  * NB log-count feature reweighting;
  * sparse logistic regression with Adam from scratch;
  * stylometric feature extraction;
  * an Extra-Trees-style classifier from scratch;
  * validation-based blending/threshold selection;
  * a verified Kaggle id,label submission.

Example (run inside TheHomiesML):
  python train_no_sklearn_high_score.py --project-root .
"""
from __future__ import annotations

import argparse, json, math, re, zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

SEED = 42
WORD_RE = re.compile(r"[a-z]+(?:['’-][a-z]+)*|\d+(?:\.\d+)?", re.I)
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
PARA_RE = re.compile(r"\n\s*\n+")
FUNCTION_WORDS = set("""a an the and but or for so of in on at by to from with without into through
during before after above below between under then once here there when where why how all any both each
few more most other some such no not only own same than too very can could may might must will would shall
should is am are was were be been being have has had do does did this that these those i me my we us our
you your he him his she her it its they them their who whom whose which what because although however
therefore moreover nevertheless thus hence while whereas""".split())
TRANSITIONS = ("however", "therefore", "moreover", "furthermore", "in addition", "for example",
               "for instance", "on the other hand", "in conclusion", "overall", "additionally",
               "consequently", "nevertheless", "as a result", "in contrast", "firstly", "secondly")
CONTRACTION_RE = re.compile(r"\b(?:\w+n['’]t|i['’]m|i['’]ve|i['’]ll|\w+['’](?:d|ll|ve|re|s))\b", re.I)


def tokens(text): return WORD_RE.findall(str(text).lower())


def ngrams(text, ngram_range=(1, 3)):
    w = tokens(text)
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(w) - n + 1):
            yield "\x1f".join(w[i:i+n])


class TfidfFromScratch:
    def __init__(self, max_features=260_000, min_df=2, ngram_range=(1, 3)):
        self.max_features, self.min_df, self.ngram_range = max_features, min_df, ngram_range

    def fit(self, texts):
        print("Learning word n-gram vocabulary ...", flush=True)
        df = Counter()
        for i, text in enumerate(texts):
            df.update(set(ngrams(text, self.ngram_range)))
            if (i + 1) % 4000 == 0: print(f"  scanned {i+1:,} texts", flush=True)
        kept = [(term, count) for term, count in df.items() if count >= self.min_df]
        kept.sort(key=lambda x: (-x[1], x[0]))
        kept = kept[:self.max_features]
        self.vocab = {term: i for i, (term, _) in enumerate(kept)}
        counts = np.asarray([count for _, count in kept], dtype=np.float32)
        self.idf = np.log((1.0 + len(texts)) / (1.0 + counts)) + 1.0
        print(f"Vocabulary size: {len(self.vocab):,}", flush=True)
        return self

    def transform(self, texts):
        indptr, indices, data = [0], [], []
        for i, text in enumerate(texts):
            row = Counter(self.vocab[g] for g in ngrams(text, self.ngram_range) if g in self.vocab)
            for j, count in row.items():
                indices.append(j); data.append(1.0 + math.log(count))
            indptr.append(len(indices))
        x = sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32),
                               np.asarray(indptr, np.int64)), shape=(len(texts), len(self.vocab)))
        x = x.multiply(self.idf).tocsr()
        norm = np.sqrt(x.multiply(x).sum(axis=1)).A1
        norm[norm == 0] = 1
        return sparse.diags(1.0 / norm).dot(x).tocsr()

    def fit_transform(self, texts): return self.fit(texts).transform(texts)


def nb_ratio(x, y, alpha=1.0):
    pos = np.asarray(x[y == 1].sum(axis=0)).ravel() + alpha
    neg = np.asarray(x[y == 0].sum(axis=0)).ravel() + alpha
    return np.log(pos / pos.sum()) - np.log(neg / neg.sum())


class SparseLogisticAdam:
    def __init__(self, epochs=32, batch_size=256, lr=.025, l2=2e-5, seed=SEED):
        self.epochs, self.batch_size, self.lr, self.l2, self.seed = epochs, batch_size, lr, l2, seed

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed); n, p = x.shape
        self.w = np.zeros(p, np.float32); self.b = 0.0
        m = np.zeros(p, np.float32); v = np.zeros(p, np.float32); mb = vb = 0.0; step = 0
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start+self.batch_size]; xb = x[idx]; yb = y[idx]
                pred = expit(xb.dot(self.w) + self.b); err = pred - yb
                grad = np.asarray(xb.T.dot(err)).ravel().astype(np.float32) / len(idx) + self.l2*self.w
                gb = float(err.mean()); step += 1
                m = .9*m + .1*grad; v = .999*v + .001*grad*grad
                mb = .9*mb + .1*gb; vb = .999*vb + .001*gb*gb
                mh = m/(1-.9**step); vh = v/(1-.999**step)
                self.w -= self.lr*mh/(np.sqrt(vh)+1e-8)
                self.b -= self.lr*(mb/(1-.9**step))/(math.sqrt(vb/(1-.999**step))+1e-8)
            if epoch in (0, 3, 7, 15, self.epochs-1):
                print(f"  linear epoch {epoch+1}/{self.epochs}", flush=True)
        return self

    def predict_proba(self, x): return expit(x.dot(self.w) + self.b)


def safe(a, b): return float(a)/b if b else 0.0
def stats(a, prefix):
    a = np.asarray(a or [0], float); mean = a.mean()
    return {prefix+"_mean":mean, prefix+"_std":a.std(), prefix+"_min":a.min(),
            prefix+"_max":a.max(), prefix+"_median":np.median(a), prefix+"_cv":safe(a.std(),mean)}


def style(text):
    text = str(text); low = text.lower(); w = tokens(text); wc = Counter(w)
    sents = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    paras = [p.strip() for p in PARA_RE.split(text) if p.strip()]
    sl = [len(tokens(s)) for s in sents]; pl = [len(tokens(p)) for p in paras]
    wl = [len(x) for x in w]; bi = list(zip(w,w[1:])); tri = list(zip(w,w[1:],w[2:]))
    starts = [tokens(s)[0] for s in sents if tokens(s)]
    enc = text.encode("utf8", "ignore"); nw=max(len(w),1); nc=max(len(text),1); ns=max(len(sents),1)
    freq = np.asarray(list(wc.values()), float)
    prob = freq/freq.sum() if len(freq) else np.array([1.])
    f = {
      "chars":len(text), "words":len(w), "sentences":len(sents), "paragraphs":len(paras),
      "unique_ratio":safe(len(wc),len(w)), "hapax_ratio":safe(sum(v==1 for v in wc.values()),len(wc)),
      "entropy":float(-(prob*np.log2(prob)).sum()), "bigram_unique":safe(len(set(bi)),len(bi)),
      "trigram_unique":safe(len(set(tri)),len(tri)), "repeat_starts":safe(len(starts)-len(set(starts)),len(starts)),
      "function_ratio":safe(sum(x in FUNCTION_WORDS for x in w),len(w)),
      "contractions_100":100*safe(len(CONTRACTION_RE.findall(low)),len(w)),
      "transitions_100":100*safe(sum(low.count(x) for x in TRANSITIONS),len(w)),
      "compression":safe(len(zlib.compress(enc,9)),len(enc)), "short_sent":safe(sum(x<=8 for x in sl),len(sl)),
      "long_sent":safe(sum(x>=30 for x in sl),len(sl)), "digits":100*safe(sum(c.isdigit() for c in text),nc),
      "upper":100*safe(sum(c.isupper() for c in text),nc), "nonascii":100*safe(sum(ord(c)>127 for c in text),nc),
      "spaces":100*safe(sum(c.isspace() for c in text),nc),
      "list_rate":safe(len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",text)),ns),
    }
    for ch,name in [(".","period"),(",","comma"),(";","semicolon"),(":","colon"),("?","question"),
                    ("!","exclaim"),("(","paren"),('"',"quote"),("-","hyphen")]: f[name]=100*text.count(ch)/nw
    for x in ("i","we","you","he","she","they","it","this","these"): f["fw_"+x]=100*wc[x]/nw
    f.update(stats(wl,"wordlen")); f.update(stats(sl,"sentlen")); f.update(stats(pl,"paralen"))
    return f


def style_matrix(texts):
    print(f"Extracting style from {len(texts):,} texts ...", flush=True)
    return pd.DataFrame([style(x) for x in texts]).replace([np.inf,-np.inf],0).fillna(0).to_numpy(np.float32)


@dataclass
class Node:
    feature:int=-1; threshold:float=0.; left:object=None; right:object=None; value:float=0.


class RandomizedTree:
    def __init__(self, max_depth=12, min_leaf=5, max_features=.35, trials=2, seed=0):
        self.max_depth=max_depth; self.min_leaf=min_leaf; self.max_features=max_features
        self.trials=trials; self.rng=np.random.default_rng(seed)
    def fit(self,x,y): self.x=x; self.y=y; self.root=self._grow(np.arange(len(y)),0); return self
    def _grow(self,idx,depth):
        yy=self.y[idx]; node=Node(value=float(yy.mean()))
        if depth>=self.max_depth or len(idx)<2*self.min_leaf or yy.min()==yy.max(): return node
        p=self.x.shape[1]; fs=self.rng.choice(p,max(1,int(p*self.max_features)),replace=False); best=None
        for f in fs:
            vals=self.x[idx,f]; lo,hi=float(vals.min()),float(vals.max())
            if lo==hi: continue
            for t in self.rng.uniform(lo,hi,self.trials):
                mask=vals<=t; nl=int(mask.sum()); nr=len(idx)-nl
                if nl<self.min_leaf or nr<self.min_leaf: continue
                yl=yy[mask]; yr=yy[~mask]
                impurity=nl*(yl.mean()*(1-yl.mean()))+nr*(yr.mean()*(1-yr.mean()))
                if best is None or impurity<best[0]: best=(impurity,f,t,mask)
        if best is None:return node
        _,node.feature,node.threshold,mask=best
        node.left=self._grow(idx[mask],depth+1); node.right=self._grow(idx[~mask],depth+1); return node
    def predict(self,x):
        out=np.empty(len(x),np.float32)
        for i,row in enumerate(x):
            n=self.root
            while n.left is not None: n=n.left if row[n.feature]<=n.threshold else n.right
            out[i]=n.value
        return out


class ExtraTreesFromScratch:
    # Deliberately compact so a pure-Python implementation finishes on a laptop.
    def __init__(self,n_trees=60,min_leaf=5,seed=SEED): self.n_trees=n_trees; self.min_leaf=min_leaf; self.seed=seed
    def fit(self,x,y):
        self.trees=[]
        for i in range(self.n_trees):
            tree=RandomizedTree(min_leaf=self.min_leaf,seed=self.seed+i).fit(x,y); self.trees.append(tree)
            if (i+1)%25==0: print(f"  style trees {i+1}/{self.n_trees}",flush=True)
        return self
    def predict_proba(self,x): return np.mean([t.predict(x) for t in self.trees],axis=0)


def macro_f1(y,p):
    scores=[]; p=np.asarray(p,dtype=int); y=np.asarray(y,dtype=int)
    for c in (0,1):
        tp=np.sum((p==c)&(y==c)); fp=np.sum((p==c)&(y!=c)); fn=np.sum((p!=c)&(y==c))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return float(np.mean(scores))


def best_threshold(y,s):
    return max((macro_f1(y,s>=t),float(t)) for t in np.arange(.25,.751,.005))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=None); args=ap.parse_args(); root=args.project_root.resolve()
    output=(args.output or root/"submissions"/"NoSklearn_HighScore_Prediction.csv").resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(root/"data/train.csv",dtype={"id":"string"}); test=pd.read_csv(root/"data/test.csv",dtype={"id":"string"})
    split=pd.read_csv(root/"data/splits/shared_validation_split.csv",dtype={"id":"string"})
    assert train.id.tolist()==split.id.tolist(); tr=split.split.eq("train").to_numpy(); va=split.split.eq("validation").to_numpy()
    texts=train.text.fillna("").astype(str).tolist(); test_text=test.text.fillna("").astype(str).tolist(); y=train.label.to_numpy(np.int8)
    print(f"Rows: train={tr.sum()}, validation={va.sum()}, test={len(test)}")

    vec=TfidfFromScratch(); xtr=vec.fit_transform([texts[i] for i in np.flatnonzero(tr)])
    xva=vec.transform([texts[i] for i in np.flatnonzero(va)]); ratio=nb_ratio(xtr,y[tr])
    xtr=xtr.multiply(ratio).tocsr(); xva=xva.multiply(ratio).tocsr()
    linear=SparseLogisticAdam().fit(xtr,y[tr]); linear_val=linear.predict_proba(xva)
    lf,lt=best_threshold(y[va],linear_val); print(f"Linear validation macro-F1: {lf:.6f} at {lt:.3f}")

    all_style=style_matrix(texts+test_text); sx=all_style[:len(train)]; stest=all_style[len(train):]
    med=np.median(sx[tr],axis=0); scale=np.percentile(sx[tr],75,axis=0)-np.percentile(sx[tr],25,axis=0); scale[scale==0]=1
    sx=(sx-med)/scale; stest=(stest-med)/scale; sx=np.clip(sx,-20,20); stest=np.clip(stest,-20,20)
    tree=ExtraTreesFromScratch().fit(sx[tr],y[tr]); tree_val=tree.predict_proba(sx[va]); tf,tt=best_threshold(y[va],tree_val)
    print(f"Style-tree validation macro-F1: {tf:.6f} at {tt:.3f}")
    best=(-1,None,None)
    for w in np.arange(0,1.001,.025):
        blend=w*linear_val+(1-w)*tree_val; f,t=best_threshold(y[va],blend)
        if f>best[0]:best=(f,float(w),t)
    print(f"Blend validation macro-F1: {best[0]:.6f}; linear weight={best[1]:.3f}; threshold={best[2]:.3f}")

    print("Refitting final no-sklearn models on all labelled data ...",flush=True)
    vec=TfidfFromScratch(); xfull=vec.fit_transform(texts); xtest=vec.transform(test_text); ratio=nb_ratio(xfull,y)
    final_linear=SparseLogisticAdam().fit(xfull.multiply(ratio).tocsr(),y)
    linear_test=final_linear.predict_proba(xtest.multiply(ratio).tocsr())
    med=np.median(all_style[:len(train)],axis=0); scale=np.percentile(all_style[:len(train)],75,axis=0)-np.percentile(all_style[:len(train)],25,axis=0); scale[scale==0]=1
    sx=np.clip((all_style[:len(train)]-med)/scale,-20,20); st=np.clip((all_style[len(train):]-med)/scale,-20,20)
    final_tree=ExtraTreesFromScratch(n_trees=90).fit(sx,y); tree_test=final_tree.predict_proba(st)
    final_score=best[1]*linear_test+(1-best[1])*tree_test; pred=(final_score>=best[2]).astype(np.int8)
    sub=pd.DataFrame({"id":test.id.astype(str),"label":pred}); sub.to_csv(output,index=False)
    meta={"validation_macro_f1":best[0],"linear_weight":best[1],"style_weight":1-best[1],"threshold":best[2],"counts":sub.label.value_counts().sort_index().to_dict()}
    output.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    assert list(sub.columns)==["id","label"] and len(sub)==len(test) and sub.id.tolist()==test.id.astype(str).tolist() and sub.label.isin([0,1]).all()
    print(f"Saved verified submission: {output}\n{meta}")

if __name__=="__main__": main()
