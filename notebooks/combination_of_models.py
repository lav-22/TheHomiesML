"""High-scoring non-deep-learning detector for human (0) vs machine (1) text.

The model blends three complementary branches:
  1) word-level TF-IDF + LinearSVC
  2) character-level TF-IDF + LinearSVC
  3) hand-engineered stylometry + ExtraTrees

Model weights, SVM C values, class weights, and the classification threshold are
selected only on the supplied validation split. The selected models are then
refit on all labelled rows and used to create a Kaggle-ready id,label file.

Run from any directory:
  python combination_of_models.py --project-root PATH_TO_TheHomiesML
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?")
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
PARA_RE = re.compile(r"\n\s*\n+")
FUNCTION_WORDS = """a an the and but or nor for so yet of in on at by to from with without into
through during before after above below between under again further then once here there when where
why how all any both each few more most other some such no not only own same than too very can will
just should now is am are was were be been being have has had do does did this that these those i me
my mine we us our ours you your yours he him his she her hers it its they them their theirs who whom
whose which what because although however therefore moreover nevertheless thus hence while whereas""".split()
FUNCTION_WORDS = set(FUNCTION_WORDS)
TRANSITIONS = [
    "however", "therefore", "moreover", "furthermore", "in addition", "for example",
    "for instance", "on the other hand", "in conclusion", "overall", "additionally",
    "consequently", "nevertheless", "as a result", "in contrast", "firstly", "secondly",
]
CONTRACTIONS_RE = re.compile(r"\b(?:\w+n['’]t|i['’]m|i['’]ve|i['’]ll|you['’]re|we['’]re|they['’]re|it['’]s|that['’]s|\w+['’](?:d|ll|ve|re|s))\b", re.I)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def distribution_stats(values: list[int], prefix: str) -> dict[str, float]:
    if not values:
        values = [0]
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(a.std()),
        f"{prefix}_min": float(a.min()),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_cv": safe_div(a.std(), mean),
        f"{prefix}_range": float(a.max() - a.min()),
    }


def stylometry(text: str) -> dict[str, float]:
    text = "" if pd.isna(text) else str(text)
    lower = text.lower()
    words = WORD_RE.findall(lower)
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    paragraphs = [p.strip() for p in PARA_RE.split(text) if p.strip()]
    n_chars, n_words, n_sent = len(text), len(words), len(sentences)
    word_counts = Counter(words)
    sent_word_counts = [len(WORD_RE.findall(s)) for s in sentences]
    para_word_counts = [len(WORD_RE.findall(p)) for p in paragraphs]
    word_lengths = [len(w) for w in alpha_words]
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    sent_starts = [WORD_RE.findall(s.lower())[0] for s in sentences if WORD_RE.findall(s.lower())]
    adjacent_changes = [abs(a-b) for a, b in zip(sent_word_counts, sent_word_counts[1:])]
    encoded = text.encode("utf-8", errors="ignore")
    compressed = zlib.compress(encoded, level=9) if encoded else b""

    f: dict[str, float] = {
        "char_count": n_chars,
        "word_count": n_words,
        "sentence_count": n_sent,
        "paragraph_count": len(paragraphs),
        "line_count": text.count("\n") + 1,
        "unique_word_ratio": safe_div(len(word_counts), n_words),
        "hapax_ratio": safe_div(sum(v == 1 for v in word_counts.values()), len(word_counts)),
        "dislegomena_ratio": safe_div(sum(v == 2 for v in word_counts.values()), len(word_counts)),
        "word_entropy": entropy(words),
        "bigram_unique_ratio": safe_div(len(set(bigrams)), len(bigrams)),
        "trigram_unique_ratio": safe_div(len(set(trigrams)), len(trigrams)),
        "repeat_bigram_ratio": safe_div(len(bigrams) - len(set(bigrams)), len(bigrams)),
        "repeat_trigram_ratio": safe_div(len(trigrams) - len(set(trigrams)), len(trigrams)),
        "repeated_sentence_ratio": safe_div(n_sent - len(set(s.lower() for s in sentences)), n_sent),
        "repeated_start_ratio": safe_div(len(sent_starts) - len(set(sent_starts)), len(sent_starts)),
        "function_word_ratio": safe_div(sum(w in FUNCTION_WORDS for w in words), n_words),
        "stoplike_diversity": safe_div(len(set(words) & FUNCTION_WORDS), len(FUNCTION_WORDS)),
        "contraction_rate": safe_div(len(CONTRACTIONS_RE.findall(lower)), n_words) * 100,
        "digit_rate": safe_div(sum(c.isdigit() for c in text), n_chars) * 100,
        "uppercase_rate": safe_div(sum(c.isupper() for c in text), n_chars) * 100,
        "nonascii_rate": safe_div(sum(ord(c) > 127 for c in text), n_chars) * 100,
        "whitespace_rate": safe_div(sum(c.isspace() for c in text), n_chars) * 100,
        "punctuation_rate": safe_div(sum(not c.isalnum() and not c.isspace() for c in text), n_chars) * 100,
        "compression_ratio": safe_div(len(compressed), len(encoded)),
        "short_sentence_ratio": safe_div(sum(x <= 8 for x in sent_word_counts), n_sent),
        "long_sentence_ratio": safe_div(sum(x >= 30 for x in sent_word_counts), n_sent),
        "adjacent_sentence_change_mean": float(np.mean(adjacent_changes)) if adjacent_changes else 0.0,
        "adjacent_sentence_change_std": float(np.std(adjacent_changes)) if adjacent_changes else 0.0,
        "markdown_heading_rate": safe_div(len(re.findall(r"(?m)^\s*#{1,6}\s", text)), max(len(paragraphs), 1)),
        "list_item_rate": safe_div(len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)), max(n_sent, 1)),
        "transition_rate": safe_div(sum(lower.count(x) for x in TRANSITIONS), n_words) * 100,
    }
    for mark, name in [(".", "period"), (",", "comma"), (";", "semicolon"), (":", "colon"),
                       ("?", "question"), ("!", "exclamation"), ("(", "parenthesis"),
                       ('"', "quote"), ("-", "hyphen")]:
        f[f"{name}_per_100_words"] = safe_div(text.count(mark), n_words) * 100
    for pronoun in ["i", "we", "you", "he", "she", "they", "it", "this", "these"]:
        f[f"fw_{pronoun}"] = safe_div(word_counts[pronoun], n_words) * 100
    f.update(distribution_stats(word_lengths, "word_length"))
    f.update(distribution_stats(sent_word_counts, "sentence_length"))
    f.update(distribution_stats(para_word_counts, "paragraph_length"))
    return f


def stylometry_frame(texts: pd.Series) -> pd.DataFrame:
    print(f"Extracting stylometry for {len(texts):,} documents ...", flush=True)
    frame = pd.DataFrame([stylometry(x) for x in texts])
    return frame.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)


def macro_f1(y, score, threshold=0.5):
    return f1_score(y, np.asarray(score) >= threshold, average="macro")


def normalized_margin(train_margin, other_margin):
    """Map an SVM margin to a stable 0..1 score without label leakage."""
    scale = max(float(np.std(train_margin)), 1e-6)
    center = float(np.median(train_margin))
    return expit((np.asarray(other_margin) - center) / scale)


def tune_svm(vectorizer, train_text, y_train, val_text, y_val, name):
    print(f"\nFitting {name} vectorizer ...", flush=True)
    xtr = vectorizer.fit_transform(train_text)
    xva = vectorizer.transform(val_text)
    print(f"{name} matrix: {xtr.shape}, {xtr.nnz:,} nonzeros", flush=True)
    best = None
    for c in (0.35, 0.6, 1.0, 1.6, 2.5):
        for cw in (None, "balanced"):
            model = LinearSVC(C=c, class_weight=cw, random_state=SEED, max_iter=5000)
            model.fit(xtr, y_train)
            tr_margin = model.decision_function(xtr)
            va_score = normalized_margin(tr_margin, model.decision_function(xva))
            score = max(macro_f1(y_val, va_score, t) for t in np.arange(0.30, 0.701, 0.01))
            print(f"  {name}: C={c:g}, class_weight={cw}, best-threshold macro-F1={score:.6f}")
            candidate = (score, c, cw, va_score)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return {"score": best[0], "c": best[1], "class_weight": best[2], "val_score": best[3]}


def tune_blend(y, score_map):
    names = list(score_map)
    best = (-1.0, None, None)
    # Coarse-to-fine simplex search for 3 model weights and threshold.
    for w0 in np.arange(0.20, 0.81, 0.05):
        for w1 in np.arange(0.10, 0.71, 0.05):
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9 or w2 > 0.50:
                continue
            weights = np.array([w0, w1, max(0.0, w2)])
            blend = sum(weights[i] * score_map[n] for i, n in enumerate(names))
            for threshold in np.arange(0.35, 0.651, 0.005):
                score = macro_f1(y, blend, threshold)
                if score > best[0]:
                    best = (score, weights.copy(), float(threshold))
    return names, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output or (root / "submissions" / "Best_Classical_Ensemble_Prediction.csv")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "data" / "train.csv", dtype={"id": "string"})
    test = pd.read_csv(root / "data" / "test.csv", dtype={"id": "string"})
    split = pd.read_csv(root / "data" / "splits" / "shared_validation_split.csv")
    assert train["id"].astype(str).tolist() == split["id"].astype(str).tolist(), "Split IDs do not match train.csv"
    train_text = train["text"].fillna("").astype(str)
    test_text = test["text"].fillna("").astype(str)
    tr = split["split"].eq("train").to_numpy()
    va = split["split"].eq("validation").to_numpy()
    y = train["label"].astype(int).to_numpy()
    print(f"Rows: train={tr.sum():,}, validation={va.sum():,}, test={len(test):,}")

    word_cfg = dict(ngram_range=(1, 3), min_df=2, max_df=0.995, max_features=280_000,
                    sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
                    token_pattern=r"(?u)\b\w[\w'-]+\b")
    char_cfg = dict(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_features=320_000,
                    sublinear_tf=True, dtype=np.float32)
    word_result = tune_svm(TfidfVectorizer(**word_cfg), train_text[tr], y[tr], train_text[va], y[va], "word")
    char_result = tune_svm(TfidfVectorizer(**char_cfg), train_text[tr], y[tr], train_text[va], y[va], "char")

    all_stylo = stylometry_frame(pd.concat([train_text, test_text], ignore_index=True))
    stylo_train = all_stylo.iloc[:len(train)].to_numpy()
    stylo_test = all_stylo.iloc[len(train):].to_numpy()
    scaler = StandardScaler()
    xs_tr = scaler.fit_transform(stylo_train[tr])
    xs_va = scaler.transform(stylo_train[va])
    best_tree = None
    for leaf in (2, 4, 7, 12):
        model = ExtraTreesClassifier(n_estimators=700, min_samples_leaf=leaf, max_features=0.8,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
        model.fit(xs_tr, y[tr])
        pred = model.predict_proba(xs_va)[:, 1]
        score = macro_f1(y[va], pred)
        print(f"  stylometry ExtraTrees: min_samples_leaf={leaf}, macro-F1={score:.6f}")
        if best_tree is None or score > best_tree[0]:
            best_tree = (score, leaf, pred)

    score_map = {"word": word_result["val_score"], "char": char_result["val_score"], "stylometry": best_tree[2]}
    names, (blend_f1, weights, threshold) = tune_blend(y[va], score_map)
    print("\nSelected configuration")
    print(f"  word: C={word_result['c']}, class_weight={word_result['class_weight']}, val={word_result['score']:.6f}")
    print(f"  char: C={char_result['c']}, class_weight={char_result['class_weight']}, val={char_result['score']:.6f}")
    print(f"  stylometry: leaf={best_tree[1]}, val={best_tree[0]:.6f}")
    print(f"  blend: {dict(zip(names, weights.round(3)))}, threshold={threshold:.3f}, val={blend_f1:.6f}")

    test_scores = {}
    for name, cfg, result in [("word", word_cfg, word_result), ("char", char_cfg, char_result)]:
        print(f"\nRefitting {name} model on all labelled data ...", flush=True)
        vectorizer = TfidfVectorizer(**cfg)
        xfull = vectorizer.fit_transform(train_text)
        xtest = vectorizer.transform(test_text)
        model = LinearSVC(C=result["c"], class_weight=result["class_weight"], random_state=SEED, max_iter=5000)
        model.fit(xfull, y)
        test_scores[name] = normalized_margin(model.decision_function(xfull), model.decision_function(xtest))

    final_scaler = StandardScaler()
    xs_full = final_scaler.fit_transform(stylo_train)
    xs_test = final_scaler.transform(stylo_test)
    tree = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=best_tree[1], max_features=0.8,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    tree.fit(xs_full, y)
    test_scores["stylometry"] = tree.predict_proba(xs_test)[:, 1]
    final_score = sum(weights[i] * test_scores[n] for i, n in enumerate(names))
    predictions = (final_score >= threshold).astype(int)
    submission = pd.DataFrame({"id": test["id"].astype(str), "label": predictions})
    submission.to_csv(output, index=False)
    metadata = {
        "validation_macro_f1": blend_f1, "threshold": threshold,
        "weights": dict(zip(names, map(float, weights))),
        "word": {k: v for k, v in word_result.items() if k != "val_score"},
        "char": {k: v for k, v in char_result.items() if k != "val_score"},
        "stylometry_min_samples_leaf": best_tree[1],
        "prediction_counts": submission["label"].value_counts().sort_index().to_dict(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    assert submission.columns.tolist() == ["id", "label"]
    assert len(submission) == len(test) and submission["label"].isin([0, 1]).all()
    print(f"\nSaved verified submission: {output}")
    print(submission["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
