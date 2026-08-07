# 50.007 Machine Learning — Course Project Report

**Team: TheHomiesML**

> **To fill in before submission:** team member names, the Kaggle public
> leaderboard scores (Section 4.6), Section 7 (Individual Reflection) and
> Section 8 (Member Contribution). Everything else is complete.

---

## 1. Introduction and Task Understanding

### 1.1 The task

Given a piece of text, decide whether a human wrote it (label 0) or a language
model generated it (label 1). This is a binary text classification problem, and
submissions are scored by **Macro F1** — the unweighted mean of the F1 score for
each class.

The data is a sample from the COLING 2025 GenAIDetect workshop (Wang et al.,
2025): 20,000 training rows and 6,999 test rows. The course supplied both the
raw text and a pre-computed 5000-feature TF-IDF representation.

### 1.2 Why it is important

The value of a lot of written work — student assignments, product reviews, news
reporting, scientific abstracts — rests on the assumption that a person wrote
it. As generated text becomes fluent enough to pass unnoticed, that assumption
stops being safe, and being able to tell the two apart automatically becomes
the only practical check at scale.

### 1.3 Why it is hard

Three things make this harder than it looks:

1. **The signal is stylistic, not topical.** Both classes discuss the same
   subjects. What separates them is *how* the text is written, not what it is
   about, so a bag-of-words model is working against the grain of the problem.
2. **The classes overlap by design.** Machine text is trained to imitate human
   text. Any decision boundary will cut through genuinely ambiguous documents.
3. **The sample is small and the domains are mixed.** This is under 5% of the
   original dataset, and the brief warns explicitly that training scores will
   run well ahead of test scores. Our own results bear this out, and we treated
   it as a reason to stop tuning early rather than push validation decimals
   that would not transfer.

Because the metric is Macro F1 and the training set is 62.5% class 1, a model
that quietly favours the majority class is punished. Handling that imbalance is
a recurring theme below.

---

## 2. Dataset Understanding

### 2.1 Class distribution

| Label | Meaning | Count | Percentage |
|---|---|---|---|
| 0 | Human-authored | 7,496 | 37.48% |
| 1 | Machine-generated | 12,504 | 62.52% |

The imbalance is moderate but it matters. A model that always predicts class 1
gets 62.5% accuracy but only **0.3847** Macro F1, because the human class scores
zero. That is the floor every model has to clear, and it is the reason we use
Macro F1 rather than accuracy throughout.

### 2.2 Validation protocol

All five of us scored against **one shared 80/20 stratified split**
(`data/splits/shared_validation_split.csv`, `random_state=42`): 16,000 training
rows and 4,000 validation rows, both at a 62.52% class-1 rate. Fixing the split
up front is what makes the numbers in Section 4 comparable — otherwise each
member's "0.74" would have meant something slightly different.

Two of us derived the split independently (one from `train.csv`, one from
`train_features.csv`) and we verified the two are row-for-row identical before
merging results.

`random_state=42` is set everywhere for reproducibility.

### 2.3 What the raw text looks like

Before any modelling, the two classes already differ in shape. Document length
in words:

| Label | Mean | Median | Std |
|---|---|---|---|
| 0 (human) | 261.5 | 176 | 307.2 |
| 1 (machine) | 237.6 | 195 | 186.7 |

The interesting part is not the averages — it is that human text has a *higher*
mean but a *lower* median, with a standard deviation 1.6× larger. Human writing
is heavily right-skewed: mostly short pieces with a long tail of very long ones.
Machine text clusters far more tightly around its middle.

The same pattern holds within documents. Measuring the standard deviation of
sentence length inside each document (sampled 2,000 per class) gives 8.86 for
human text against 7.82 for machine text — human writing varies its rhythm more.

So the discriminative signal is in the **variability**, not the level. A plain
bag-of-words model cannot see variability at all, which is exactly why the
provided TF-IDF features stall around 0.75. This observation is what motivated
the stylometry features in Section 3.2, and it turned out to be the most
valuable thing we noticed in the whole project.

---

## 3. Data Preprocessing and Feature Engineering

### 3.1 The provided features

For Tasks 1 and 2 we used the course's 5000 TF-IDF features exactly as given,
as required. No resampling, no class rebalancing at the data level — we handled
imbalance inside the models instead (via class weights and the decision
threshold), which keeps all 20,000 rows in play.

### 3.2 Our own features (Task 3 only)

Every model we ran on the provided features landed between 0.67 and 0.76 Macro
F1. That ceiling looked more like a feature problem than a model problem, so for
Task 3 we built our own representation from the raw text. Three parts:

**(a) Word + character TF-IDF.** Two vectorizers, concatenated:

- word 1–2 grams (`min_df=2`, `max_df=0.98`) — vocabulary and short phrases;
- character 3–5 grams inside word boundaries (`char_wb`) — spelling,
  morphology, punctuation habits, and the general *texture* of the writing.

Character n-grams are the important half. They pick up how a text is written
rather than what it says, which is exactly the signal this task turns on. Each
block is L2-normalised separately before being concatenated, so neither
dominates on length alone. Together they give ~200,000 features.

We implemented the vectorizer ourselves: document frequencies and the
vocabulary, then
idf(t) = log((1+N)/(1+df(t))) + 1, sublinear term frequency 1 + log(tf), and an
L2 row norm.

**(b) 78 stylometry features.** Hand-built measures of *how* a text is written:

| Group | Examples |
|---|---|
| Sentence rhythm | mean/std/median sentence length, coefficient of variation, IQR, share of very short and very long sentences |
| Vocabulary richness | type-token ratio, Guiraud and Herdan indices, hapax and dis legomena rates, unigram entropy |
| Repetition | repeated bigram/trigram rates, repeated sentence-opening rate |
| Function words | rates for 10 groups (pronouns by person, articles, conjunctions, prepositions, demonstratives, modals, negations, discourse transitions) |
| Punctuation & format | rates for `, ; : ? ! - ( ) " '`, uppercase, digits, newlines, bullet lists, citation patterns, acronyms |
| Readability | syllables per word, Flesch Reading Ease, Flesch-Kincaid, Gunning Fog, complex-word ratio |

The unifying idea is **burstiness**: human writing varies its rhythm far more
than generated text, which tends toward a steady middle. Several of these
features measure that variation directly.

The 78 columns are standardised, then scaled by a `style_weight` before being
concatenated onto the ~200,000 TF-IDF columns — without it they are simply
drowned out. We tuned that weight (Section 4.5).

**(c) Nothing else.** We deliberately did not add embeddings, language-model
perplexity, or any deep-learning feature — the brief forbids deep learning, and
we wanted the whole pipeline to stay explainable.

---

## 4. Models Explored and Results

All scores are **Macro F1 on the shared validation split**. "Tuned" means the
decision threshold was also chosen on validation.

### 4.1 Task 1 — Logistic Regression (from scratch)

**How it works.** A linear score z = wᵀx + b is squashed by the sigmoid
σ(z) = 1/(1+e⁻ᶻ) into a probability. Training minimises the log loss

L = −(1/m) Σ [ yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ) ]

whose gradients are ∂L/∂w = (1/m)Xᵀ(ŷ−y) and ∂L/∂b = (1/m)Σ(ŷᵢ−yᵢ). We follow
those downhill in mini-batches. Implemented with NumPy only — `sigmoid`,
`loss`, `gradients`, `train`, `predict` — with no logistic regression package
anywhere in the notebook.

**Roadmap.** The first version trained on fixed, unshuffled mini-batches with no
regularisation and reached 0.7352. We then tested three changes:

| Configuration | Val Macro F1 | With tuned threshold |
|---|---|---|
| Baseline (no shuffle, no L2), lr=1.0, 300 epochs, bs=64 | 0.7352 | 0.7376 |
| + shuffle each epoch | 0.7254 | 0.7386 |
| + shuffle, L2 = 1e-5 | 0.7225 | 0.7394 |
| + shuffle, L2 = 1e-5, lr = 0.5 | 0.7227 | **0.7440** |
| + shuffle, L2 = 1e-5, class_weight = balanced | 0.7353 | 0.7410 |
| + shuffle, L2 = 1e-5, 500 epochs | 0.7374 | 0.7391 |
| + shuffle, L2 = 1e-3 | 0.4005 | 0.6943 |

Before this, a wider sweep over learning rates showed the real problem. At
lr = 0.001 the model returned exactly 0.3847 — the always-predict-1 score — for
every combination of epochs and batch size we tried. At lr = 0.01 it barely
moved, reaching 0.4937 at best (500 epochs, batch 64). Only from lr = 0.1
upwards did it train properly, peaking around lr = 0.5–1.0. The TF-IDF values
are small in magnitude, so the gradients are small and the model needs an
unusually large learning rate. That was the first genuinely surprising thing we
hit.

**What actually helped.** Only the decision threshold, and only by about +0.01.
Shuffling and light L2 are individually within noise; heavy L2 (≥1e-3) destroys
the model. We report the selection honestly: the spread across the top six
configurations is roughly 0.005, which is inside the noise of a 4,000-row
validation set, so the ranking among the leaders should not be over-read.

**Final:** lr = 0.5, 300 epochs, bs = 64, L2 = 1e-5, threshold 0.627 →
**0.7440** validation. Refitted on all 20,000 rows for
`submissions/LogReg_Prediction.csv`.

### 4.2 Task 2 — PCA + KNN (`n_neighbors=2`)

| Components | Cumulative explained variance | Val Macro F1 | Predicted class-1 rate |
|---|---|---|---|
| 2000 | 78.2% | 0.4007 | 0.083 |
| 1000 | 57.5% | 0.4753 | 0.151 |
| 500 | 39.7% | 0.5603 | 0.271 |
| 100 | 16.0% | **0.6556** | 0.448 |

**The result runs backwards, and that is the interesting part.** Keeping *more*
variance makes the classifier *worse*. Two effects explain it, and we verified
both.

**(a) Distance concentration.** KNN only works if some neighbours are
meaningfully closer than others. As dimensionality grows, the distances from a
point to all its neighbours converge, and "nearest" stops carrying information.
Cutting to 100 components is what makes the neighbourhoods meaningful again —
even though 100 components retain only 16% of the variance. Variance retained
and usefulness for classification are simply not the same quantity: the
discarded directions were mostly noise that was hurting the distance metric.

**(b) An even `k` forces a tie-break that is not neutral.** With `n_neighbors=2`
the two neighbours frequently disagree, and sklearn settles a tie by taking the
lowest class index — class 0. We measured it:

| Components | Tie rate | Of tied rows, share truly class 1 | F1 at k=2 | F1 at k=1 |
|---|---|---|---|---|
| 2000 | 23.1% | 73.8% | 0.4007 | 0.4944 |
| 100 | 35.4% | 54.8% | 0.6556 | 0.6679 |

At 2000 components, 23% of validation rows are ties and 74% of them are
genuinely class 1 — every one is handed to class 0 by the tie rule. That is why
the model predicts class 1 for only 8.3% of rows when the true rate is 62.5%.
Dropping to `k=1`, which removes the tie-break entirely, recovers about **0.09**
Macro F1 at 2000 components.

We kept `k=2` because the brief fixes it, but it is the single largest thing
holding this model back — a far bigger effect than the choice of component
count. An odd `k` would remove the problem outright.

### 4.3 Task 3 — models on the provided features

Everything here uses the same 5000 TF-IDF features and the same split, so the
comparison isolates the model.

| Model | Implementation | Best setting | Val Macro F1 | Tuned |
|---|---|---|---|---|
| Linear classifier by SGD | from scratch | modified Huber, α=1e-4, balanced | **0.7446** | 0.7461 |
| Extra Trees | sklearn (ensemble — permitted) | 800 trees, max_features=sqrt | 0.7152 | **0.7621** |
| Soft vote (linear + Extra Trees + NB) | ensemble | rank-averaged | 0.7271 | 0.7462 |
| Logistic Regression | from scratch (Task 1) | lr=0.5, L2=1e-5 | 0.7227 | 0.7440 |
| XGBoost | library (ensemble — permitted) | 300 trees, depth 6, lr=0.1 | 0.7273 | — |
| Multinomial Naive Bayes | from scratch | α=0.001 | 0.6741 | 0.6819 |
| Complement Naive Bayes | from scratch | α=2.0 | 0.6669 | 0.6740 |

> The XGBoost row was produced on a teammate's machine; `xgboost` is not
> installed in the environment used for the final run, so unlike every other row
> in this table it was not re-verified. It is reported for completeness only and
> is not part of the final model.

**Naive Bayes (from scratch).** Applies Bayes' rule assuming features are
independent given the class. Tuning `alpha` (Laplace smoothing) showed smaller is
better — 0.6741 at α=0.001 falling to 0.6605 at α=1.0. The reason is that the
provided features are TF-IDF *weights*, not raw counts, so the per-class mass in
each feature is small and a smoothing constant of 1.0 is large relative to the
real signal. Complement NB, which we expected to do better on imbalanced data,
did not (0.6669); it prefers the opposite end of the alpha range (α=2.0).

**Linear classifier by SGD (from scratch).** One trainer covering three losses,
so switching between them is a one-line change:

| Loss | Val Macro F1 |
|---|---|
| hinge (= linear SVM) | 0.6995 |
| modified Huber | **0.7446** |
| log loss | 0.6963 |

Modified Huber won clearly, by about 0.045 over plain hinge. It is a smoothed
hinge — quadratic near the margin and linear far below it — so it is less thrown
by outliers than plain hinge while still producing a margin. Regularisation
strength barely mattered (α=1e-5 → 0.7417, 1e-4 → 0.7446, 1e-3 → 0.7395). All
runs used `class_weight="balanced"`, which is the principled way to handle the
class split when the metric weights both classes equally.

**Extra Trees (library ensemble, permitted).** Many decision trees, each on a
random feature subset with randomly chosen split points, votes averaged. The
extra randomness makes each tree weak but decorrelates them, and averaging
decorrelated weak learners is what makes the ensemble strong. Tuning:

| Setting changed | Value | Train F1 | Val F1 | Tuned |
|---|---|---|---|---|
| baseline | 200 trees, sqrt | 1.000 | 0.7147 | 0.7578 |
| n_estimators | 500 | 1.000 | 0.7189 | 0.7598 |
| n_estimators | 800 | 1.000 | 0.7152 | **0.7621** |
| max_depth | 20 | 0.533 | 0.4634 | 0.7007 |
| max_depth | 40 | 0.831 | 0.5466 | 0.7292 |
| max_features | log2 | 1.000 | 0.6504 | 0.7501 |
| max_features | 100 | 1.000 | 0.7172 | 0.7600 |
| class_weight | balanced | 1.000 | 0.7154 | 0.7486 |

Train F1 is 1.000 against validation near 0.72 — the trees memorise the training
set completely. Capping `max_depth` reduced that gap but made validation much
*worse*, so we left the trees unrestricted and accepted the overfitting: it is
the ensemble average, not any single tree, that generalises.

Extra Trees is also where the decision threshold pays most (+0.047, versus
+0.002 for the linear classifier), because averaged tree votes are poorly
calibrated probabilities and 0.5 is far from the best operating point.

**Soft-vote ensemble.** The three models' scores live on different scales (a
probability, a signed margin, a log-probability), so we converted each to a rank
in [0,1] before averaging. The vote landed *between* its members (0.7462 tuned)
rather than above the best of them. Averaging helps when members are strong and
err differently; here Naive Bayes is far weaker than the other two, so it drags
the average down more than its diversity adds back. A negative result, but a
clear one.

### 4.4 Task 3 — changing the features instead

Same from-scratch trainer, our own features from Section 3.2:

| Features | Val Macro F1 |
|---|---|
| Provided 5000 TF-IDF | 0.7446 |
| Word + char TF-IDF (~200k) + 78 stylometry | **0.8491** |

**This is the finding of the project.** Swapping the input lifted the *same*
model by about 0.105 — far more than any model change, hyperparameter, or
ensemble achieved. The model was never the bottleneck; the representation was.

### 4.5 Tuning the final model

**The question we nearly forgot to re-ask.** Section 4.3 established that
modified Huber clearly beat hinge on the provided features. We then built the
hybrid pipeline using an averaged **hinge**-loss SGD and never re-tested that
choice against the new features. When we finally did:

| Loss (hybrid features, lr=0.5, 60 epochs) | Val Macro F1 |
|---|---|
| hinge | 0.8104 |
| log loss | 0.7824 |
| **modified Huber** | **0.8491** |
| *(previous final model: averaged hinge SGD)* | *0.8289* |

That single change was worth about **+0.020** — more than every other Task 3
tuning decision combined. The lesson is general: a hyperparameter validated on
one representation does not automatically carry to another.

**Style weight and training length.** With modified Huber fixed:

| `style_weight` | Epochs | Val Macro F1 | With tuned threshold |
|---|---|---|---|
| 0.10 | 60 | **0.8491** | 0.8498 |
| 0.05 | 100 | 0.8484 | 0.8481 |
| 0.10 | 100 | 0.8477 | 0.8503 |
| 0.05 | 60 | 0.8421 | 0.8431 |
| 0.10 | 30 | 0.8420 | 0.8431 |
| 0.05 | 30 | 0.8374 | 0.8367 |

We checked that 60 epochs is a plateau rather than the edge of the grid by
extending to 200 (0.8491 / 0.8477 / 0.8485 / 0.8487 at 60 / 100 / 150 / 200) —
flat within 0.002, so we took the cheapest point on the plateau.

**On the threshold.** Tuning it gains only +0.0007 here, against +0.047 for
Extra Trees. We therefore **do not** apply it: our selection rule adopts a tuned
threshold only when it beats the default by more than 0.005, on the grounds that
a cut-off fitted to 4,000 validation rows is as likely to be noise as signal.
The margin-based losses already put their natural cut-off at 0.

**What did not work.** Averaging three random seeds gave no improvement
(0.8293 vs 0.8305 on the earlier hinge pipeline), because the optimiser already
averages its weights across epochs, so there was little seed variance left to
remove. We dropped it rather than pay three times the compute for noise.

**Final model:** hybrid word + character TF-IDF + 78 stylometry features
(`style_weight=0.10`), linear classifier trained by mini-batch SGD with modified
Huber loss (lr=0.5, α=1e-5, 60 epochs, batch 256, balanced class weights, L2
penalty), decision threshold 0. Validation Macro F1 **0.8491**. Refitted on all
20,000 labelled rows → `submissions/Final_Prediction.csv` (class-1 rate 0.5954,
against a training rate of 0.6252).

### 4.6 Kaggle public leaderboard scores

> **To fill in after submitting.** Per the FAQ, Tasks 1 and 2 need five
> submissions and the public leaderboard score is what should be reported.

| Submission | File | Validation Macro F1 | Kaggle public LB |
|---|---|---|---|
| Task 1 — Logistic Regression | `LogReg_Prediction.csv` | 0.7440 | *TBC* |
| Task 2 — PCA 2000 + KNN | `PCA2000_KNN_Prediction.csv` | 0.4007 | *TBC* |
| Task 2 — PCA 1000 + KNN | `PCA1000_KNN_Prediction.csv` | 0.4753 | *TBC* |
| Task 2 — PCA 500 + KNN | `PCA500_KNN_Prediction.csv` | 0.5603 | *TBC* |
| Task 2 — PCA 100 + KNN | `PCA100_KNN_Prediction.csv` | 0.6556 | *TBC* |
| Task 3 — final model | `Final_Prediction.csv` | 0.8491 | *TBC* |

---

## 5. Discussion

### 5.1 Difficulties and what we did about them

**The learning rate that looked like a broken model.** Our first logistic
regression runs returned exactly 0.3847 over and over — the always-predict-1
score. We assumed a bug in the gradient. It was not: TF-IDF values are small, so
gradients are small, and learning rates in the usual 0.001–0.1 range moved the
weights almost not at all in 500 epochs. Raising the rate to 0.5–1.0 fixed it.
The lesson we took is to check the *scale* of the features against the step size
before going looking for a bug.

**A stale results file that would have gone into this report.** Our best model
was recorded at 0.8477 in a committed results CSV. When we re-ran it, it scored
0.8305. The cause: the pipeline had been rewritten to be fully from-scratch
(replacing sklearn's `TfidfVectorizer`, `SGDClassifier` and `StandardScaler`),
but the results file was never regenerated, so it still held numbers from the
older sklearn version. We re-ran every experiment from the current code and this
report contains only re-verified numbers. The genuine lesson: a results file is
only as trustworthy as the last time the code that produced it was actually run.

**Work scattered across five branches.** We each worked on a separate branch and
the results drifted out of sync — different splits derived independently,
different scripts writing to the same filenames, and no single place showing all
models side by side. We fixed this by agreeing one shared validation split,
verifying the independently derived splits matched row-for-row, and consolidating
everything into one notebook with a single comparison table.

**A settled decision that quietly went stale.** We compared three losses on the
provided features, picked modified Huber, and moved on. When we later built the
hybrid TF-IDF + stylometry pipeline we wrote it with hinge loss and never
re-ran that comparison — the choice felt already made. Re-testing it at the end
was worth about +0.02 Macro F1, more than every other Task 3 tuning decision
combined. What we take from this is that a hyperparameter is only validated
*against the features it was tuned on*; changing the representation invalidates
the tuning done on the old one. We now re-run the cheap comparisons whenever the
input changes.

**Knowing when to stop.** We had a submission that forced the test-set class-1
rate to 55% by thresholding on a quantile of the *test* scores. It scored well on
validation-by-proxy, but it is fitting to the test distribution rather than
learning from the data, and it would be a poor bet on the private leaderboard.
We removed it. In the same spirit we kept the threshold grid coarse, and set a
rule that a tuned threshold is only adopted when it beats the default by more
than 0.005 — for the final model it does not, so we ship the plain cut-off.

### 5.2 Limitations

1. **The dataset is small and mixed.** Under 5% of the original corpus, across
   several domains. The brief warns test scores will trail training scores, and
   our train-vs-validation gaps (Extra Trees at 1.000 vs 0.72) confirm it. More
   data would help more than any further tuning.
2. **A single validation split.** Every number rests on one 4,000-row split, so
   differences below roughly 0.01 are not meaningful. Five-fold cross-validation
   would let us distinguish real gains from noise; we skipped it because the
   from-scratch pipeline takes a couple of minutes per fit and we chose to spend
   that budget on features instead. This is the change we would make first.
3. **Hyperparameters selected on the same split used to report scores.** We
   mitigated this by keeping grids coarse and by declining the tuned threshold
   for the final model, but a nested split would be cleaner and is the honest
   fix.
4. **Stylometry is English-specific and partly hand-tuned.** The readability
   formulas, function-word lists and syllable heuristic assume English prose and
   would not transfer to another language.
5. **`n_neighbors=2` cripples Task 2.** Documented in Section 4.2; fixed by the
   brief, so we could not change it.

### 5.3 What we would do next

Cross-validation to put error bars on everything; calibrating the Extra Trees
probabilities so the ensemble members combine on a common scale; and feature
selection over the 200,000 TF-IDF columns, which are certainly redundant.

---

## 6. Conclusion

We built a GenAI content detector reaching **0.8491** Macro F1 on our held-out
validation split: a linear classifier trained by mini-batch SGD with modified
Huber loss, on a combination of word-level TF-IDF, character-level TF-IDF, and
78 hand-built stylometry features — all implemented from scratch with NumPy.

The single most useful thing we learned is that **the representation mattered far
more than the model**. Six different algorithms on the provided 5000 TF-IDF
features all landed between 0.67 and 0.76, a spread of under 0.09 across model
families as different as Naive Bayes and random forests. Changing the *features*
and keeping the model fixed moved us from 0.7446 to 0.8491 on its own. Time spent
looking at the data was worth more than time spent tuning.

Three smaller findings we did not expect. Dimension reduction helped KNN *more*
the more variance it threw away. The decision threshold — an afterthought in most
of our first drafts — was worth more than most hyperparameters for the tree
ensemble, though almost nothing for the final margin-based model. And a
hyperparameter we had settled early (the loss function) turned out to be wrong
once the features changed underneath it, which cost us 0.02 Macro F1 until we
thought to re-test it.

---

## 7. Individual Reflection

> **To fill in — one short subsection per member.** Cover (a) what you
> self-learned beyond the course, and (b) the difficulties *you personally*
> faced. Candidates from this project, if useful:
>
> - **Beyond the course:** stylometry and authorship attribution as a field;
>   character n-grams (`char_wb`) and why they capture style; averaged SGD as a
>   variance-reduction technique; Complement Naive Bayes; why the
>   curse of dimensionality specifically breaks distance-based methods;
>   threshold selection as a decision-theory problem separate from model fitting;
>   Extra Trees vs Random Forests.
> - **Difficulties:** the learning-rate/feature-scale interaction; keeping five
>   branches consistent; trusting a stale results file; deciding when to stop
>   tuning.

### Member A — *name*

*TBC*

### Member B — *name*

*TBC*

*(repeat per member)*

---

## 8. Member Contribution

> **To fill in.** Example format below — replace with your actual split.

| Team Member | Role Description |
|---|---|
| Member A | Task 1 logistic regression from scratch, hyperparameter tuning |
| Member B | Task 2 PCA + KNN, component and tie-break analysis |
| Member C | Task 3 Naive Bayes and Complement Naive Bayes from scratch |
| Member D | Dataset analysis, shared validation split, Linear SVM / SGD from scratch |
| Member E | Task 3 feature engineering (hybrid TF-IDF + stylometry), Extra Trees, final model |

---

## Appendix A — Reproducing the results

```bash
# One notebook covering Tasks 1-3 (this is the graded deliverable)
jupyter-nbconvert --to notebook --execute --inplace \
    notebooks/TheHomiesML_Final_Submission.ipynb

# Or the individual experiment drivers
python3 scripts/task1_logreg_tuning.py        # Task 1 tuning grid
python3 scripts/task2_pca_knn.py              # Task 2 PCA + KNN, 4 submissions
python3 scripts/task3_model_comparison.py     # Task 3 all models, one table
python3 scripts/task3_extra_trees_tuning.py   # Task 3 Extra Trees tuning
python3 scripts/task3_loss_on_hybrid.py       # Task 3 loss re-test on our features
python3 scripts/task3_final_model.py          # Task 3 final model + submission
python3 scripts/verify_submissions.py         # format-check every submission
```

`random_state=42` throughout. `data/train.csv` and `data/test.csv` are
gitignored because of their size and must be downloaded from Kaggle.

## Appendix B — Submission files

| File | Task | Model |
|---|---|---|
| `LogReg_Prediction.csv` | 1 | Logistic regression from scratch |
| `PCA2000_KNN_Prediction.csv` | 2 | PCA 2000 components + KNN (k=2) |
| `PCA1000_KNN_Prediction.csv` | 2 | PCA 1000 components + KNN (k=2) |
| `PCA500_KNN_Prediction.csv` | 2 | PCA 500 components + KNN (k=2) |
| `PCA100_KNN_Prediction.csv` | 2 | PCA 100 components + KNN (k=2) |
| `Final_Prediction.csv` | 3 | **Final model** — hybrid TF-IDF + stylometry + linear SVM |

These are the six files the brief asks for: one for Task 1, four for Task 2, and
the final Task 3 submission. The other Task 3 models in Section 4.3 were
compared on validation only and did not need their own submission.

## Appendix C — Declaration of AI assistance

> **To confirm and edit.** The competition rules require declaring the use of AI
> assistants. Describe honestly what was used and for what — for example code
> review and refactoring, consolidating results across branches, and drafting
> this report — and confirm that the team understands and can explain every
> model and design decision presented here.

## References

Wang, Y., Shelmanov, A., Mansurov, J., Tsvigun, A., Mikhailov, V., Xing, R., …
& Nakov, P. (2025). GenAI content detection task 1: English and multilingual
machine-generated text detection: AI vs. human. *Proceedings of the 1st Workshop
on GenAI Content Detection (GenAIDetect)*, 244–261.
