"""Assemble the single submission notebook for Tasks 1-3.

The from-scratch implementations are embedded straight from src/ so the
notebook and the modules used by the other scripts can never drift apart.

Run from the repository root:

    python3 scripts/build_final_notebook.py
    jupyter-nbconvert --to notebook --execute --inplace \
        notebooks/TheHomiesML_Final_Submission.ipynb
"""

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "TheHomiesML_Final_Submission.ipynb"

SEPARATOR = re.compile(r"^# -{70,}$", re.M)


def sections(path):
    """Split a source file into chunks on its `# ---- ... ----` separators.

    The separators come in pairs around each heading comment, so the chunks
    alternate: a leading import block, then heading, code, heading, code...
    """
    text = (ROOT / path).read_text()
    text = text.split('"""', 2)[2].lstrip("\n")     # drop the module docstring
    chunks = [c.strip("\n") for c in SEPARATOR.split(text)]
    return [c for c in chunks if c.strip()]


def source_blocks(path):
    """Map each heading to its code block, keyed by a lowercase slug.

    Keying by name rather than position means reordering the source file
    cannot silently put the wrong implementation under the wrong heading.
    """
    chunks = sections(path)
    preamble = chunks[0]                            # the module's own imports
    blocks = {}
    for heading, body in zip(chunks[1::2], chunks[2::2]):
        title = heading.split("\n")[0].lstrip("#").strip()
        blocks[title.lower()] = body
    return preamble, blocks


PREAMBLE, BLOCKS = source_blocks("src/scratch_models.py")


def text_feature_block(first, last=None):
    """Slice src/text_features.py from one top-level definition up to another.

    `last` is exclusive; omitting it runs to the end of the file.
    """
    text = (ROOT / "src/text_features.py").read_text()
    start = text.index(f"\n{first}")
    end = text.index(f"\n{last}") if last else len(text)
    return text[start:end].strip("\n")


def block(name):
    """Fetch one implementation block by heading, failing loudly if renamed."""
    key = name.lower()
    if key not in BLOCKS:
        raise KeyError(f"{name!r} not found in src/scratch_models.py; "
                       f"available: {sorted(BLOCKS)}")
    return BLOCKS[key]


cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ============================================================================
md("""
# 50.007 Machine Learning — Course Project
## GenAI Content Detection: human-authored vs machine-generated text

**Team: TheHomiesML**

This notebook contains our code for Tasks 1 to 3.

| Section | Task | Models |
|---|---|---|
| 0 | Setup and dataset understanding | — |
| 1 | Logistic Regression **from scratch** | Logistic Regression |
| 2 | PCA + KNN (`n_neighbors=2`) | KNN on 2000 / 1000 / 500 / 100 components |
| 3 | Other models, race to the top | Naive Bayes, Complement NB, linear classifier by SGD, Extra Trees, soft vote, hybrid TF-IDF + stylometry |

**On the "from scratch" rule.** Task 1 and Task 3 models are written out in
this notebook using NumPy only — no sklearn estimator is used to learn
anything. sklearn appears in three permitted places: PCA and KNN in Task 2
(the brief allows it), Extra Trees in Task 3 (the brief allows libraries for
ensemble models), and `f1_score` as the evaluation metric.

Every experiment uses one fixed train/validation split, shared across the
team, and `random_state=42` throughout.
""")

# ---------------------------------------------------------------- Section 0
md("""
---
## 0. Setup and dataset understanding
""")

code("""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import calculate_macro_f1
from src.submission import create_submission

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
for directory in (RESULTS_DIR, SUBMISSIONS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ID_COLUMN = "id"
LABEL_COLUMN = "label"

print("data files present:",
      all((DATA_DIR / name).exists() for name in
          ["train.csv", "test.csv", "train_features.csv", "test_features.csv"]))
""")

md("""
### 0.1 The raw data

`train.csv` and `test.csv` hold the original text. `train_features.csv` and
`test_features.csv` hold the 5000 TF-IDF features the course pre-computed for
us, which Tasks 1 and 2 are required to use.
""")

code("""
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")

print("train:", train_raw.shape, " test:", test_raw.shape)
print("missing values:", train_raw.isnull().sum().sum(), test_raw.isnull().sum().sum())
train_raw.head(3)
""")

md("""
### 0.2 Class distribution

The classes are not balanced: roughly 62.5% of the training rows are
machine-generated. Because the competition scores **Macro F1**, both classes
count equally regardless of how many rows they have, so this imbalance is
something the models have to be told about rather than something we can ignore.
""")

code("""
class_distribution = train_raw[LABEL_COLUMN].value_counts().sort_index()
display(pd.DataFrame({
    "Label": ["0 (human-authored)", "1 (machine-generated)"],
    "Count": class_distribution.values,
    "Percentage": (class_distribution.values / len(train_raw) * 100).round(2),
}))

# A model that always guesses the majority class sets the bar to clear
majority = np.ones(len(train_raw), dtype=int)
print("always-predict-1 Macro F1:",
      round(calculate_macro_f1(train_raw[LABEL_COLUMN], majority), 4))
""")

md("""
### 0.3 Text length by class

A quick look at what separates the two classes before any modelling.
""")

code("""
import re

lengths = train_raw.assign(
    n_words=train_raw["text"].str.split().str.len(),
    n_chars=train_raw["text"].str.len(),
)
display(lengths.groupby(LABEL_COLUMN)[["n_words", "n_chars"]]
        .agg(["mean", "median", "std"]).round(1))


def sentence_length_std(text):
    \"\"\"How much sentence length varies inside one document.\"\"\"
    counts = [len(re.findall(r"\\b\\w+\\b", part))
              for part in re.split(r"[.!?]+", text) if part.strip()]
    return np.std(counts) if len(counts) > 1 else 0.0


sample = train_raw.groupby(LABEL_COLUMN, group_keys=False).sample(2000, random_state=RANDOM_SEED)
display(sample.assign(sentence_std=sample["text"].map(sentence_length_std))
        .groupby(LABEL_COLUMN)["sentence_std"].agg(["mean", "median"]).round(2))
""")

md("""
The averages are not the story. Human text has a **higher mean** but a **lower
median** length, with a standard deviation about 1.6x larger — human writing is
right-skewed, mostly short with a long tail, while machine text clusters tightly
around its middle. The same holds inside documents: sentence length varies more
in human writing (8.86 vs 7.82).

The signal is in the **variability**, not the level — and a bag-of-words model
cannot see variability at all. That is the hint we follow up in Section 3.6.
""")

md("""
### 0.4 The shared validation split

Every member scored their models against the same 80/20 stratified split, so
the numbers in this notebook are directly comparable with each other. The split
is stored in `data/splits/shared_validation_split.csv`.
""")

code("""
split = pd.read_csv(DATA_DIR / "splits" / "shared_validation_split.csv")
assert len(split) == len(train_raw)
assert split[ID_COLUMN].astype("string").tolist() == train_raw[ID_COLUMN].astype("string").tolist()

is_train = (split["split"] == "train").to_numpy()
print("train rows:", is_train.sum(), " validation rows:", (~is_train).sum())
print("class-1 rate — train:", round(train_raw.loc[is_train, LABEL_COLUMN].mean(), 4),
      " validation:", round(train_raw.loc[~is_train, LABEL_COLUMN].mean(), 4))
""")

md("""
### 0.5 Loading the provided TF-IDF features

Tasks 1 and 2 must use these. Task 3 starts here too, so that the model
comparison is about the models rather than about the features; we only build
our own features later in Section 3.6.
""")

code("""
FEATURES = [c for c in pd.read_csv(DATA_DIR / "train_features.csv", nrows=0).columns
            if c not in (ID_COLUMN, LABEL_COLUMN)]
assert len(FEATURES) == 5000, f"expected 5000 features, got {len(FEATURES)}"

dtypes = {c: np.float32 for c in FEATURES}
train_features = pd.read_csv(DATA_DIR / "train_features.csv", dtype=dtypes)
test_features = pd.read_csv(DATA_DIR / "test_features.csv", dtype=dtypes)

X_all = train_features[FEATURES].to_numpy(np.float32)
y_all = train_features[LABEL_COLUMN].to_numpy()
X_test = test_features[FEATURES].to_numpy(np.float32)
test_ids = test_features[ID_COLUMN].copy()

X_train, y_train = X_all[is_train], y_all[is_train]
X_val, y_val = X_all[~is_train], y_all[~is_train]

del train_features
print(X_train.shape, X_val.shape, X_test.shape)
""")

# ---------------------------------------------------------------- Section 1
md("""
---
# Task 1 — Logistic Regression from scratch

No logistic regression package is used here. The five functions the brief asks
for are `sigmoid`, `loss`, `gradients`, `train` and `predict`, all written with
NumPy.

**How it works.** Logistic regression puts a linear score
$z = \\mathbf{w}^\\top \\mathbf{x} + b$ through the sigmoid
$\\sigma(z) = 1/(1+e^{-z})$ to get a probability. Training minimises the log
loss

$$L = -\\frac{1}{m}\\sum_{i=1}^{m}\\Big[y_i\\log\\hat{y}_i + (1-y_i)\\log(1-\\hat{y}_i)\\Big]$$

whose gradients take the neat form
$\\partial L/\\partial \\mathbf{w} = \\frac{1}{m}X^\\top(\\hat{\\mathbf{y}}-\\mathbf{y})$
and $\\partial L/\\partial b = \\frac{1}{m}\\sum_i (\\hat{y}_i - y_i)$.
We follow those gradients downhill in mini-batches.
""")

md("### 1.1 Implementation")
code(PREAMBLE + "\n\n\n" + block("Task 1: Logistic Regression"))

md("""
### 1.2 A first fit

Training with the settings we started from, to confirm the loss actually goes
down and the model learns something better than guessing.
""")

code("""
w, b, losses = train(X_train, y_train.astype(np.float64),
                     bs=64, epochs=300, lr=1.0, shuffle=False, verbose=True)

print("loss:", round(losses[0], 4), "->", round(losses[-1], 4))
print("validation Macro F1:", round(calculate_macro_f1(y_val, predict(X_val, w, b)), 4))
assert losses[-1] < losses[0], "loss should decrease"

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses) + 1), losses, linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("Training log loss")
plt.title("Task 1 — training loss (lr=1.0, bs=64)")
plt.grid(alpha=0.3)
plt.savefig(FIGURES_DIR / "logreg_training_loss.png", dpi=150)
plt.show()
""")

md("""
### 1.3 Tuning

Three things are varied against that starting point:

1. **Shuffling** the rows before each epoch, instead of reusing one fixed
   batch order every time.
2. An **L2 penalty**, since 5000 features on 16000 rows leaves room to overfit.
3. The **decision threshold**. This is the one that matters most: Macro F1 on a
   62.5 / 37.5 split is not maximised at the default 0.5 cut-off, because
   moving the cut-off trades recall on one class for the other and the two
   classes are weighted equally in the metric.

The threshold search is shared with Task 3, so it lives alongside the models:
""")

code(block("Shared helper"))

code("""
results = []


def run_config(label, **kwargs):
    settings = {"bs": 64, "epochs": 300, "lr": 1.0}
    settings.update(kwargs)

    t0 = time.time()
    w, b, hist = train(X_train, y_train.astype(np.float64), **settings)
    probabilities = predict_proba(X_val, w, b)

    default_f1 = calculate_macro_f1(y_val, (probabilities >= 0.5).astype(int))
    threshold, tuned_f1 = best_threshold(y_val, probabilities, calculate_macro_f1)

    results.append({
        "config": label,
        "learning_rate": settings["lr"], "epochs": settings["epochs"],
        "batch_size": settings["bs"], "l2": settings.get("l2", 0.0),
        "class_weight": str(settings.get("class_weight")),
        "shuffle": settings.get("shuffle", True),
        "final_train_loss": hist[-1],
        "val_macro_f1": default_f1,
        "best_threshold": threshold,
        "val_macro_f1_tuned": tuned_f1,
        "runtime_s": round(time.time() - t0, 1),
    })
    print(f"{label:<40} F1={default_f1:.4f}  tuned={tuned_f1:.4f} (t={threshold:.3f})")


run_config("baseline (no shuffle, no L2)", shuffle=False)
run_config("shuffle")
for l2 in [1e-5, 1e-4, 1e-3]:
    run_config(f"shuffle + l2={l2}", l2=l2)
run_config("shuffle + l2=1e-05 + balanced", l2=1e-5, class_weight="balanced")
for lr in [0.5, 2.0]:
    run_config(f"shuffle + l2=1e-05 + lr={lr}", l2=1e-5, lr=lr)
run_config("shuffle + l2=1e-05 + epochs=500", l2=1e-5, epochs=500)

logreg_results = (pd.DataFrame(results)
                  .sort_values("val_macro_f1_tuned", ascending=False)
                  .reset_index(drop=True))
logreg_results.to_csv(RESULTS_DIR / "logreg_tuning_results.csv", index=False)
logreg_results
""")

md("""
**Reading the table.** The gains are real but small. Shuffling on its own does
not reliably help, a light L2 (`1e-5`) helps a little, and the threshold is
worth roughly +0.01. The spread across the top few rows is about 0.005, which
is inside the noise of a 4000-row validation set — so we should not read much
into the exact ordering of the leaders.
""")

code("""
best = logreg_results.iloc[0]
BEST_LR = float(best["learning_rate"])
BEST_EPOCHS = int(best["epochs"])
BEST_BS = int(best["batch_size"])
BEST_L2 = float(best["l2"])
BEST_CLASS_WEIGHT = None if best["class_weight"] == "None" else best["class_weight"]
BEST_THRESHOLD = float(best["best_threshold"])

print(f"selected: lr={BEST_LR}, epochs={BEST_EPOCHS}, bs={BEST_BS}, l2={BEST_L2}, "
      f"class_weight={BEST_CLASS_WEIGHT}, threshold={BEST_THRESHOLD:.4f}")
print("validation Macro F1:", round(float(best["val_macro_f1_tuned"]), 4))
""")

md("""
### 1.4 Refit on all labelled data and predict the test set

The hyperparameters are fixed above using the validation split; the final model
is then refitted on all 20000 labelled rows so it sees as much data as possible.
""")

code("""
final_w, final_b, _ = train(X_all, y_all.astype(np.float64),
                            bs=BEST_BS, epochs=BEST_EPOCHS, lr=BEST_LR,
                            l2=BEST_L2, class_weight=BEST_CLASS_WEIGHT,
                            verbose=True)

logreg_test_preds = predict(X_test, final_w, final_b, threshold=BEST_THRESHOLD)

logreg_submission = create_submission(
    test_ids=test_ids,
    predictions=logreg_test_preds,
    output_path=SUBMISSIONS_DIR / "LogReg_Prediction.csv",
    id_column=ID_COLUMN,
    label_column=LABEL_COLUMN,
)
print("class-1 rate:", round(float(logreg_test_preds.mean()), 4))
logreg_submission.head()
""")

code("""
# Format check: right columns, right length, IDs in their original order
saved = pd.read_csv(SUBMISSIONS_DIR / "LogReg_Prediction.csv", dtype={ID_COLUMN: "string"})
assert saved.columns.tolist() == [ID_COLUMN, LABEL_COLUMN]
assert len(saved) == len(test_ids)
assert saved[ID_COLUMN].tolist() == test_ids.astype("string").tolist()
assert saved[LABEL_COLUMN].isnull().sum() == 0
assert set(saved[LABEL_COLUMN].unique()).issubset({0, 1})
print("LogReg_Prediction.csv verified — ready to upload")
""")

# ---------------------------------------------------------------- Section 2
md("""
---
# Task 2 — PCA and KNN

PCA finds the directions along which the data varies most and re-expresses each
document using only the strongest ones. The brief allows sklearn here, and
fixes `n_neighbors=2`.
""")

code("""
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

N_NEIGHBOURS = 2                       # fixed by the task brief
COMPONENT_SETTINGS = [2000, 1000, 500, 100]


def evaluate_pca(n_components):
    \"\"\"Fit PCA and KNN on the training split and score on the validation split.\"\"\"
    start = time.time()

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
    knn.fit(X_train_pca, y_train)
    predictions = knn.predict(X_val_pca)

    return {
        "components": n_components,
        "val_macro_f1": calculate_macro_f1(y_val, predictions),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pred_class1_rate": float(predictions.mean()),
        "runtime_s": round(time.time() - start, 1),
    }


pca_results = pd.DataFrame([evaluate_pca(n) for n in COMPONENT_SETTINGS])
pca_results.to_csv(RESULTS_DIR / "pca_knn_results.csv", index=False)
pca_results
""")

md("""
### 2.1 The result is backwards — and that is the interesting part

Macro F1 gets **worse** as we keep more components: 100 components score far
better than 2000, even though 2000 components retain 78% of the variance and
100 retain only 16%. Two things cause this.

**Distance concentration.** KNN depends on some neighbours being meaningfully
closer than others. In high dimensions the distances between a point and all of
its neighbours converge, so "nearest" stops carrying much signal. Cutting to
100 components is what makes the neighbourhoods meaningful again.

**An even `k` needs a tie-break.** With `n_neighbors=2` the two neighbours
often disagree, and sklearn settles a tie by taking the lowest class index —
class 0. The cell below counts how often that happens.
""")

code("""
tie_rows = []
for n_components in COMPONENT_SETTINGS:
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS).fit(X_train_pca, y_train)
    _, neighbour_idx = knn.kneighbors(X_val_pca)
    neighbour_labels = y_train[neighbour_idx]

    is_tie = neighbour_labels[:, 0] != neighbour_labels[:, 1]
    tie_rows.append({
        "components": n_components,
        "tie_rate": float(is_tie.mean()),
        "true_class1_rate_among_ties": float(y_val[is_tie].mean()),
        "val_macro_f1_k2": calculate_macro_f1(y_val, knn.predict(X_val_pca)),
        # the nearest neighbour on its own, i.e. the same model without the tie rule
        "val_macro_f1_k1": calculate_macro_f1(y_val, neighbour_labels[:, 0]),
    })

tie_analysis = pd.DataFrame(tie_rows)
tie_analysis.to_csv(RESULTS_DIR / "pca_knn_tie_analysis.csv", index=False)
tie_analysis
""")

md("""
At 2000 components about 23% of validation rows are ties, and roughly 74% of
those tied rows are genuinely class 1 — every one of them is handed to class 0
by the tie-break rule. That is why the model predicts class 1 for only 8% of
rows when the true rate is 63%.

The `k=1` column isolates the effect: dropping the second neighbour, and with
it the tie-break, recovers about 0.09 Macro F1 at 2000 components. We keep
`k=2` because the brief fixes it, but it is the single biggest thing holding
this model back.
""")

code("""
full_pca = PCA(n_components=max(COMPONENT_SETTINGS), random_state=RANDOM_SEED)
full_pca.fit(X_train)
cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

variance_analysis = pd.DataFrame([{
    "components": n,
    "share_of_original_features": n / len(FEATURES),
    "cumulative_explained_variance": float(cumulative_variance[n - 1]),
} for n in sorted(COMPONENT_SETTINGS)])
variance_analysis.to_csv(RESULTS_DIR / "pca_variance_analysis.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, linewidth=2)
for n in COMPONENT_SETTINGS:
    axes[0].axvline(n, color="grey", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Number of components"); axes[0].set_ylabel("Cumulative explained variance")
axes[0].set_title("How much variance the components retain"); axes[0].grid(alpha=0.3)

axes[1].plot(pca_results["components"], pca_results["val_macro_f1"],
             marker="o", linewidth=2, color="darkorange")
axes[1].set_xlabel("Number of PCA components"); axes[1].set_ylabel("Validation Macro F1")
axes[1].set_title(f"Components vs Macro F1 (KNN, k={N_NEIGHBOURS})"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pca_knn_analysis.png", dpi=150)
plt.show()
display(variance_analysis)
""")

md("""
### 2.2 The four Kaggle submissions

One file per component setting, each refitted on all 20000 labelled rows.
""")

code("""
for n_components in COMPONENT_SETTINGS:
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_all_pca = pca.fit_transform(X_all)
    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS).fit(X_all_pca, y_all)
    test_preds = knn.predict(X_test_pca)

    create_submission(
        test_ids=test_ids,
        predictions=test_preds,
        output_path=SUBMISSIONS_DIR / f"PCA{n_components}_KNN_Prediction.csv",
        id_column=ID_COLUMN,
        label_column=LABEL_COLUMN,
    )
    print(f"PCA{n_components}_KNN_Prediction.csv saved; "
          f"class-1 rate={test_preds.mean():.4f}")
""")

# ---------------------------------------------------------------- Section 3
md("""
---
# Task 3 — Other models

Task 3 requires the models to be written from scratch, and only allows existing
libraries for **ensemble** models. So the Naive Bayes, Complement Naive Bayes
and SGD linear classifier below are all written out with NumPy; Extra Trees is
taken from sklearn because it is an ensemble.

Sections 3.1 to 3.5 all use the provided 5000 TF-IDF features, so the
comparison is about the models. Section 3.6 then changes the features instead —
which turns out to matter far more than the choice of model.
""")

md("""
## 3.1 Multinomial Naive Bayes (from scratch)

Naive Bayes applies Bayes' rule and assumes the features are independent given
the class — "naive" because that is plainly false for words, yet it works
anyway. For each class it stores how much total TF-IDF mass each feature
carries, smoothed by `alpha` so an unseen term does not zero out the whole
product, and scores a document by summing log-probabilities.
""")

code(block("Task 3: Multinomial Naive Bayes"))

code("""
nb_grid = []
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
    classes, prior, feature_log_prob = nb_fit(X_train, y_train, alpha=alpha)
    score = calculate_macro_f1(y_val, nb_predict(X_val, classes, prior, feature_log_prob))
    nb_grid.append({"alpha": alpha, "val_macro_f1": score})
    print(f"alpha={alpha:<7} F1={score:.4f}")

nb_grid = pd.DataFrame(nb_grid).sort_values("val_macro_f1", ascending=False)
nb_grid.to_csv(RESULTS_DIR / "task3_nb_alpha.csv", index=False)
nb_grid
""")

md("""
`alpha` matters a lot here, and smaller is better. The provided features are
TF-IDF weights rather than raw counts, so the per-class mass in each feature is
small; a smoothing constant of 1.0 is then large relative to the real signal
and washes it out.
""")

md("""
## 3.2 Complement Naive Bayes (from scratch)

A variant built for uneven class sizes. Rather than asking "how typical is this
document of class *c*", it asks "how *untypical* is it of everything that is
not *c*", and it normalises the weights so long documents cannot dominate.
""")

code(block("Task 3: Complement Naive Bayes"))

code("""
cnb_grid = []
for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    classes_c, weights_c = cnb_fit(X_train, y_train, alpha=alpha)
    score = calculate_macro_f1(y_val, cnb_predict(X_val, classes_c, weights_c))
    cnb_grid.append({"alpha": alpha, "val_macro_f1": score})
    print(f"alpha={alpha:<7} F1={score:.4f}")

cnb_grid = pd.DataFrame(cnb_grid).sort_values("val_macro_f1", ascending=False)
cnb_grid.to_csv(RESULTS_DIR / "task3_cnb_alpha.csv", index=False)
cnb_grid
""")

md("""
## 3.3 Linear classifier trained by SGD (from scratch)

One trainer covers three losses, so comparing them is a one-line change:

* **hinge** — this is a linear SVM: it only pays a penalty for points inside
  the margin, so points already classified confidently stop contributing;
* **modified Huber** — a smoothed hinge that is gentler on outliers;
* **log loss** — logistic regression again, for reference.

`class_weight="balanced"` reweights the two classes to equal total mass, which
is the right thing to do when the metric is Macro F1.
""")

code(block("Task 3: linear classifier trained by mini-batch SGD"))

code("""
sgd_grid = []
for loss_name in ["hinge", "modified_huber", "log_loss"]:
    t0 = time.time()
    w_s, b_s, _, _ = sgd_fit(X_train, y_train, loss=loss_name, penalty="l2",
                             alpha=1e-4, class_weight="balanced", lr=0.5, epochs=100)
    score = calculate_macro_f1(y_val, sgd_predict(X_val, w_s, b_s))
    sgd_grid.append({"setting": f"loss={loss_name}", "val_macro_f1": score,
                     "fit_seconds": round(time.time() - t0, 1)})
    print(f"loss={loss_name:<16} F1={score:.4f}")

BEST_SGD_LOSS = max(sgd_grid, key=lambda r: r["val_macro_f1"])["setting"].split("=")[1]

for alpha in [1e-5, 1e-4, 1e-3]:
    t0 = time.time()
    w_s, b_s, _, _ = sgd_fit(X_train, y_train, loss=BEST_SGD_LOSS, penalty="l2",
                             alpha=alpha, class_weight="balanced", lr=0.5, epochs=100)
    score = calculate_macro_f1(y_val, sgd_predict(X_val, w_s, b_s))
    sgd_grid.append({"setting": f"{BEST_SGD_LOSS}, alpha={alpha}", "val_macro_f1": score,
                     "fit_seconds": round(time.time() - t0, 1)})
    print(f"alpha={alpha:<8} F1={score:.4f}")

sgd_grid = pd.DataFrame(sgd_grid)
sgd_grid.to_csv(RESULTS_DIR / "task3_sgd_grid.csv", index=False)
sgd_grid
""")

md("""
### Convergence check

Training loss against validation Macro F1 per epoch, to confirm the optimiser
is actually converging and to see where it stops helping.
""")

code("""
w_check, b_check, losses_check, val_f1_check = sgd_fit(
    X_train, y_train, loss="hinge", penalty="l2", alpha=1e-4,
    lr=0.5, epochs=100, bs=256, class_weight="balanced",
    X_val=X_val, y_val=y_val, eval_fn=calculate_macro_f1,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, len(losses_check) + 1), losses_check, linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Hinge loss (train)")
axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3)

axes[1].plot(range(1, len(val_f1_check) + 1), val_f1_check, linewidth=2, color="darkorange")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Macro F1")
axes[1].set_title("Validation F1 per epoch"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "sgd_convergence_check.png", dpi=150)
plt.show()

print("val F1 first 3:", [round(f, 4) for f in val_f1_check[:3]])
print("val F1 last 3:", [round(f, 4) for f in val_f1_check[-3:]])
""")

md("""
## 3.4 Extra Trees (library ensemble — permitted)

Extra Trees grows many decision trees, each on a random subset of features and
with randomly chosen split points, then averages their votes. The extra
randomness makes each individual tree weak but decorrelates them, and averaging
decorrelated weak learners is what makes the ensemble strong.

Tuning is one-factor-at-a-time from a baseline, so each setting's effect is
readable on its own. The full sweep lives in
`scripts/task3_extra_trees_tuning.py`; the best configuration is refitted here.
""")

code("""
from sklearn.ensemble import ExtraTreesClassifier   # ensemble model, permitted

t0 = time.time()
extra_trees = ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                   random_state=RANDOM_SEED, n_jobs=-1)
extra_trees.fit(X_train, y_train)
et_proba = extra_trees.predict_proba(X_val)[:, 1]

et_default = calculate_macro_f1(y_val, extra_trees.predict(X_val))
et_threshold, et_tuned = best_threshold(y_val, et_proba, calculate_macro_f1)

print(f"train Macro F1     : {calculate_macro_f1(y_train, extra_trees.predict(X_train)):.4f}")
print(f"validation Macro F1: {et_default:.4f}")
print(f"with tuned cut-off : {et_tuned:.4f} (threshold={et_threshold:.3f})  [{time.time()-t0:.0f}s]")
""")

md("""
The train score is 1.0 against a validation score near 0.72 — the trees memorise
the training set completely. Limiting `max_depth` was tried and made things
worse, so we left the trees unrestricted and accepted the gap: the ensemble
average, not each tree, is what generalises.

Note also how much the threshold is worth here compared to the linear models.
Extra Trees produces badly calibrated probabilities, so the default 0.5 cut-off
is a long way from the best operating point.
""")

md("""
## 3.5 Soft-vote ensemble

The three models disagree in different ways, so averaging them should be more
stable than any one alone. Their scores live on completely different scales
(a probability, a signed margin, a log-probability), so each is converted to a
rank in [0, 1] before averaging — that makes the vote scale-free.
""")

code("""
def to_rank(scores):
    \"\"\"Map scores onto [0, 1] by rank, so different scales can be averaged.\"\"\"
    return np.argsort(np.argsort(scores)) / (len(scores) - 1)


classes, prior, feature_log_prob = nb_fit(X_train, y_train,
                                          alpha=float(nb_grid.iloc[0]["alpha"]))
nb_proba = nb_predict_proba(X_val, prior, feature_log_prob)[:, 1]

w_sgd, b_sgd, _, _ = sgd_fit(X_train, y_train, loss=BEST_SGD_LOSS, penalty="l2",
                             alpha=1e-4, class_weight="balanced", lr=0.5, epochs=100)
sgd_scores = sgd_decision_function(X_val, w_sgd, b_sgd)

vote = np.mean([to_rank(sgd_scores), to_rank(et_proba), to_rank(nb_proba)], axis=0)
vote_threshold, vote_tuned = best_threshold(y_val, vote, calculate_macro_f1)

print("soft vote, default cut-off:", round(calculate_macro_f1(y_val, (vote >= 0.5).astype(int)), 4))
print("soft vote, tuned cut-off  :", round(vote_tuned, 4), f"(threshold={vote_threshold:.3f})")
""")

md("""
The vote lands between its members rather than above the best of them. Averaging
helps when the members are strong *and* make different mistakes; here Naive
Bayes is much weaker than the other two, so it pulls the average down more than
its diversity adds back.
""")

md("""
## 3.6 Our own features: hybrid TF-IDF + stylometry

Everything so far reads the same 5000 provided features, and everything so far
lands between 0.67 and 0.76. The brief allows Task 3 to use our own features as
long as we describe them, so this is where we changed the input instead of the
model — and it is where the real gain came from.

Three changes:

**1. Word *and* character TF-IDF.** Word 1–2 grams capture vocabulary and short
phrases. Character 3–5 grams inside word boundaries capture spelling, morphology
and punctuation habits — the texture of the writing rather than its content.
The two blocks are normalised separately and concatenated.

**2. Stylometry.** 78 hand-built features measuring *how* a text is written
rather than what it says: sentence-length variability (human writing is
burstier), vocabulary richness, function-word rates, repetition of sentence
openings, punctuation and formatting rates, and readability scores. These were
motivated by Section 0.3, where the two classes already differed in length and
variability.

**3. The same from-scratch linear SVM**, trained by averaged SGD.

None of this uses an sklearn estimator — the TF-IDF weighting, the scaler and
the optimiser are all written out below.
""")

md("""
#### The TF-IDF vectorizer, written from scratch

`ScratchTfidfVectorizer` builds the vocabulary, computes
$\\text{idf}(t) = \\log\\frac{1+N}{1+\\text{df}(t)} + 1$, applies sublinear term
frequency $1 + \\log(\\text{tf})$ and L2-normalises each row.
`ScratchHybridTfidf` runs one word-level and one character-level vectorizer and
concatenates them.
""")

code("from collections import Counter\nimport math\nimport re\n\nfrom scipy import sparse\n\n\n"
     + text_feature_block("class ScratchStandardScaler:", "class ScratchAveragedHingeSGD:"))

md("""
#### The 78 stylometry features

Measures of *how* a text is written: burstiness of sentence length, vocabulary
richness (hapax rate, entropy), repeated n-grams and sentence openings,
function-word rates by group, punctuation and formatting rates, and readability
scores.
""")

code(text_feature_block("FUNCTION_WORD_GROUPS = {", "class ScratchStandardScaler:")
     + "\n\n\n" + text_feature_block("def safe_stats(values):", "def combine(tfidf, styles, weight):")
     + "\n\n\n" + text_feature_block("def combine(tfidf, styles, weight):"))

md("""
The classifier is the **same `sgd_fit` from Section 3.3** — nothing about the
model changes here, only its input. That is deliberate: it isolates the effect
of the features.
""")

code("""
train_text = train_raw.loc[is_train, "text"]
val_text = train_raw.loc[~is_train, "text"]
y_train_text = train_raw.loc[is_train, LABEL_COLUMN]
y_val_text = train_raw.loc[~is_train, LABEL_COLUMN].to_numpy()

t0 = time.time()
vectorizer = ScratchHybridTfidf()
train_tfidf = vectorizer.fit_transform(train_text)
val_tfidf = vectorizer.transform(val_text)

scaler = ScratchStandardScaler()
train_styles = scaler.fit_transform(style_matrix(train_text))
val_styles = scaler.transform(style_matrix(val_text))

print(f"TF-IDF block: {train_tfidf.shape}, stylometry block: {train_styles.shape}"
      f"  [{time.time()-t0:.0f}s]")
""")

md("""
### Which loss? The question we nearly forgot to re-ask

Section 3.3 found modified Huber clearly best on the provided features. When we
first built this pipeline we used hinge loss and never re-tested that choice on
the new features — an easy thing to miss, and it turned out to cost about 0.02
Macro F1. The lesson: a hyperparameter chosen on one representation does not
automatically carry to another.
""")

code("""
x_train_hybrid = combine(train_tfidf, train_styles, 0.10)
x_val_hybrid = combine(val_tfidf, val_styles, 0.10)

loss_rows = []
for loss_name in ["hinge", "modified_huber", "log_loss"]:
    t0 = time.time()
    w_h, b_h, _, _ = sgd_fit(x_train_hybrid, y_train_text, loss=loss_name,
                             penalty="l2", alpha=1e-5, class_weight="balanced",
                             lr=0.5, epochs=60, bs=256, random_state=RANDOM_SEED)
    score = calculate_macro_f1(y_val_text, sgd_predict(x_val_hybrid, w_h, b_h))
    loss_rows.append({"loss": loss_name, "val_macro_f1": score,
                      "fit_seconds": round(time.time() - t0, 1)})
    print(f"loss={loss_name:<16} F1={score:.4f}")

pd.DataFrame(loss_rows).to_csv(RESULTS_DIR / "task3_loss_on_hybrid.csv", index=False)
pd.DataFrame(loss_rows)
""")

md("""
### Tuning the stylometry weight and the training length

`style_weight` scales the 78 stylometry columns against the ~200,000 TF-IDF
columns — without it they are simply drowned out.
""")

code("""
style_rows = []
for weight in [0.05, 0.10]:
    x_tr = combine(train_tfidf, train_styles, weight)
    x_va = combine(val_tfidf, val_styles, weight)

    for epochs in [30, 60, 100]:
        t0 = time.time()
        w_s, b_s, _, _ = sgd_fit(x_tr, y_train_text, loss="modified_huber",
                                 penalty="l2", alpha=1e-5, class_weight="balanced",
                                 lr=0.5, epochs=epochs, bs=256, random_state=RANDOM_SEED)
        scores = sgd_decision_function(x_va, w_s, b_s)

        default_f1 = calculate_macro_f1(y_val_text, (scores >= 0).astype(int))
        threshold, tuned_f1 = best_threshold(y_val_text, scores, calculate_macro_f1)
        style_rows.append({
            "style_weight": weight, "epochs": epochs,
            "val_macro_f1": default_f1, "best_threshold": threshold,
            "val_macro_f1_tuned": tuned_f1, "fit_seconds": round(time.time() - t0, 1),
        })
        print(f"style_weight={weight:<5} epochs={epochs:<4} "
              f"F1={default_f1:.4f} tuned={tuned_f1:.4f}")

style_results = (pd.DataFrame(style_rows)
                 .sort_values("val_macro_f1", ascending=False)
                 .reset_index(drop=True))
style_results.to_csv(RESULTS_DIR / "task3_final_model_results.csv", index=False)
style_results
""")

md("""
This is the jump: from **0.7446** on the provided features to **0.8491** on our
own, with the same from-scratch trainer doing the learning. The model was never
the bottleneck — the features were.

Two smaller observations. Training past 60 epochs stops helping (we checked out
to 200 epochs: the score sits flat within 0.002), so 60 is the plateau rather
than the edge of our grid. And the tuned threshold is worth only +0.0007 here,
unlike Extra Trees where it was worth +0.047 — the margin-based losses put the
natural cut-off at 0 already.
""")

md("""
### Final model: refit on all labelled data and predict the test set
""")

code("""
BEST_STYLE_WEIGHT = float(style_results.iloc[0]["style_weight"])
BEST_EPOCHS = int(style_results.iloc[0]["epochs"])

# A threshold picked on 4000 validation rows can easily be noise, so we only
# adopt one when it earns more than that noise; otherwise we keep the plain 0.
THRESHOLD_MARGIN = 0.005
threshold_gain = (float(style_results.iloc[0]["val_macro_f1_tuned"])
                  - float(style_results.iloc[0]["val_macro_f1"]))
BEST_STYLE_THRESHOLD = (float(style_results.iloc[0]["best_threshold"])
                        if threshold_gain > THRESHOLD_MARGIN else 0.0)

print(f"selected style_weight={BEST_STYLE_WEIGHT}, epochs={BEST_EPOCHS}, "
      f"threshold={BEST_STYLE_THRESHOLD:.4f} (tuning would gain {threshold_gain:+.4f})")

t0 = time.time()
final_vectorizer = ScratchHybridTfidf()
all_tfidf = final_vectorizer.fit_transform(train_raw["text"])
test_tfidf = final_vectorizer.transform(test_raw["text"])

final_scaler = ScratchStandardScaler()
all_styles = final_scaler.fit_transform(style_matrix(train_raw["text"]))
test_styles = final_scaler.transform(style_matrix(test_raw["text"]))

x_all = combine(all_tfidf, all_styles, BEST_STYLE_WEIGHT)
x_test = combine(test_tfidf, test_styles, BEST_STYLE_WEIGHT)

final_w, final_b, _, _ = sgd_fit(x_all, train_raw[LABEL_COLUMN].to_numpy(),
                                 loss="modified_huber", penalty="l2", alpha=1e-5,
                                 class_weight="balanced", lr=0.5, epochs=BEST_EPOCHS,
                                 bs=256, random_state=RANDOM_SEED)
final_preds = sgd_predict(x_test, final_w, final_b, threshold=BEST_STYLE_THRESHOLD)

create_submission(
    test_ids=test_raw[ID_COLUMN],
    predictions=final_preds,
    output_path=SUBMISSIONS_DIR / "Final_Prediction.csv",
    id_column=ID_COLUMN,
    label_column=LABEL_COLUMN,
)
print(f"Final_Prediction.csv saved; class-1 rate={final_preds.mean():.4f}"
      f"  [{time.time()-t0:.0f}s]")
""")

code("""
# Format check on the final submission
saved = pd.read_csv(SUBMISSIONS_DIR / "Final_Prediction.csv", dtype={ID_COLUMN: "string"})
assert saved.columns.tolist() == [ID_COLUMN, LABEL_COLUMN]
assert len(saved) == len(test_raw)
assert saved[ID_COLUMN].tolist() == test_raw[ID_COLUMN].astype("string").tolist()
assert saved[LABEL_COLUMN].isnull().sum() == 0
assert set(saved[LABEL_COLUMN].unique()).issubset({0, 1})
print("Final_Prediction.csv verified — ready to upload")
""")

# ---------------------------------------------------------------- Summary
md("""
---
# Summary of every model we explored

All scores are Macro F1 on the same shared validation split. "Tuned" means the
decision threshold was chosen on validation as well.
""")

code("""
summary = pd.DataFrame([
    {"task": 1, "model": "Logistic Regression", "features": "provided 5000 TF-IDF",
     "implementation": "from scratch",
     "val_macro_f1": float(logreg_results.iloc[0]["val_macro_f1"]),
     "val_macro_f1_tuned": float(logreg_results.iloc[0]["val_macro_f1_tuned"])},
    {"task": 2, "model": "PCA(100) + KNN k=2", "features": "provided 5000 TF-IDF",
     "implementation": "sklearn (permitted)",
     "val_macro_f1": float(pca_results.set_index("components").loc[100, "val_macro_f1"]),
     "val_macro_f1_tuned": np.nan},
    {"task": 3, "model": "Multinomial Naive Bayes", "features": "provided 5000 TF-IDF",
     "implementation": "from scratch",
     "val_macro_f1": float(nb_grid.iloc[0]["val_macro_f1"]), "val_macro_f1_tuned": np.nan},
    {"task": 3, "model": "Complement Naive Bayes", "features": "provided 5000 TF-IDF",
     "implementation": "from scratch",
     "val_macro_f1": float(cnb_grid.iloc[0]["val_macro_f1"]), "val_macro_f1_tuned": np.nan},
    {"task": 3, "model": "Linear classifier by SGD", "features": "provided 5000 TF-IDF",
     "implementation": "from scratch",
     "val_macro_f1": float(sgd_grid["val_macro_f1"].max()), "val_macro_f1_tuned": np.nan},
    {"task": 3, "model": "Extra Trees", "features": "provided 5000 TF-IDF",
     "implementation": "sklearn ensemble (permitted)",
     "val_macro_f1": et_default, "val_macro_f1_tuned": et_tuned},
    {"task": 3, "model": "Soft vote (linear + Extra Trees + NB)", "features": "provided 5000 TF-IDF",
     "implementation": "ensemble",
     "val_macro_f1": calculate_macro_f1(y_val, (vote >= 0.5).astype(int)),
     "val_macro_f1_tuned": vote_tuned},
    {"task": 3, "model": "Hybrid TF-IDF + stylometry + SGD (modified Huber)",
     "features": "our own", "implementation": "from scratch",
     "val_macro_f1": float(style_results.iloc[0]["val_macro_f1"]),
     "val_macro_f1_tuned": float(style_results.iloc[0]["val_macro_f1_tuned"])},
])

summary = summary.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)
summary.round(4)
""")

md("""
### What we take from this

1. **Features beat models.** Every model on the provided 5000 features lands
   between 0.67 and 0.76 — a spread of under 0.09 across families as different
   as Naive Bayes and random forests. Swapping in our own word + character
   TF-IDF and stylometry lifted the *same* trainer from 0.7446 to 0.8491. The
   biggest single win came from changing the input, not the algorithm.
2. **Re-ask settled questions when the inputs change.** We picked modified Huber
   on the provided features in Section 3.3, then built the hybrid pipeline with
   hinge loss and never re-tested. Re-testing was worth about +0.02 — more than
   every other tuning decision in Task 3 combined.
3. **The decision threshold is not a detail — but it is not always worth
   taking.** It is worth +0.047 for Extra Trees, whose votes are poorly
   calibrated, and only +0.0007 for the final model. We adopt it only when the
   gain clears the noise of a 4000-row split.
4. **An even `k` hurts KNN.** `n_neighbors=2` forces a tie-break that
   systematically favours class 0, costing about 0.09 Macro F1 at 2000
   components — a larger effect than the component count itself.
5. **We expect the test score to trail validation.** The brief warns this sample
   is under 5% of the original dataset. We stopped tuning rather than chase
   validation decimals that will not transfer, and we removed an earlier
   submission that forced the test-set class balance to a fixed quantile.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"wrote {OUT} with {len(cells)} cells")
