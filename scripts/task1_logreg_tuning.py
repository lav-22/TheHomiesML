"""Task 1: tuning experiments for the from-scratch logistic regression.

No logistic regression package is used anywhere in this file. Only NumPy is
used for the maths, and sklearn is used for the Macro F1 metric alone.

The three changes tested against the original implementation are:

1. shuffling the rows before each epoch instead of reusing one fixed order,
2. an optional L2 penalty on the weights,
3. an optional class weight, plus a decision threshold tuned on validation.

The last one matters because the metric is Macro F1 and the training set is
62.5% class 1, so the default 0.5 cut-off is not the best operating point.

Run from the repository root:

    python3 scripts/task1_logreg_tuning.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import calculate_macro_f1
from src.scratch_models import best_threshold, predict_proba, train

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ID_COLUMN = "id"
LABEL_COLUMN = "label"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
FEATURES = [c for c in pd.read_csv(DATA_DIR / "train_features.csv", nrows=0).columns
            if c not in (ID_COLUMN, LABEL_COLUMN)]
assert len(FEATURES) == 5000, f"Expected 5000 features, got {len(FEATURES)} — wrong file?"

dtypes = {c: np.float32 for c in FEATURES}
train_df = pd.read_csv(DATA_DIR / "train_features.csv", dtype=dtypes)

X_all = train_df[FEATURES].to_numpy(np.float32)
y_all = train_df[LABEL_COLUMN].to_numpy(np.float64)

split = pd.read_csv(DATA_DIR / "splits" / "shared_validation_split.csv")
assert len(split) == len(X_all)
is_train = (split["split"] == "train").to_numpy()

X_train, y_train = X_all[is_train], y_all[is_train]
X_val, y_val = X_all[~is_train], y_all[~is_train]
print(X_train.shape, X_val.shape, round(y_train.mean(), 4), round(y_val.mean(), 4))

results = []


def run(label, **kwargs):
    """Train one configuration and record both default and tuned-threshold F1."""
    t0 = time.time()
    settings = {"bs": 64, "epochs": 300, "lr": 1.0}
    settings.update(kwargs)
    w, b, hist = train(X_train, y_train, **settings)

    probabilities = predict_proba(X_val, w, b)
    default_f1 = calculate_macro_f1(y_val, (probabilities >= 0.5).astype(int))
    tuned_t, tuned_f1 = best_threshold(y_val, probabilities, calculate_macro_f1)

    results.append({
        "config": label,
        "learning_rate": settings["lr"],
        "epochs": settings["epochs"],
        "batch_size": settings["bs"],
        "l2": settings.get("l2", 0.0),
        "class_weight": str(settings.get("class_weight")),
        "shuffle": settings.get("shuffle", True),
        "final_train_loss": hist[-1],
        "val_macro_f1": default_f1,
        "best_threshold": tuned_t,
        "val_macro_f1_tuned": tuned_f1,
        "runtime_s": round(time.time() - t0, 1),
    })
    print(f"{label:<46} F1={default_f1:.4f}  tuned={tuned_f1:.4f} (t={tuned_t:.3f})"
          f"  [{time.time()-t0:.0f}s]")


# Original configuration from the first pass, kept as the reference point
run("baseline (no shuffle, no L2)", shuffle=False)

# 1. does shuffling each epoch help?
run("shuffle")

# 2. how much L2 does it want?
for l2 in [1e-5, 1e-4, 1e-3, 1e-2]:
    run(f"shuffle + l2={l2}", l2=l2)

# 3. longer training with the best regularisation setting so far
best_l2 = max(
    (r for r in results if r["shuffle"]),
    key=lambda r: r["val_macro_f1_tuned"],
)["l2"]
print(f"\nbest L2 so far: {best_l2}\n")

for epochs in [500, 800]:
    run(f"shuffle + l2={best_l2} + epochs={epochs}", l2=best_l2, epochs=epochs)

# 4. class weighting, which targets the same imbalance the threshold does
run(f"shuffle + l2={best_l2} + balanced", l2=best_l2, class_weight="balanced")

# 5. a couple of learning rates around the previous best
for lr in [0.5, 2.0]:
    run(f"shuffle + l2={best_l2} + lr={lr}", l2=best_l2, lr=lr)

results_df = (pd.DataFrame(results)
              .sort_values("val_macro_f1_tuned", ascending=False)
              .reset_index(drop=True))
results_df.to_csv(RESULTS_DIR / "logreg_improved_results.csv", index=False)

print("\nTop configurations by tuned validation Macro F1:")
print(results_df.head(10).to_string(index=False))
