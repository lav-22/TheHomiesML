"""Task 3: does the loss function that won on the provided features also win
on our own features?

On the provided 5000 TF-IDF features, modified Huber beat plain hinge by about
0.045 Macro F1. The hybrid TF-IDF + stylometry pipeline only ever used hinge,
so this checks whether that gap carries over to the larger feature space.

Both trainers are from scratch: `ScratchAveragedHingeSGD` from
src/text_features.py, and the multi-loss `sgd_fit` from src/scratch_models.py.

Run from the repository root:

    python3 scripts/task3_loss_on_hybrid.py
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
from src.scratch_models import best_threshold, sgd_decision_function, sgd_fit
from src.text_features import (
    ScratchAveragedHingeSGD,
    ScratchHybridTfidf,
    ScratchStandardScaler,
    combine,
    style_matrix,
)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42
STYLE_WEIGHT = 0.10          # selected in task3_final_model.py

train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", "text", "label"])
split = pd.read_csv(DATA_DIR / "splits" / "shared_validation_split.csv")
train_mask = split["split"].eq("train").to_numpy()

train_text = train.loc[train_mask, "text"]
val_text = train.loc[~train_mask, "text"]
y_train = train.loc[train_mask, "label"].to_numpy()
y_val = train.loc[~train_mask, "label"].to_numpy()

print("Building the hybrid features once...")
t0 = time.time()
vectorizer = ScratchHybridTfidf()
train_tfidf = vectorizer.fit_transform(train_text)
val_tfidf = vectorizer.transform(val_text)

scaler = ScratchStandardScaler()
train_styles = scaler.fit_transform(style_matrix(train_text))
val_styles = scaler.transform(style_matrix(val_text))

x_train = combine(train_tfidf, train_styles, STYLE_WEIGHT)
x_val = combine(val_tfidf, val_styles, STYLE_WEIGHT)
print(f"  {x_train.shape}  [{time.time()-t0:.0f}s]")

rows = []


def record(name, scores, seconds):
    default_f1 = calculate_macro_f1(y_val, (scores >= 0).astype(int))
    threshold, tuned_f1 = best_threshold(y_val, scores, calculate_macro_f1)
    rows.append({
        "model": name,
        "val_macro_f1": default_f1,
        "best_threshold": threshold,
        "val_macro_f1_tuned": tuned_f1,
        "fit_seconds": round(seconds, 1),
    })
    print(f"{name:<44} F1={default_f1:.4f}  tuned={tuned_f1:.4f}  [{seconds:.0f}s]")


# The current final model: averaged hinge SGD
t0 = time.time()
model = ScratchAveragedHingeSGD(alpha=1e-4, random_state=RANDOM_SEED)
model.fit(x_train, y_train)
record("averaged hinge SGD (current final model)",
       model.decision_function(x_val), time.time() - t0)

# The multi-loss trainer, so hinge and modified Huber are compared like for like
for loss_name in ["hinge", "modified_huber", "log_loss"]:
    for lr in [0.5, 5.0]:
        t0 = time.time()
        w, b, _, _ = sgd_fit(x_train, y_train, loss=loss_name, penalty="l2",
                             alpha=1e-4, class_weight="balanced",
                             lr=lr, epochs=30, bs=256, random_state=RANDOM_SEED)
        record(f"sgd_fit loss={loss_name}, lr={lr}",
               sgd_decision_function(x_val, w, b), time.time() - t0)

results = (pd.DataFrame(rows)
           .sort_values("val_macro_f1_tuned", ascending=False)
           .reset_index(drop=True))
results.to_csv(RESULTS_DIR / "task3_loss_on_hybrid.csv", index=False)

print("\n=== Loss comparison on the hybrid features ===")
print(results.to_string(index=False))


# ---------------------------------------------------------------------------
# Refinement: modified Huber won the first pass by a clear margin, so check the
# win holds across a small grid rather than resting on one lucky setting.
# ---------------------------------------------------------------------------
print("\n=== Refining modified Huber ===")
refine_rows = []
for lr in [0.3, 0.5, 1.0]:
    for epochs in [30, 60]:
        for alpha in [1e-5, 1e-4]:
            t0 = time.time()
            w, b, _, _ = sgd_fit(x_train, y_train, loss="modified_huber",
                                 penalty="l2", alpha=alpha, class_weight="balanced",
                                 lr=lr, epochs=epochs, bs=256,
                                 random_state=RANDOM_SEED)
            scores = sgd_decision_function(x_val, w, b)
            default_f1 = calculate_macro_f1(y_val, (scores >= 0).astype(int))
            threshold, tuned_f1 = best_threshold(y_val, scores, calculate_macro_f1)
            refine_rows.append({
                "lr": lr, "epochs": epochs, "alpha": alpha,
                "val_macro_f1": default_f1, "best_threshold": threshold,
                "val_macro_f1_tuned": tuned_f1,
                "fit_seconds": round(time.time() - t0, 1),
            })
            print(f"lr={lr:<5} epochs={epochs:<4} alpha={alpha:<7} "
                  f"F1={default_f1:.4f} tuned={tuned_f1:.4f}")

refined = (pd.DataFrame(refine_rows)
           .sort_values("val_macro_f1_tuned", ascending=False)
           .reset_index(drop=True))
refined.to_csv(RESULTS_DIR / "task3_modified_huber_refine.csv", index=False)
print("\n", refined.to_string(index=False), sep="")
