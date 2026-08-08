"""Task 3: hyperparameter tuning for Extra Trees.

Extra Trees is a library model, which the task brief allows because it is an
ensemble. Tuning is one-factor-at-a-time from a baseline: each block below
changes a single setting and leaves the rest alone, which keeps the effect of
each setting readable.

Run from the repository root:

    python3 scripts/task3_extra_trees_tuning.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import calculate_macro_f1
from src.scratch_models import best_threshold

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "extra" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
ID_COLUMN, LABEL_COLUMN = "id", "label"

train_df = pd.read_csv(DATA_DIR / "train_features.csv")
FEATURES = [c for c in train_df.columns if c not in (ID_COLUMN, LABEL_COLUMN)]
X_all = train_df[FEATURES].to_numpy(np.float32)
y_all = train_df[LABEL_COLUMN].to_numpy()

split = pd.read_csv(DATA_DIR / "splits" / "shared_validation_split.csv")
is_train = (split["split"] == "train").to_numpy()
X_train, y_train = X_all[is_train], y_all[is_train]
X_val, y_val = X_all[~is_train], y_all[~is_train]

BASELINE = {"n_estimators": 200, "max_depth": None, "max_features": "sqrt",
            "class_weight": None}

rows = []


def run(stage, **overrides):
    settings = dict(BASELINE)
    settings.update(overrides)

    t0 = time.time()
    model = ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=-1, **settings)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    seconds = time.time() - t0

    default_f1 = calculate_macro_f1(y_val, model.predict(X_val))
    threshold, tuned_f1 = best_threshold(y_val, proba, calculate_macro_f1)

    rows.append({
        "stage": stage,
        **settings,
        "train_macro_f1": calculate_macro_f1(y_train, model.predict(X_train)),
        "val_macro_f1": default_f1,
        "best_threshold": threshold,
        "val_macro_f1_tuned": tuned_f1,
        "fit_seconds": round(seconds, 1),
    })
    print(f"{stage:<14} {settings}  F1={default_f1:.4f} tuned={tuned_f1:.4f} [{seconds:.0f}s]")


run("baseline")
for n_estimators in [500, 800]:
    run("n_estimators", n_estimators=n_estimators)
for max_depth in [20, 40]:
    run("max_depth", max_depth=max_depth)
for max_features in ["log2", 100]:
    run("max_features", max_features=max_features)
run("class_weight", class_weight="balanced")

results = pd.DataFrame(rows)
results.to_csv(RESULTS_DIR / "task3_extra_trees_tuning.csv", index=False)

print("\n=== Extra Trees tuning ===")
print(results.sort_values("val_macro_f1_tuned", ascending=False).to_string(index=False))
