"""Task 3: compare every model we explored on one common footing.

All models are scored on the same shared validation split and, apart from the
Extra Trees, all of them use the pre-processed 5000 TF-IDF features. That keeps
the comparison about the model rather than about the features.

Task 3 requires the models to be written from scratch and only allows existing
libraries for ensemble models. So:

  * Naive Bayes, Complement Naive Bayes, the SGD linear classifier and the
    logistic regression all come from src/scratch_models.py,
  * Extra Trees is a library model, which the brief permits because it is an
    ensemble,
  * the soft-vote at the end combines the models we already fitted.

Run from the repository root:

    python3 scripts/task3_model_comparison.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier   # ensemble model, permitted

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import calculate_macro_f1
from src.scratch_models import (
    best_threshold,
    cnb_fit,
    cnb_predict,
    cnb_predict_scores,
    nb_fit,
    nb_predict,
    nb_predict_proba,
    predict_proba,
    sgd_decision_function,
    sgd_fit,
    train,
)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
ID_COLUMN, LABEL_COLUMN = "id", "label"

# ----------------------------------------------------------------------------
# Data: the provided 5000 TF-IDF features and the shared split
# ----------------------------------------------------------------------------
train_df = pd.read_csv(DATA_DIR / "train_features.csv")
FEATURES = [c for c in train_df.columns if c not in (ID_COLUMN, LABEL_COLUMN)]
assert len(FEATURES) == 5000, f"expected 5000 features, got {len(FEATURES)}"

X_all = train_df[FEATURES].to_numpy(np.float32)
y_all = train_df[LABEL_COLUMN].to_numpy()

split = pd.read_csv(DATA_DIR / "splits" / "shared_validation_split.csv")
is_train = (split["split"] == "train").to_numpy()
X_train, y_train = X_all[is_train], y_all[is_train]
X_val, y_val = X_all[~is_train], y_all[~is_train]
print(X_train.shape, X_val.shape, round(y_train.mean(), 4), round(y_val.mean(), 4))

rows = []
val_scores = {}          # model name -> continuous score on validation, for the vote


def record(name, family, settings, y_pred, seconds, scores=None):
    f1 = calculate_macro_f1(y_val, y_pred)
    entry = {
        "model": name,
        "family": family,
        "settings": settings,
        "val_macro_f1": f1,
        "pred_class1_rate": float(np.mean(y_pred)),
        "fit_seconds": round(seconds, 1),
    }
    if scores is not None:
        threshold, tuned = best_threshold(y_val, scores, calculate_macro_f1)
        entry["best_threshold"] = threshold
        entry["val_macro_f1_tuned"] = tuned
        val_scores[name] = scores
    rows.append(entry)
    print(f"{name:<34} F1={f1:.4f}"
          + (f"  tuned={entry['val_macro_f1_tuned']:.4f}" if scores is not None else "")
          + f"  [{seconds:.0f}s]")


# ----------------------------------------------------------------------------
# 1. Multinomial Naive Bayes (from scratch)
# ----------------------------------------------------------------------------
print("\n--- Multinomial Naive Bayes ---")
nb_grid = []
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
    classes, prior, feature_log_prob = nb_fit(X_train, y_train, alpha=alpha)
    score = calculate_macro_f1(y_val, nb_predict(X_val, classes, prior, feature_log_prob))
    nb_grid.append({"alpha": alpha, "val_macro_f1": score})
    print(f"  alpha={alpha:<7} F1={score:.4f}")
pd.DataFrame(nb_grid).to_csv(RESULTS_DIR / "task3_nb_alpha.csv", index=False)

best_nb_alpha = max(nb_grid, key=lambda r: r["val_macro_f1"])["alpha"]
t0 = time.time()
classes, prior, feature_log_prob = nb_fit(X_train, y_train, alpha=best_nb_alpha)
nb_proba = nb_predict_proba(X_val, prior, feature_log_prob)[:, 1]
record("Multinomial Naive Bayes", "from scratch", f"alpha={best_nb_alpha}",
       nb_predict(X_val, classes, prior, feature_log_prob), time.time() - t0, nb_proba)

# ----------------------------------------------------------------------------
# 2. Complement Naive Bayes (from scratch)
# ----------------------------------------------------------------------------
print("\n--- Complement Naive Bayes ---")
cnb_grid = []
for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    classes_c, weights_c = cnb_fit(X_train, y_train, alpha=alpha)
    score = calculate_macro_f1(y_val, cnb_predict(X_val, classes_c, weights_c))
    cnb_grid.append({"alpha": alpha, "val_macro_f1": score})
    print(f"  alpha={alpha:<7} F1={score:.4f}")
pd.DataFrame(cnb_grid).to_csv(RESULTS_DIR / "task3_cnb_alpha.csv", index=False)

best_cnb_alpha = max(cnb_grid, key=lambda r: r["val_macro_f1"])["alpha"]
t0 = time.time()
classes_c, weights_c = cnb_fit(X_train, y_train, alpha=best_cnb_alpha)
# CNB picks the class with the *lowest* complement score, so the margin
# between the two columns is the score that rises with class 1
cnb_raw_scores = cnb_predict_scores(X_val, weights_c)
cnb_raw = cnb_raw_scores[:, 0] - cnb_raw_scores[:, 1]
record("Complement Naive Bayes", "from scratch", f"alpha={best_cnb_alpha}",
       cnb_predict(X_val, classes_c, weights_c), time.time() - t0, cnb_raw)

# ----------------------------------------------------------------------------
# 3. Linear classifier trained by SGD (from scratch) — hinge loss is a linear SVM
# ----------------------------------------------------------------------------
print("\n--- SGD linear classifier ---")
sgd_grid = []
for loss_name in ["hinge", "modified_huber", "log_loss"]:
    t0 = time.time()
    w, b, _, _ = sgd_fit(X_train, y_train, loss=loss_name, penalty="l2",
                         alpha=1e-4, class_weight="balanced", lr=0.5, epochs=100)
    score = calculate_macro_f1(y_val, (sgd_decision_function(X_val, w, b) >= 0).astype(int))
    sgd_grid.append({"loss": loss_name, "val_macro_f1": score,
                     "fit_seconds": round(time.time() - t0, 1)})
    print(f"  loss={loss_name:<16} F1={score:.4f}")

best_sgd_loss = max(sgd_grid, key=lambda r: r["val_macro_f1"])["loss"]
for alpha in [1e-5, 1e-4, 1e-3]:
    t0 = time.time()
    w, b, _, _ = sgd_fit(X_train, y_train, loss=best_sgd_loss, penalty="l2",
                         alpha=alpha, class_weight="balanced", lr=0.5, epochs=100)
    score = calculate_macro_f1(y_val, (sgd_decision_function(X_val, w, b) >= 0).astype(int))
    sgd_grid.append({"loss": f"{best_sgd_loss} alpha={alpha}", "val_macro_f1": score,
                     "fit_seconds": round(time.time() - t0, 1)})
    print(f"  alpha={alpha:<8} F1={score:.4f}")
pd.DataFrame(sgd_grid).to_csv(RESULTS_DIR / "task3_sgd_grid.csv", index=False)

best_sgd_alpha = 1e-4
t0 = time.time()
w_sgd, b_sgd, _, _ = sgd_fit(X_train, y_train, loss=best_sgd_loss, penalty="l2",
                             alpha=best_sgd_alpha, class_weight="balanced",
                             lr=0.5, epochs=100)
sgd_scores = sgd_decision_function(X_val, w_sgd, b_sgd)
record("Linear SVM by SGD", "from scratch",
       f"loss={best_sgd_loss}, alpha={best_sgd_alpha}, balanced",
       (sgd_scores >= 0).astype(int), time.time() - t0, sgd_scores)

# ----------------------------------------------------------------------------
# 4. Logistic Regression (from scratch) — Task 1 model, reported for reference.
#    The brief says it does not count towards the Task 3 model count.
# ----------------------------------------------------------------------------
print("\n--- Logistic Regression (Task 1 model, reference only) ---")
t0 = time.time()
w_lr, b_lr, _ = train(X_train, y_train.astype(np.float64), bs=64, epochs=300,
                      lr=0.5, l2=1e-5, shuffle=True, seed=RANDOM_SEED)
lr_proba = predict_proba(X_val, w_lr, b_lr)
record("Logistic Regression", "from scratch (Task 1)", "lr=0.5, l2=1e-5, 300 epochs",
       (lr_proba >= 0.5).astype(int), time.time() - t0, lr_proba)

# ----------------------------------------------------------------------------
# 5. Extra Trees — a library model, allowed because it is an ensemble
# ----------------------------------------------------------------------------
print("\n--- Extra Trees (library ensemble) ---")
t0 = time.time()
extra_trees = ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                   random_state=RANDOM_SEED, n_jobs=-1)
extra_trees.fit(X_train, y_train)
et_proba = extra_trees.predict_proba(X_val)[:, 1]
record("Extra Trees", "library ensemble", "n_estimators=500, max_features=sqrt",
       extra_trees.predict(X_val), time.time() - t0, et_proba)

# ----------------------------------------------------------------------------
# 6. Soft-vote ensemble of the models above
#
# The scores live on different scales, so each one is converted to a rank in
# [0, 1] before averaging. That makes the vote scale-free.
# ----------------------------------------------------------------------------
print("\n--- Soft-vote ensemble ---")


def to_rank(scores):
    order = np.argsort(np.argsort(scores))
    return order / (len(scores) - 1)


members = ["Linear SVM by SGD", "Extra Trees", "Multinomial Naive Bayes"]
t0 = time.time()
vote = np.mean([to_rank(val_scores[name]) for name in members], axis=0)
threshold, tuned = best_threshold(y_val, vote, calculate_macro_f1)
record(f"Soft vote ({len(members)} models)", "ensemble", " + ".join(members),
       (vote >= 0.5).astype(int), time.time() - t0, vote)

results = (pd.DataFrame(rows)
           .sort_values("val_macro_f1_tuned", ascending=False)
           .reset_index(drop=True))
results.to_csv(RESULTS_DIR / "task3_model_comparison.csv", index=False)

print("\n=== Task 3 model comparison (provided 5000 TF-IDF features) ===")
print(results.to_string(index=False))
