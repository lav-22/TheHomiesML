"""Tune a small XGBoost classifier set and generate Kaggle test predictions."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSION = ROOT / "submissions" / "XGBoost_Prediction.csv"
SEED = 42


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def load_data():
    header = pd.read_csv(DATA / "train_features.csv", nrows=0).columns
    features = [column for column in header if column not in {"id", "label"}]
    train_types = {column: "float32" for column in features}
    train_types["label"] = "int8"
    train = pd.read_csv(DATA / "train_features.csv", dtype=train_types)
    test = pd.read_csv(DATA / "test_features.csv", dtype={c: "float32" for c in features})
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")

    assert len(train) == len(split) == 20_000
    assert train["id"].astype("string").tolist() == split["id"].astype("string").tolist()
    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()

    # TF-IDF is mostly zero; CSR avoids making XGBoost repeatedly scan zeros.
    x_all = sparse.csr_matrix(train[features].to_numpy(dtype=np.float32, copy=False))
    x_test = sparse.csr_matrix(test[features].to_numpy(dtype=np.float32, copy=False))
    y_all = train["label"].to_numpy(dtype=np.int8)
    return x_all, y_all, x_test, test["id"].copy(), train_mask, val_mask


def make_model(params, early_stopping_rounds=None):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    x_all, y_all, x_test, test_ids, train_mask, val_mask = load_data()
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    # Small, sensible comparison only; no broad grid search.
    candidates = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.10},
    ]
    rows = []
    for params in candidates:
        start = time.perf_counter()
        model = make_model(params, early_stopping_rounds=30)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        predictions = model.predict(x_val)
        selected_trees = int(model.best_iteration + 1)
        score = macro_f1(y_val, predictions)
        rows.append({
            **params,
            "selected_trees": selected_trees,
            "validation_macro_f1": score,
            "fit_seconds": time.perf_counter() - start,
        })
        print(params, f"selected_trees={selected_trees}, validation Macro F1={score:.6f}")

    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    results.to_csv(RESULTS / "xgboost_results.csv", index=False)
    best = results.iloc[0]
    final_params = {
        "n_estimators": int(best["selected_trees"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
    }

    print("Retraining selected XGBoost configuration on all labelled rows:", final_params)
    final_model = make_model(final_params)
    final_model.fit(x_all, y_all, verbose=False)
    test_predictions = final_model.predict(x_test).astype(int)

    submission = pd.DataFrame({"id": test_ids, "label": test_predictions})
    assert submission.columns.tolist() == ["id", "label"]
    assert len(submission) == 6_999 and submission["id"].is_unique
    assert not submission.isna().any().any()
    assert set(submission["label"]).issubset({0, 1})
    submission.to_csv(SUBMISSION, index=False)
    print(f"Saved {len(submission)} predictions to {SUBMISSION}")
    print(submission["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
