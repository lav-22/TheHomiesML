"""Member 5 Extra Trees tuning and validation ensembles.

This script deliberately uses a small, assignment-driven search rather than a
large hyperparameter grid. Model selection uses validation Macro F1 only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "member5" / "results"
SEED = 42


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def load_fixed_split():
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")
    header = pd.read_csv(DATA / "train_features.csv", nrows=0).columns
    feature_columns = [c for c in header if c not in {"id", "label"}]
    dtypes = {c: "float32" for c in feature_columns}
    dtypes.update({"id": "string", "label": "int8"})
    frame = pd.read_csv(DATA / "train_features.csv", dtype=dtypes)

    assert len(frame) == len(split) == 20_000
    assert split["row_index"].tolist() == list(range(len(split)))
    assert frame["id"].tolist() == split["id"].astype("string").tolist()
    assert set(split["split"]) == {"train", "validation"}

    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()
    assert train_mask.sum() == 16_000 and val_mask.sum() == 4_000

    x = frame[feature_columns].to_numpy(dtype=np.float32, copy=False)
    y = frame["label"].to_numpy(dtype=np.int8)
    ids = frame["id"].to_numpy()
    return x[train_mask], x[val_mask], y[train_mask], y[val_mask], ids[val_mask], train_mask, val_mask


def run_extra_trees(x_train, x_val, y_train, y_val):
    rows = []
    fitted = {}

    def evaluate(stage, **params):
        key = tuple(sorted(params.items()))
        if key in fitted:
            model, train_score, val_score, elapsed = fitted[key]
        else:
            start = time.perf_counter()
            model = ExtraTreesClassifier(random_state=SEED, n_jobs=-1, **params)
            model.fit(x_train, y_train)
            elapsed = time.perf_counter() - start
            train_score = macro_f1(y_train, model.predict(x_train))
            val_score = macro_f1(y_val, model.predict(x_val))
            fitted[key] = model, train_score, val_score, elapsed
        rows.append({
            "stage": stage,
            **params,
            "training_macro_f1": train_score,
            "validation_macro_f1": val_score,
            "fit_seconds": elapsed,
        })
        print(stage, params, f"validation Macro F1={val_score:.6f}")
        return model, val_score

    baseline = dict(n_estimators=200, max_depth=None, max_features="sqrt", class_weight=None)
    baseline_model, _ = evaluate("baseline", **baseline)

    # Grow one forest for the tree-count study instead of rebuilding its first
    # 200/500 trees. This changes no model setting and saves substantial time.
    tree_model = baseline_model
    tree_model.set_params(warm_start=True)
    tree_models = {200: tree_model}
    for value in (200, 500, 800):
        if value > 200:
            start = time.perf_counter()
            tree_model.set_params(n_estimators=value)
            tree_model.fit(x_train, y_train)
            elapsed = time.perf_counter() - start
            train_score = macro_f1(y_train, tree_model.predict(x_train))
            val_score = macro_f1(y_val, tree_model.predict(x_val))
            rows.append({
                "stage": "n_estimators",
                **(baseline | {"n_estimators": value}),
                "training_macro_f1": train_score,
                "validation_macro_f1": val_score,
                "fit_seconds": elapsed,
            })
            print("n_estimators", value, f"validation Macro F1={val_score:.6f}")
        else:
            rows.append(rows[0] | {"stage": "n_estimators"})
        if value == 800:
            tree_models[value] = tree_model

    # All remaining studies hold the baseline settings constant so each table
    # isolates the requested parameter rather than becoming a grid search.
    for value in (None, 20, 40):
        evaluate("max_depth", **(baseline | {"max_depth": value}))

    for value in ("sqrt", "log2", 100):
        evaluate("max_features", **(baseline | {"max_features": value}))

    for value in (None, "balanced"):
        evaluate("class_weight", **(baseline | {"class_weight": value}))

    results = pd.DataFrame(rows).drop_duplicates(
        subset=["stage", "n_estimators", "max_depth", "max_features", "class_weight"]
    )
    best_row = results.loc[results["validation_macro_f1"].idxmax()]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "max_features": best_row["max_features"],
        "class_weight": None if pd.isna(best_row["class_weight"]) else best_row["class_weight"],
    }
    key = tuple(sorted(best_params.items()))
    if best_params["n_estimators"] == 800 and best_params == (baseline | {"n_estimators": 800}):
        best_model = tree_models[800]
    elif key in fitted and best_params != baseline:
        best_model = fitted[key][0]
    else:
        best_model = ExtraTreesClassifier(random_state=SEED, n_jobs=-1, **best_params).fit(x_train, y_train)
    return best_model, best_params, results


def train_member1_logreg(x_train, y_train, x_val):
    """Reproduce Member 1's selected from-scratch configuration."""
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0
    batch_size, epochs, learning_rate = 64, 300, 1.0
    for _ in range(epochs):
        for start in range(0, len(y_train), batch_size):
            xb = x_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]
            probabilities = expit(xb @ weights + bias)
            error = probabilities - yb
            weights -= learning_rate * ((xb.T @ error) / len(yb))
            bias -= learning_rate * error.mean()
    return expit(x_val @ weights + bias)


def train_member4_svm(train_mask, val_mask):
    """Reproduce Member 4's selected raw-text SVM configuration."""
    raw = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(raw.loc[train_mask, "text"])
    x_val = vectorizer.transform(raw.loc[val_mask, "text"])
    model = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)
    model.fit(x_train, raw.loc[train_mask, "label"])
    return expit(model.decision_function(x_val))


def evaluate_ensembles(y_val, scores):
    individual = []
    for name, score in scores.items():
        # ExtraTrees.predict resolves exact 0.5 vote ties to class 0.
        prediction = score > 0.5 if name == "Extra Trees" else score >= 0.5
        individual.append({
            "model": name,
            "validation_macro_f1": macro_f1(y_val, prediction),
            "weights": "individual",
        })
    ranked = sorted(individual, key=lambda row: row["validation_macro_f1"], reverse=True)
    names = [row["model"] for row in ranked]

    rows = list(individual)
    two_weight_sets = ((0.5, 0.5), (0.6, 0.4), (0.7, 0.3))
    for weights in two_weight_sets:
        combined = sum(w * scores[n] for w, n in zip(weights, names[:2]))
        rows.append({
            "model": "Ensemble 1",
            "validation_macro_f1": macro_f1(y_val, combined >= 0.5),
            "weights": json.dumps(dict(zip(names[:2], weights))),
        })

    three_weight_sets = ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.3, 0.2), (0.6, 0.2, 0.2))
    for weights in three_weight_sets:
        combined = sum(w * scores[n] for w, n in zip(weights, names))
        rows.append({
            "model": "Ensemble 2",
            "validation_macro_f1": macro_f1(y_val, combined >= 0.5),
            "weights": json.dumps(dict(zip(names, weights))),
        })
    return pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    x_train, x_val, y_train, y_val, val_ids, train_mask, val_mask = load_fixed_split()
    print(f"Fixed split loaded: train={len(y_train)}, validation={len(y_val)}")

    extra_model, best_params, extra_results = run_extra_trees(x_train, x_val, y_train, y_val)
    extra_results.to_csv(RESULTS / "extra_trees_results.csv", index=False)
    (RESULTS / "best_extra_trees.json").write_text(
        json.dumps(best_params, indent=2, default=str) + "\n", encoding="utf-8"
    )

    scores = {
        "Extra Trees": extra_model.predict_proba(x_val)[:, 1],
        "Logistic Regression": train_member1_logreg(x_train, y_train, x_val),
        "Linear SVM": train_member4_svm(train_mask, val_mask),
    }
    validation = pd.DataFrame({"id": val_ids, "y_true": y_val})
    for name, score in scores.items():
        slug = name.lower().replace(" ", "_")
        validation[f"{slug}_score"] = score
        thresholded = score > 0.5 if name == "Extra Trees" else score >= 0.5
        validation[f"{slug}_prediction"] = thresholded.astype(int)
    validation.to_csv(RESULTS / "validation_predictions.csv", index=False)

    comparison = evaluate_ensembles(y_val, scores)
    comparison.to_csv(RESULTS / "final_comparison.csv", index=False)
    print("\nBest Extra Trees configuration:", best_params)
    print("\nFinal comparison:\n", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
