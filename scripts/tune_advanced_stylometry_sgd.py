"""Tune advanced-stylometry weight and SGD decision threshold."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from advanced_stylometry_sgd import combine, save_submission, style_matrix
from hybrid_tfidf_linear_svm import build_vectorizer
from hybrid_tfidf_sgd import make_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MINIMUM_F1 = 0.85


def macro_f1(y_true, scores, threshold):
    predictions = np.asarray(scores >= threshold, dtype=int)
    return float(f1_score(y_true, predictions, average="macro", zero_division=0))


def tune_threshold(y_true, scores):
    # A deliberately small grid limits overfitting to the fixed validation set.
    thresholds = np.linspace(-0.25, 0.25, 41)
    rows = [
        (float(threshold), macro_f1(y_true, scores, threshold))
        for threshold in thresholds
    ]
    return max(rows, key=lambda item: item[1])


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")
    assert train["id"].astype("string").tolist() == split["id"].astype("string").tolist()

    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()
    train_text = train.loc[train_mask, "text"]
    val_text = train.loc[val_mask, "text"]
    y_train = train.loc[train_mask, "label"]
    y_val = train.loc[val_mask, "label"]

    vectorizer = build_vectorizer()
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    scaler = StandardScaler()
    train_styles = scaler.fit_transform(style_matrix(train_text))
    val_styles = scaler.transform(style_matrix(val_text))

    rows = []
    for weight in (0.10, 0.15, 0.20, 0.30):
        x_train = combine(train_tfidf, train_styles, weight)
        x_val = combine(val_tfidf, val_styles, weight)
        start = time.perf_counter()
        model = make_model("hinge", 1e-4)
        model.fit(x_train, y_train)
        scores = model.decision_function(x_val)
        default_score = macro_f1(y_val, scores, 0.0)
        threshold, tuned_score = tune_threshold(y_val, scores)
        rows.append({
            "style_weight": weight,
            "default_threshold": 0.0,
            "default_validation_macro_f1": default_score,
            "best_threshold": threshold,
            "tuned_validation_macro_f1": tuned_score,
            "fit_seconds": time.perf_counter() - start,
        })
        print(
            f"weight={weight:.2f}: default F1={default_score:.6f}; "
            f"threshold={threshold:.4f}, tuned F1={tuned_score:.6f}"
        )

    results = pd.DataFrame(rows).sort_values(
        "tuned_validation_macro_f1", ascending=False
    )
    results.to_csv(RESULTS / "advanced_stylometry_threshold_results.csv", index=False)
    best = results.iloc[0]
    best_score = float(best["tuned_validation_macro_f1"])
    if best_score <= MINIMUM_F1:
        print(
            f"Best tuned validation Macro F1={best_score:.6f} did not exceed "
            f"{MINIMUM_F1:.2f}; no new Kaggle files were generated."
        )
        return

    print(
        f"Validation target passed: F1={best_score:.6f}, "
        f"style_weight={best['style_weight']:.2f}, "
        f"threshold={best['best_threshold']:.4f}"
    )
    final_vectorizer = build_vectorizer()
    all_tfidf = final_vectorizer.fit_transform(train["text"])
    test_tfidf = final_vectorizer.transform(test["text"])
    final_scaler = StandardScaler()
    all_styles = final_scaler.fit_transform(style_matrix(train["text"]))
    test_styles = final_scaler.transform(style_matrix(test["text"]))
    x_all = combine(all_tfidf, all_styles, float(best["style_weight"]))
    x_test = combine(test_tfidf, test_styles, float(best["style_weight"]))

    model = make_model("hinge", 1e-4)
    model.fit(x_all, train["label"])
    test_scores = model.decision_function(x_test)
    save_submission(
        "Advanced_Stylometry_TFIDF_SGD_TunedThreshold_Prediction.csv",
        test["id"],
        test_scores >= float(best["best_threshold"]),
    )
    rank55_cutoff = np.quantile(test_scores, 0.45)
    save_submission(
        "Advanced_Stylometry_TFIDF_SGD_TunedWeight_Rank55_Prediction.csv",
        test["id"],
        test_scores >= rank55_cutoff,
    )


if __name__ == "__main__":
    main()
