"""Hybrid word/character TF-IDF plus stylometry with an SGD classifier."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from hybrid_tfidf_linear_svm import STYLE_WEIGHT, build_vectorizer, style_matrix
from hybrid_tfidf_sgd import make_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def add_stylometry(tfidf, texts, scaler, fit):
    raw_style = style_matrix(texts)
    scaled_style = (
        scaler.fit_transform(raw_style) if fit else scaler.transform(raw_style)
    )
    scaled_style *= STYLE_WEIGHT
    return sparse.hstack(
        [tfidf, sparse.csr_matrix(scaled_style)], format="csr"
    )


def save_submission(filename, ids, predictions):
    frame = pd.DataFrame({"id": ids, "label": np.asarray(predictions, dtype=int)})
    assert frame.columns.tolist() == ["id", "label"]
    assert len(frame) == 6_999 and frame["id"].is_unique
    assert not frame.isna().any().any() and set(frame["label"]) <= {0, 1}
    path = SUBMISSIONS / filename
    frame.to_csv(path, index=False)
    print(f"Saved {path}; class-1 rate={frame['label'].mean():.4f}")


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")
    assert len(train) == len(split) == 20_000
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
    x_train = add_stylometry(train_tfidf, train_text, scaler, fit=True)
    x_val = add_stylometry(val_tfidf, val_text, scaler, fit=False)

    rows = []
    for loss in ("hinge", "modified_huber"):
        for alpha in (1e-5, 1e-4):
            start = time.perf_counter()
            model = make_model(loss, alpha)
            model.fit(x_train, y_train)
            score = macro_f1(y_val, model.predict(x_val))
            rows.append({
                "loss": loss,
                "alpha": alpha,
                "style_weight": STYLE_WEIGHT,
                "validation_macro_f1": score,
                "fit_seconds": time.perf_counter() - start,
            })
            print(f"loss={loss}, alpha={alpha:g}: validation Macro F1={score:.6f}")

    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    results.to_csv(RESULTS / "hybrid_stylometry_sgd_results.csv", index=False)
    best = results.iloc[0]

    final_vectorizer = build_vectorizer()
    all_tfidf = final_vectorizer.fit_transform(train["text"])
    test_tfidf = final_vectorizer.transform(test["text"])
    final_scaler = StandardScaler()
    x_all = add_stylometry(all_tfidf, train["text"], final_scaler, fit=True)
    x_test = add_stylometry(test_tfidf, test["text"], final_scaler, fit=False)

    final_model = make_model(str(best["loss"]), float(best["alpha"]))
    final_model.fit(x_all, train["label"])
    scores = final_model.decision_function(x_test)

    save_submission(
        "Hybrid_TFIDF_Stylometry_SGD_Prediction.csv",
        test["id"],
        scores >= 0,
    )
    cutoff = np.quantile(scores, 0.45)
    save_submission(
        "Hybrid_TFIDF_Stylometry_SGD_Rank55_Prediction.csv",
        test["id"],
        scores >= cutoff,
    )


if __name__ == "__main__":
    main()
