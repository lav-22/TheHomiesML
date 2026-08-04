"""Tune TF-IDF + RBF-kernel SVM and create Kaggle test predictions."""

from pathlib import Path
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "member5" / "results"
SUBMISSION = ROOT / "submissions" / "TFIDF_RBF_SVM_Prediction.csv"
SEED = 42


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def make_model(c_value):
    return SVC(
        C=c_value,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        cache_size=2048,
        random_state=SEED,
    )


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")
    assert len(train) == len(split) == 20_000
    assert train["id"].astype("string").tolist() == split["id"].astype("string").tolist()

    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train.loc[train_mask, "text"])
    x_val = vectorizer.transform(train.loc[val_mask, "text"])
    y_train = train.loc[train_mask, "label"]
    y_val = train.loc[val_mask, "label"]

    # One standard configuration is intentional: exact RBF SVM inference is
    # quadratic in the number of support vectors on this dataset, so a broad
    # sweep would add hours without being required for this comparison.
    rows = []
    for c_value in (1.0,):
        start = time.perf_counter()
        model = make_model(c_value)
        model.fit(x_train, y_train)
        val_score = macro_f1(y_val, model.predict(x_val))
        rows.append({
            "C": c_value,
            "gamma": "scale",
            "class_weight": "balanced",
            "validation_macro_f1": val_score,
            "fit_seconds": time.perf_counter() - start,
        })
        print(
            f"C={c_value}: validation Macro F1={val_score:.6f}"
        )

    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "rbf_svm_results.csv", index=False)
    best_c = float(results.iloc[0]["C"])

    print(f"Retraining selected C={best_c} model on all 20,000 labelled rows...")
    final_vectorizer = TfidfVectorizer()
    x_all = final_vectorizer.fit_transform(train["text"])
    x_test = final_vectorizer.transform(test["text"])
    final_model = make_model(best_c)
    final_model.fit(x_all, train["label"])
    predictions = final_model.predict(x_test).astype(int)

    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    assert submission.columns.tolist() == ["id", "label"]
    assert len(submission) == 6_999 and submission["id"].is_unique
    assert not submission.isna().any().any()
    assert set(submission["label"]).issubset({0, 1})
    submission.to_csv(SUBMISSION, index=False)
    print(f"Saved {len(submission)} predictions to {SUBMISSION}")
    print(submission["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
