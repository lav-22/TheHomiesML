"""Train selected models on all labelled data and create Kaggle predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from extra_trees_and_ensembles import SEED, train_member1_logreg


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "submissions" / "Ensemble2_Prediction.csv"


def load_precomputed_features():
    header = pd.read_csv(DATA / "train_features.csv", nrows=0).columns
    features = [column for column in header if column not in {"id", "label"}]
    dtypes = {column: "float32" for column in features}
    dtypes["label"] = "int8"
    train = pd.read_csv(DATA / "train_features.csv", dtype=dtypes)
    test = pd.read_csv(DATA / "test_features.csv", dtype={c: "float32" for c in features})
    assert train[features].shape == (20_000, 5_000)
    assert test[features].shape == (6_999, 5_000)
    return (
        train[features].to_numpy(dtype=np.float32, copy=False),
        train["label"].to_numpy(dtype=np.int8),
        test[features].to_numpy(dtype=np.float32, copy=False),
        test["id"].copy(),
    )


def svm_test_scores():
    train = pd.read_csv(DATA / "train.csv", usecols=["text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    model = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)
    model.fit(x_train, train["label"])
    return test["id"], expit(model.decision_function(x_test))


def main():
    x_train, y_train, x_test, test_ids = load_precomputed_features()

    print("Training selected Extra Trees model on all 20,000 rows...")
    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        class_weight=None,
        random_state=SEED,
        n_jobs=-1,
    )
    extra_trees.fit(x_train, y_train)
    extra_scores = extra_trees.predict_proba(x_test)[:, 1]

    print("Training Member 1's selected from-scratch logistic regression...")
    logistic_scores = train_member1_logreg(x_train, y_train, x_test)

    print("Training Member 4's selected Linear SVM...")
    raw_test_ids, svm_scores = svm_test_scores()
    assert test_ids.astype("string").tolist() == raw_test_ids.astype("string").tolist()

    ensemble_scores = 0.6 * svm_scores + 0.2 * logistic_scores + 0.2 * extra_scores
    predictions = (ensemble_scores >= 0.5).astype(int)
    submission = pd.DataFrame({"id": test_ids, "label": predictions})

    assert submission.columns.tolist() == ["id", "label"]
    assert len(submission) == 6_999
    assert submission["id"].is_unique and not submission.isna().any().any()
    assert set(submission["label"]).issubset({0, 1})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT, index=False)
    print(f"Saved {len(submission)} predictions to {OUTPUT}")
    print(submission["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
