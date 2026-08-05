"""Tune word/character TF-IDF with SGD and generate Kaggle predictions."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"
SEED = 42


def build_features():
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=100_000,
                dtype=np.float32,
            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=100_000,
                dtype=np.float32,
            ),
        ),
    ])


def make_model(loss, alpha):
    return SGDClassifier(
        loss=loss,
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=2_000,
        tol=1e-4,
        random_state=SEED,
        average=True,
    )


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def save_submission(name, ids, predictions):
    frame = pd.DataFrame({"id": ids, "label": np.asarray(predictions, dtype=int)})
    assert frame.columns.tolist() == ["id", "label"]
    assert len(frame) == 6_999 and frame["id"].is_unique
    assert not frame.isna().any().any() and set(frame["label"]) <= {0, 1}
    path = SUBMISSIONS / name
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

    features = build_features()
    x_train = features.fit_transform(train.loc[train_mask, "text"])
    x_val = features.transform(train.loc[val_mask, "text"])
    y_train = train.loc[train_mask, "label"]
    y_val = train.loc[val_mask, "label"]

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
                "validation_macro_f1": score,
                "fit_seconds": time.perf_counter() - start,
            })
            print(f"loss={loss}, alpha={alpha:g}: validation Macro F1={score:.6f}")

    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "hybrid_sgd_results.csv", index=False)
    best = results.iloc[0]

    final_features = build_features()
    x_all = final_features.fit_transform(train["text"])
    x_test = final_features.transform(test["text"])
    final_model = make_model(str(best["loss"]), float(best["alpha"]))
    final_model.fit(x_all, train["label"])
    scores = final_model.decision_function(x_test)

    save_submission(
        "Hybrid_Word_Char_TFIDF_SGD_Prediction.csv",
        test["id"],
        scores >= 0,
    )

    # Optional domain-shift variant matching the successful SGD submission's
    # approximately 55% class-1 prediction rate. Ranking preserves the model's
    # ordering and changes only the final decision threshold.
    cutoff = np.quantile(scores, 0.45)
    save_submission(
        "Hybrid_Word_Char_TFIDF_SGD_Rank55_Prediction.csv",
        test["id"],
        scores >= cutoff,
    )


if __name__ == "__main__":
    main()
