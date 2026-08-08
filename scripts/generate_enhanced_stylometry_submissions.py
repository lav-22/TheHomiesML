"""Generate test predictions for the best enhanced-stylometry configuration."""

from pathlib import Path
import gzip
import pickle
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = PROJECT / "submissions"
MODEL_OUTPUT = PROJECT / "models" / "enhanced_from_scratch"
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(HERE))

from advanced_stylometry_sgd import (  # noqa: E402
    ScratchAveragedHingeSGD,
    ScratchStandardScaler,
    combine,
    style_matrix,
)
from scratch_feature_space_search import make_features, transform  # noqa: E402
from scratch_final_ensemble_search import extra_style  # noqa: E402


WORD_WEIGHT = 1.50
STYLE_WEIGHT = 0.10
ALPHA = 1e-4
LEARNING_RATE = 50.0
BATCH_SIZE = 256
EPOCHS = 150
TUNED_THRESHOLD = -0.014493


def save_submission(filename, ids, predictions):
    frame = pd.DataFrame({"id": ids, "label": np.asarray(predictions, dtype=int)})
    if frame.columns.tolist() != ["id", "label"]:
        raise ValueError("Submission must contain id,label columns.")
    if len(frame) != 6_999 or not frame["id"].is_unique:
        raise ValueError("Submission must contain 6,999 unique test IDs.")
    if frame.isna().any().any() or not set(frame["label"]) <= {0, 1}:
        raise ValueError("Submission contains missing or invalid labels.")
    path = OUTPUT / filename
    frame.to_csv(path, index=False)
    print(f"Saved {path}; class-1 rate={frame['label'].mean():.4f}")


def main():
    train = pd.read_csv(PROJECT / "data/train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(PROJECT / "data/test.csv", usecols=["id", "text"])

    # Best feature space: word 1-3 grams and raw character 3-5 grams.
    word, character, train_tfidf = make_features(
        train["text"], (1, 3), (3, 5), True, 160_000
    )
    test_tfidf = transform(word, character, test["text"])

    # Apply the validation-selected 1.50 multiplier to word TF-IDF columns.
    word_columns = len(word.vocabulary_)
    for matrix in (train_tfidf, test_tfidf):
        for row in range(matrix.shape[0]):
            word_mask = matrix.row_indices[row] < word_columns
            matrix.row_values[row][word_mask] *= WORD_WEIGHT

    # Combine the original stylometry with the additional scientific-format
    # and AI-phrase indicators. Scaling is fitted on labelled training data.
    train_style_raw = np.column_stack([
        style_matrix(train["text"]),
        np.asarray([extra_style(text) for text in train["text"]]),
    ])
    test_style_raw = np.column_stack([
        style_matrix(test["text"]),
        np.asarray([extra_style(text) for text in test["text"]]),
    ])
    scaler = ScratchStandardScaler()
    train_style = scaler.fit_transform(train_style_raw)
    test_style = scaler.transform(test_style_raw)
    x_train = combine(train_tfidf, train_style, STYLE_WEIGHT)
    x_test = combine(test_tfidf, test_style, STYLE_WEIGHT)

    model = ScratchAveragedHingeSGD(
        alpha=ALPHA,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        random_state=42,
        tolerance=1e-5,
    )
    model.fit(x_train, train["label"])
    scores = model.decision_function(x_test)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_submission(
        "Enhanced_Stylometry_FromScratch_TunedThreshold_Prediction.csv",
        test["id"],
        scores >= TUNED_THRESHOLD,
    )
    rank55_threshold = float(np.quantile(scores, 0.45))
    save_submission(
        "Enhanced_Stylometry_FromScratch_Rank55_Prediction.csv",
        test["id"],
        scores >= rank55_threshold,
    )
    pd.DataFrame({"id": test["id"], "decision_score": scores}).to_csv(
        OUTPUT / "Enhanced_Stylometry_FromScratch_DecisionScores.csv", index=False
    )

    # Save every fitted component required for later inference. The classes are
    # project-owned implementations, so loading requires this scripts directory
    # to be available on Python's import path.
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "word_vectorizer": word,
        "character_vectorizer": character,
        "style_scaler": scaler,
        "classifier": model,
        "hyperparameters": {
            "word_ngram_range": (1, 3),
            "character_ngram_range": (3, 5),
            "max_features_per_family": 160_000,
            "word_weight": WORD_WEIGHT,
            "style_weight": STYLE_WEIGHT,
            "loss": "hinge",
            "alpha": ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "tuned_threshold": TUNED_THRESHOLD,
        },
    }
    with gzip.open(MODEL_OUTPUT / "enhanced_stylometry_sgd.pkl.gz", "wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"epochs_run={model.n_iter_}; rank55_threshold={rank55_threshold:.6f}")


if __name__ == "__main__":
    main()
