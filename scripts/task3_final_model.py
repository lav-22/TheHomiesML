"""Task 3: the final model, selected on the shared validation split.

Everything that learns here is written from scratch: the TF-IDF vectorizer, the
feature scaler and the stylometry features come from src/text_features.py, and
the multi-loss mini-batch SGD trainer from src/scratch_models.py. NumPy and
SciPy provide array and sparse-matrix storage only.

The model is a linear classifier on word + character TF-IDF plus 78 stylometry
features. Two things are chosen on validation and nothing is read from the test
labels:

1. `style_weight`, which scales the stylometry block against the TF-IDF block,
2. the number of epochs.

`style_weight` is chosen on the cluster-holdout score from
task3_shift_validation.py rather than on the random split, because the random
split cannot see the domain shift that separates our validation score from the
leaderboard.

The loss is modified Huber. We originally used hinge here and only later
re-tested the loss on these features, which turned out to be worth about +0.02
Macro F1 — see scripts/task3_loss_on_hybrid.py.

Run from the repository root:

    python3 scripts/task3_final_model.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation import calculate_macro_f1
from src.scratch_models import best_threshold, sgd_decision_function, sgd_fit
from src.submission import create_submission
from src.text_features import (
    ScratchHybridTfidf,
    ScratchStandardScaler,
    combine,
    style_matrix,
)

DATA = ROOT / "data"
RESULTS = ROOT / "extra" / "results"
SUBMISSIONS = ROOT / "submissions"

RANDOM_SEED = 42
LOSS = "modified_huber"
LEARNING_RATE = 0.5
ALPHA = 1e-5
BATCH_SIZE = 256
# Fixed, not re-tuned here. The random split ranks 0.10 and 0.25 as a tie
# (0.8513 vs 0.8471) but cannot see domain shift; the cluster-holdout in
# task3_shift_validation.py prefers 0.25 (0.7958 vs 0.7931), which is the
# regime the leaderboard is in. Both differences are small, so we take the
# choice that leans on the domain-independent stylometry block.
STYLE_WEIGHT = 0.25
EPOCHS = 60          # start of the plateau; flat out to 200 epochs


def fit_and_score(x_train, y_train, x_val, y_val, epochs):
    """Train one configuration and score it with and without a tuned cut-off."""
    start = time.perf_counter()
    w, b, _, _ = sgd_fit(x_train, y_train, loss=LOSS, penalty="l2", alpha=ALPHA,
                         class_weight="balanced", lr=LEARNING_RATE, epochs=epochs,
                         bs=BATCH_SIZE, random_state=RANDOM_SEED)
    scores = sgd_decision_function(x_val, w, b)
    default_f1 = calculate_macro_f1(y_val, (scores >= 0).astype(int))
    threshold, tuned_f1 = best_threshold(y_val, scores, calculate_macro_f1)
    return {
        "epochs": epochs,
        "val_macro_f1": default_f1,
        "best_threshold": threshold,
        "val_macro_f1_tuned": tuned_f1,
        "fit_seconds": round(time.perf_counter() - start, 1),
    }


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")
    assert train["id"].astype("string").tolist() == split["id"].astype("string").tolist()

    train_mask = split["split"].eq("train").to_numpy()
    train_text = train.loc[train_mask, "text"]
    val_text = train.loc[~train_mask, "text"]
    y_train = train.loc[train_mask, "label"].to_numpy()
    y_val = train.loc[~train_mask, "label"].to_numpy()

    # The expensive step, so it is done once and reused for every setting
    print("Vectorizing the training split...")
    vectorizer = ScratchHybridTfidf()
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    scaler = ScratchStandardScaler()
    train_styles = scaler.fit_transform(style_matrix(train_text))
    val_styles = scaler.transform(style_matrix(val_text))
    print(f"  TF-IDF block: {train_tfidf.shape}, stylometry block: {train_styles.shape}")

    # One configuration, reported rather than selected: the choice was already
    # made on the cluster-holdout score, so there is nothing to tune here.
    result = fit_and_score(combine(train_tfidf, train_styles, STYLE_WEIGHT), y_train,
                           combine(val_tfidf, val_styles, STYLE_WEIGHT), y_val, EPOCHS)
    result["style_weight"] = STYLE_WEIGHT
    results = pd.DataFrame([result])
    results.to_csv(RESULTS / "task3_final_model_results.csv", index=False)
    print("\n", results.to_string(index=False), sep="")

    best_weight, best_epochs = STYLE_WEIGHT, EPOCHS

    # A threshold fitted to 4000 validation rows is as likely to be noise as
    # signal, so it is only adopted when it clears that noise.
    THRESHOLD_MARGIN = 0.005
    threshold_gain = result["val_macro_f1_tuned"] - result["val_macro_f1"]
    if threshold_gain > THRESHOLD_MARGIN:
        best_threshold_value = result["best_threshold"]
        print(f"\nTuned threshold adopted: it gains {threshold_gain:+.4f}")
    else:
        best_threshold_value = 0.0
        print(f"\nTuned threshold rejected: it only gains {threshold_gain:+.4f}, "
              f"which is inside the noise of a 4000-row split")

    print(f"style_weight={best_weight}, epochs={best_epochs}, "
          f"threshold={best_threshold_value:.4f}, "
          f"random-split Macro F1={result['val_macro_f1']:.6f}")

    # Refit on all 20000 labelled rows with the chosen settings
    print("\nRefitting on the full training set...")
    final_vectorizer = ScratchHybridTfidf()
    all_tfidf = final_vectorizer.fit_transform(train["text"])
    test_tfidf = final_vectorizer.transform(test["text"])
    final_scaler = ScratchStandardScaler()
    all_styles = final_scaler.fit_transform(style_matrix(train["text"]))
    test_styles = final_scaler.transform(style_matrix(test["text"]))

    x_all = combine(all_tfidf, all_styles, best_weight)
    x_test = combine(test_tfidf, test_styles, best_weight)

    w, b, _, _ = sgd_fit(x_all, train["label"].to_numpy(), loss=LOSS, penalty="l2",
                         alpha=ALPHA, class_weight="balanced", lr=LEARNING_RATE,
                         epochs=best_epochs, bs=BATCH_SIZE, random_state=RANDOM_SEED)
    test_scores = sgd_decision_function(x_test, w, b)
    test_preds = (test_scores >= best_threshold_value).astype(int)

    create_submission(
        test_ids=test["id"],
        predictions=test_preds,
        output_path=SUBMISSIONS / "Final_Prediction.csv",
        id_column="id",
        label_column="label",
    )
    print(f"Saved Final_Prediction.csv; class-1 rate={test_preds.mean():.4f}")


if __name__ == "__main__":
    main()
