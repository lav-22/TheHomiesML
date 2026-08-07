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

ROOT = Path(__file__).resolve().parents[1]
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
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"

RANDOM_SEED = 42
LOSS = "modified_huber"
LEARNING_RATE = 0.5
ALPHA = 1e-5
BATCH_SIZE = 256
STYLE_WEIGHTS = (0.05, 0.10)
EPOCH_SETTINGS = (30, 60, 100)


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

    rows = []
    for weight in STYLE_WEIGHTS:
        x_train = combine(train_tfidf, train_styles, weight)
        x_val = combine(val_tfidf, val_styles, weight)
        for epochs in EPOCH_SETTINGS:
            result = fit_and_score(x_train, y_train, x_val, y_val, epochs)
            result["style_weight"] = weight
            rows.append(result)
            print(f"style_weight={weight:<5} epochs={epochs:<4} "
                  f"F1={result['val_macro_f1']:.4f} tuned={result['val_macro_f1_tuned']:.4f}")

    # Select on the *untuned* score. A threshold picked on 4000 validation rows
    # carries real overfitting risk, so we only adopt one when it earns more
    # than the noise of that split — otherwise we keep the plain cut-off of 0.
    THRESHOLD_MARGIN = 0.005

    results = (pd.DataFrame(rows)
               .sort_values("val_macro_f1", ascending=False)
               .reset_index(drop=True))
    results.to_csv(RESULTS / "task3_final_model_results.csv", index=False)
    print("\n", results.to_string(index=False), sep="")

    best = results.iloc[0]
    best_weight = float(best["style_weight"])
    best_epochs = int(best["epochs"])

    threshold_gain = float(best["val_macro_f1_tuned"]) - float(best["val_macro_f1"])
    if threshold_gain > THRESHOLD_MARGIN:
        best_threshold_value = float(best["best_threshold"])
        selected_f1 = float(best["val_macro_f1_tuned"])
        print(f"\nTuned threshold adopted: it gains {threshold_gain:+.4f}")
    else:
        best_threshold_value = 0.0
        selected_f1 = float(best["val_macro_f1"])
        print(f"\nTuned threshold rejected: it only gains {threshold_gain:+.4f}, "
              f"which is inside the noise of a 4000-row split")

    print(f"Selected style_weight={best_weight}, epochs={best_epochs}, "
          f"threshold={best_threshold_value:.4f}, "
          f"validation Macro F1={selected_f1:.6f}")

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
