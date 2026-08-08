"""Hyperparameter search for the fully from-scratch AI-text detector.

No scikit-learn, SciPy, Optuna, or pretrained/deep model is used. The search
tests ordinary TF-IDF and NBSVM-style class-conditional feature reweighting.
"""

from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from advanced_stylometry_sgd import (  # noqa: E402
    ScratchAveragedHingeSGD,
    ScratchSparseMatrix,
    ScratchStandardScaler,
    build_vectorizer,
    combine,
    macro_f1_score,
    style_matrix,
)


OUTPUT = PROJECT / "results"
SEED = 42


def column_scale(matrix, scales):
    """Return a sparse matrix with each column multiplied by its scale."""
    return ScratchSparseMatrix(
        matrix.row_indices,
        [values * scales[indices]
         for indices, values in zip(matrix.row_indices, matrix.row_values)],
        matrix.shape[1],
    )


def nb_log_count_ratio(matrix, labels):
    """Compute smoothed class-conditional log-count ratios from training only."""
    labels = np.asarray(labels, dtype=int)
    positive = np.ones(matrix.shape[1], dtype=np.float64)
    negative = np.ones(matrix.shape[1], dtype=np.float64)
    for row, label in enumerate(labels):
        target = positive if label == 1 else negative
        # Binary occurrence counts avoid long documents dominating the ratio.
        np.add.at(target, matrix.row_indices[row], 1.0)
    positive /= positive.sum()
    negative /= negative.sum()
    return np.log(positive / negative).astype(np.float32)


def tuned_macro_f1(labels, scores):
    """Select a threshold from validation score quantiles."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.15, 0.85, 281)))
    best_threshold, best_score = 0.0, -1.0
    for threshold in candidates:
        score = macro_f1_score(labels, scores >= threshold)
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, float(best_score)


def trial_configs():
    # Broad deterministic coverage, followed by focused variations around the
    # strongest previous schedule. This is reproducible adaptive-style search.
    broad = [
        # name, NB ratio, word weight, char weight, style weight,
        # alpha, learning rate, batch size, epochs
        ("baseline", False, 1.0, 1.0, .10, 1e-4, 20., 256, 100),
        ("lower_alpha", False, 1.0, 1.0, .10, 3e-5, 20., 256, 120),
        ("lowest_alpha", False, 1.0, 1.0, .10, 1e-5, 20., 256, 120),
        ("higher_alpha", False, 1.0, 1.0, .10, 3e-4, 20., 256, 100),
        ("lr10", False, 1.0, 1.0, .10, 1e-4, 10., 256, 120),
        ("lr35", False, 1.0, 1.0, .10, 1e-4, 35., 256, 120),
        ("lr50", False, 1.0, 1.0, .10, 1e-4, 50., 256, 120),
        ("batch128", False, 1.0, 1.0, .10, 1e-4, 20., 128, 120),
        ("batch512", False, 1.0, 1.0, .10, 1e-4, 20., 512, 120),
        ("char125", False, 1.0, 1.25, .10, 1e-4, 20., 256, 120),
        ("char150", False, 1.0, 1.50, .10, 1e-4, 20., 256, 120),
        ("word125", False, 1.25, 1.0, .10, 1e-4, 20., 256, 120),
        ("style05", False, 1.0, 1.0, .05, 1e-4, 20., 256, 120),
        ("style075", False, 1.0, 1.0, .075, 1e-4, 20., 256, 120),
        ("nbsvm", True, 1.0, 1.0, .10, 1e-4, 20., 256, 120),
        ("nbsvm_low_alpha", True, 1.0, 1.0, .10, 3e-5, 20., 256, 120),
        ("nbsvm_lr10", True, 1.0, 1.0, .10, 1e-4, 10., 256, 120),
        ("nbsvm_char125", True, 1.0, 1.25, .10, 1e-4, 20., 256, 120),
        ("nbsvm_style05", True, 1.0, 1.0, .05, 1e-4, 20., 256, 120),
    ]
    focused = [
        ("lr60", False, 1.0, 1.0, .10, 1e-4, 60., 256, 150),
        ("lr70", False, 1.0, 1.0, .10, 1e-4, 70., 256, 150),
        ("lr85", False, 1.0, 1.0, .10, 1e-4, 85., 256, 150),
        ("lr100", False, 1.0, 1.0, .10, 1e-4, 100., 256, 150),
        ("lr50_alpha03", False, 1.0, 1.0, .10, 3e-5, 50., 256, 150),
        ("lr50_alpha01", False, 1.0, 1.0, .10, 1e-5, 50., 256, 150),
        ("lr50_alpha3", False, 1.0, 1.0, .10, 3e-4, 50., 256, 150),
        ("lr50_batch128", False, 1.0, 1.0, .10, 1e-4, 50., 128, 150),
        ("lr50_batch512", False, 1.0, 1.0, .10, 1e-4, 50., 512, 150),
        ("lr50_epoch200", False, 1.0, 1.0, .10, 1e-4, 50., 256, 200),
        ("lr50_word125", False, 1.25, 1.0, .10, 1e-4, 50., 256, 150),
        ("lr50_char125", False, 1.0, 1.25, .10, 1e-4, 50., 256, 150),
        ("lr50_char150", False, 1.0, 1.50, .10, 1e-4, 50., 256, 150),
        ("lr50_both125", False, 1.25, 1.25, .10, 1e-4, 50., 256, 150),
        ("lr50_style05", False, 1.0, 1.0, .05, 1e-4, 50., 256, 150),
        ("lr50_style075", False, 1.0, 1.0, .075, 1e-4, 50., 256, 150),
        ("lr50_style15", False, 1.0, 1.0, .15, 1e-4, 50., 256, 150),
    ]
    return focused if os.environ.get("SEARCH_PHASE") == "focused" else broad


def main():
    train = pd.read_csv(PROJECT / "data/train.csv", usecols=["id", "text", "label"])
    split = pd.read_csv(PROJECT / "data/splits/shared_validation_split.csv")
    if train["id"].astype(str).tolist() != split["id"].astype(str).tolist():
        raise ValueError("Training data does not match the shared split.")
    train_mask = split["split"].eq("train").to_numpy()
    validation_mask = split["split"].eq("validation").to_numpy()
    train_text = train.loc[train_mask, "text"]
    validation_text = train.loc[validation_mask, "text"]
    y_train = train.loc[train_mask, "label"].to_numpy()
    y_validation = train.loc[validation_mask, "label"].to_numpy()

    vectorizer = build_vectorizer()
    train_tfidf = vectorizer.fit_transform(train_text)
    validation_tfidf = vectorizer.transform(validation_text)
    word_columns = len(vectorizer.word.vocabulary_)
    tfidf_columns = train_tfidf.shape[1]

    scaler = ScratchStandardScaler()
    train_styles = scaler.fit_transform(style_matrix(train_text))
    validation_styles = scaler.transform(style_matrix(validation_text))
    ratio = nb_log_count_ratio(train_tfidf, y_train)

    rows = []
    for config in trial_configs():
        (name, use_nb, word_weight, char_weight, style_weight,
         alpha, learning_rate, batch_size, epochs) = config
        scales = np.ones(tfidf_columns, dtype=np.float32)
        scales[:word_columns] *= word_weight
        scales[word_columns:] *= char_weight
        if use_nb:
            scales *= ratio
        x_train = combine(column_scale(train_tfidf, scales), train_styles, style_weight)
        x_validation = combine(
            column_scale(validation_tfidf, scales), validation_styles, style_weight
        )
        model = ScratchAveragedHingeSGD(
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_state=SEED,
            tolerance=1e-5,
        )
        start = time.perf_counter()
        model.fit(x_train, y_train)
        scores = model.decision_function(x_validation)
        default_f1 = macro_f1_score(y_validation, scores >= 0.0)
        threshold, tuned_f1 = tuned_macro_f1(y_validation, scores)
        row = {
            "trial": name,
            "nb_log_count_ratio": use_nb,
            "word_weight": word_weight,
            "character_weight": char_weight,
            "style_weight": style_weight,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "epochs_run": model.n_iter_,
            "default_validation_macro_f1": default_f1,
            "best_threshold": threshold,
            "tuned_validation_macro_f1": tuned_f1,
            "fit_seconds": time.perf_counter() - start,
        }
        rows.append(row)
        print(
            f"{name}: default={default_f1:.6f}, tuned={tuned_f1:.6f}, "
            f"threshold={threshold:.5f}"
        )

    results = pd.DataFrame(rows).sort_values(
        "tuned_validation_macro_f1", ascending=False
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase = os.environ.get("SEARCH_PHASE", "broad")
    results.to_csv(OUTPUT / f"scratch_hyperparameter_results_{phase}.csv", index=False)
    print("\n", results.head(10).to_string(index=False))
    print(f"\nBest Macro F1: {results.iloc[0]['tuned_validation_macro_f1']:.6f}")


if __name__ == "__main__":
    main()
