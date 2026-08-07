"""Task 3: the final model, selected on the shared validation split.

Everything that learns in this file is written from scratch and imported from
src/text_features.py: the TF-IDF vectorizer, the feature scaler, the averaged
hinge-loss SGD, and the Macro F1 metric. NumPy and SciPy provide array and
sparse-matrix storage only.

Two choices are made on validation and nothing is read from the test labels:

1. style_weight, which scales the stylometry block against the TF-IDF block,
2. the decision threshold, chosen on a small grid because Macro F1 on a 62.5%
   / 37.5% split is not maximised at the default cut-off of 0.

Averaging a few seeds is a plain variance-reduction step: the only thing that
changes between members is the order SGD visits the rows in.

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

from src.submission import create_submission
from src.text_features import (
    ScratchAveragedHingeSGD,
    ScratchHybridTfidf,
    ScratchStandardScaler,
    combine,
    macro_f1_score,
    style_matrix,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"

STYLE_WEIGHTS = (0.05, 0.10, 0.15)
SEEDS = (42, 43, 44)
ALPHA = 1e-4


def fit_seed_ensemble(x_train, y_train, seeds=SEEDS):
    """Fit one model per seed and return them all."""
    models = []
    for seed in seeds:
        model = ScratchAveragedHingeSGD(alpha=ALPHA, random_state=seed)
        model.fit(x_train, y_train)
        models.append(model)
    return models


def ensemble_decision(models, x):
    """Average the decision functions of the seed ensemble."""
    return np.mean([model.decision_function(x) for model in models], axis=0)


def tune_threshold(y_true, scores):
    """Pick the cut-off with the best Macro F1 on a deliberately coarse grid."""
    thresholds = np.linspace(-0.25, 0.25, 41)
    scored = [(float(t), macro_f1_score(y_true, (scores >= t).astype(int)))
              for t in thresholds]
    return max(scored, key=lambda item: item[1])


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
    y_val = train.loc[val_mask, "label"].to_numpy()

    # The expensive part, so it is done once and reused for every setting
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
        start = time.perf_counter()

        models = fit_seed_ensemble(x_train, y_train)

        # single seed, default cut-off: the plain reference point
        single = models[0].decision_function(x_val)
        single_f1 = macro_f1_score(y_val, (single >= 0.0).astype(int))

        # seed-averaged, default cut-off
        averaged = ensemble_decision(models, x_val)
        averaged_f1 = macro_f1_score(y_val, (averaged >= 0.0).astype(int))

        # seed-averaged, tuned cut-off
        threshold, tuned_f1 = tune_threshold(y_val, averaged)

        rows.append({
            "style_weight": weight,
            "single_seed_macro_f1": single_f1,
            "seed_averaged_macro_f1": averaged_f1,
            "best_threshold": threshold,
            "tuned_macro_f1": tuned_f1,
            "fit_seconds": round(time.perf_counter() - start, 1),
        })
        print(f"style_weight={weight:.2f}: single={single_f1:.6f} "
              f"averaged={averaged_f1:.6f} tuned={tuned_f1:.6f} (t={threshold:.4f})")

    results = (pd.DataFrame(rows)
               .sort_values("tuned_macro_f1", ascending=False)
               .reset_index(drop=True))
    results.to_csv(RESULTS / "task3_final_model_results.csv", index=False)
    print("\n", results.to_string(index=False), sep="")

    best = results.iloc[0]
    best_weight = float(best["style_weight"])
    best_threshold = float(best["best_threshold"])
    print(f"\nSelected style_weight={best_weight}, threshold={best_threshold:.4f}, "
          f"validation Macro F1={float(best['tuned_macro_f1']):.6f}")

    # Refit everything on all 20000 labelled rows using the chosen settings
    print("\nRefitting on the full training set...")
    final_vectorizer = ScratchHybridTfidf()
    all_tfidf = final_vectorizer.fit_transform(train["text"])
    test_tfidf = final_vectorizer.transform(test["text"])
    final_scaler = ScratchStandardScaler()
    all_styles = final_scaler.fit_transform(style_matrix(train["text"]))
    test_styles = final_scaler.transform(style_matrix(test["text"]))

    x_all = combine(all_tfidf, all_styles, best_weight)
    x_test = combine(test_tfidf, test_styles, best_weight)

    final_models = fit_seed_ensemble(x_all, train["label"])
    test_scores = ensemble_decision(final_models, x_test)

    test_preds = (test_scores >= best_threshold).astype(int)
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
