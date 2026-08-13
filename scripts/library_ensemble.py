"""Validate and generate the library-based SVM + stylometry ensemble.

How the two components complement each other
---------------------------------------------
* LinearSVC sees a large sparse representation of word phrases, character
  patterns, and scaled style measurements. It is effective when many weak text
  clues combine approximately linearly.
* HistGradientBoosting sees only the compact stylometry matrix. Its shallow
  boosted trees can learn nonlinear rules such as one stylistic measurement
  becoming important only when another is also unusually high.
* Their raw scores have different units, so each score vector is converted to
  percentile ranks before a 50/50 blend. Ranking preserves ordering while
  making the two components comparable without probability calibration.

Existing ML libraries are intentionally used here because this is an ensemble
model. Standalone models elsewhere on this branch remain from-scratch only.
"""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from advanced_stylometry_sgd import style_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions" / "experiments"
MODELS = ROOT / "models" / "library_ensemble"
SEED = 42


def extra_style_features(text):
    """Measure interpretable document-style signals rather than topic words.

    The features include section headings, citations, mathematical markup,
    list structure, transitions, pronouns, selected formulaic words, Unicode
    frequency and line-layout statistics. Counts are divided by word count
    where appropriate so document length is not the main signal.
    """
    text = "" if pd.isna(text) else str(text)
    words = re.findall(r"\b[a-z]+\b", text.lower())
    word_count = max(len(words), 1)
    ai_words = {
        "delve", "tapestry", "multifaceted", "realm", "landscape",
        "underscore", "pivotal", "intricate", "comprehensive", "robust",
        "seamless", "foster", "leverage", "nuanced", "notably", "crucial",
        "moreover", "furthermore",
    }
    patterns = [
        r"(?im)^\s*(?:abstract|introduction|background|methods?|results?|discussion|conclusions?|references)\b",
        r"\\[a-zA-Z]+|\$[^$]+\$",
        r"\b(?:fig(?:ure)?|table|equation)\s*\d+",
        r"https?://|www\.|\bdoi\b",
        r"\bet al\.\b",
        r"\[[0-9,\-\s]+\]",
        r"(?m)^\s*\d+(?:\.\d+)*[.)]?\s+",
        r"(?m)^\s*[-*•]\s+",
        r"\b(?:however|therefore|moreover|furthermore|additionally|consequently)\b",
        r"\b(?:i|me|my|we|us|our)\b",
    ]
    lines = text.splitlines()
    values = [len(re.findall(pattern, text)) / word_count for pattern in patterns]
    values.extend([
        sum(word in ai_words for word in words) / word_count,
        len(set(words) & ai_words) / len(ai_words),
        sum(ord(character) > 127 for character in text) / max(len(text), 1),
        sum(not line.strip() for line in lines) / max(len(lines), 1),
        float(np.std([len(line) for line in lines])) if lines else 0.0,
    ])
    return values


def enhanced_style_matrix(texts):
    return np.column_stack([
        style_matrix(texts),
        np.asarray([extra_style_features(text) for text in texts], dtype=np.float32),
    ])


def make_vectorizers():
    return (
        TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.98, sublinear_tf=True,
            max_features=160_000, dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=160_000, dtype=np.float32,
        ),
    )


def make_models():
    return (
        LinearSVC(
            C=1.0, class_weight="balanced", random_state=SEED,
            max_iter=20_000,
        ),
        HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=SEED,
        ),
    )


def rank_scores(scores):
    """Map arbitrary model scores to the interval [0, 1] by percentile rank."""
    order = np.argsort(np.argsort(scores))
    return order.astype(np.float64) / max(len(scores) - 1, 1)


def ensemble_scores(svm_scores, booster_scores):
    return .50 * rank_scores(svm_scores) + .50 * rank_scores(booster_scores)


def exact_best_threshold(labels, scores):
    """Evaluate every distinct score boundary and return the best Macro-F1 cut.

    Sorting once allows cumulative confusion-matrix counts to be updated for
    every possible split. Threshold selection is performed only on validation
    labels; test labels are never inspected.
    """
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels == 1)
    false_positive = np.cumsum(sorted_labels == 0)
    total_positive = int(np.sum(labels == 1))
    total_negative = len(labels) - total_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive
    f1_positive = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    f1_negative = 2 * true_negative / np.maximum(
        2 * true_negative + false_negative + false_positive, 1
    )
    macro = (f1_positive + f1_negative) / 2
    sorted_scores = scores[order]
    valid = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    valid_indices = np.flatnonzero(valid)
    best_index = valid_indices[np.argmax(macro[valid_indices])]
    if best_index == len(scores) - 1:
        threshold = np.nextafter(sorted_scores[best_index], -np.inf)
    else:
        threshold = (sorted_scores[best_index] + sorted_scores[best_index + 1]) / 2
    return float(threshold), float(macro[best_index])


def build_hybrid(word_matrix, char_matrix, styles):
    """Join complementary features after applying validation-selected weights.

    Word features receive extra emphasis (1.5), character features keep unit
    weight, and standardized style features receive 0.10 because a compact
    dense block would otherwise compete too strongly with sparse TF-IDF.
    """
    return sparse.hstack([
        word_matrix * 1.5,
        char_matrix,
        sparse.csr_matrix(styles * .10),
    ], format="csr")


def save_submission(name, ids, labels):
    frame = pd.DataFrame({"id": ids, "label": np.asarray(labels, dtype=int)})
    assert len(frame) == 6_999 and frame["id"].nunique() == 6_999
    assert not frame.isna().any().any() and set(frame["label"]) <= {0, 1}
    frame.to_csv(SUBMISSIONS / name, index=False)
    print(f"Saved {name}; class-1 rate={frame['label'].mean():.4f}")


def validate(train, split):
    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()
    train_text = train.loc[train_mask, "text"]
    val_text = train.loc[val_mask, "text"]
    y_train = train.loc[train_mask, "label"].to_numpy()
    y_val = train.loc[val_mask, "label"].to_numpy()

    word, character = make_vectorizers()
    train_word = word.fit_transform(train_text)
    val_word = word.transform(val_text)
    train_char = character.fit_transform(train_text)
    val_char = character.transform(val_text)
    scaler = StandardScaler()
    train_style = scaler.fit_transform(enhanced_style_matrix(train_text))
    val_style = scaler.transform(enhanced_style_matrix(val_text))
    x_train = build_hybrid(train_word, train_char, train_style)
    x_val = build_hybrid(val_word, val_char, val_style)

    svm, booster = make_models()
    svm.fit(x_train, y_train)
    booster.fit(train_style, y_train)
    svm_scores = svm.decision_function(x_val)
    booster_scores = booster.predict_proba(val_style)[:, 1]
    combined = ensemble_scores(svm_scores, booster_scores)
    threshold, ensemble_f1 = exact_best_threshold(y_val, combined)

    rows = [
        {"model": "LinearSVC_C1", "type": "individual",
         "validation_macro_f1": f1_score(
             y_val, svm_scores >= 0, average="macro", zero_division=0)},
        {"model": "HistGradientBoosting_stylometry", "type": "individual",
         "validation_macro_f1": f1_score(
             y_val, booster_scores >= .5, average="macro", zero_division=0)},
        {"model": "SVM + stylometry gradient boosting", "type": "ensemble",
         "validation_macro_f1": ensemble_f1},
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        "validation_macro_f1", ascending=False
    ).to_csv(RESULTS / "library_ensemble_results.csv", index=False)
    pd.DataFrame({
        "id": train.loc[val_mask, "id"], "label": y_val,
        "svm_score": svm_scores, "style_booster_score": booster_scores,
        "ensemble_rank_score": combined,
    }).to_csv(RESULTS / "library_ensemble_validation_scores.csv", index=False)
    print(f"Ensemble validation Macro F1={ensemble_f1:.6f}")
    return threshold


def fit_all_and_predict(train, test, validation_threshold):
    word, character = make_vectorizers()
    train_word = word.fit_transform(train["text"])
    test_word = word.transform(test["text"])
    train_char = character.fit_transform(train["text"])
    test_char = character.transform(test["text"])
    scaler = StandardScaler()
    train_style = scaler.fit_transform(enhanced_style_matrix(train["text"]))
    test_style = scaler.transform(enhanced_style_matrix(test["text"]))
    x_train = build_hybrid(train_word, train_char, train_style)
    x_test = build_hybrid(test_word, test_char, test_style)

    svm, booster = make_models()
    svm.fit(x_train, train["label"])
    booster.fit(train_style, train["label"])
    svm_scores = svm.decision_function(x_test)
    booster_scores = booster.predict_proba(test_style)[:, 1]
    combined = ensemble_scores(svm_scores, booster_scores)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    save_submission(
        "Library_Ensemble_ValidationThreshold_Prediction.csv",
        test["id"], combined >= validation_threshold,
    )
    save_submission(
        "Library_Ensemble_Rank55_Prediction.csv",
        test["id"], combined >= np.quantile(combined, .45),
    )
    pd.DataFrame({
        "id": test["id"], "svm_score": svm_scores,
        "style_booster_score": booster_scores,
        "ensemble_rank_score": combined,
    }).to_csv(SUBMISSIONS / "Library_Ensemble_DecisionScores.csv", index=False)

    for fraction in (.525, .575, .60):
        save_submission(
            f"Library_Ensemble_GlobalRank{fraction * 100:g}_Prediction.csv",
            test["id"], combined >= np.quantile(combined, 1 - fraction),
        )

    numeric = test["id"].astype(str).str.fullmatch(r"\d+").to_numpy()
    for numeric_fraction, uuid_fraction in (
        (.55, .55), (.50, .625), (.55, .625), (.60, .625),
    ):
        labels = np.zeros(len(test), dtype=int)
        labels[numeric] = (
            combined[numeric] >= np.quantile(combined[numeric], 1 - numeric_fraction)
        )
        labels[~numeric] = (
            combined[~numeric] >= np.quantile(combined[~numeric], 1 - uuid_fraction)
        )
        save_submission(
            f"Library_Ensemble_GroupRank_N{numeric_fraction*100:g}_U{uuid_fraction*100:g}_Prediction.csv",
            test["id"], labels,
        )

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(word, MODELS / "word_tfidf.joblib", compress=3)
    joblib.dump(character, MODELS / "character_tfidf.joblib", compress=3)
    joblib.dump(scaler, MODELS / "style_scaler.joblib", compress=3)
    joblib.dump(svm, MODELS / "linear_svm.joblib", compress=3)
    joblib.dump(booster, MODELS / "style_hist_gradient_booster.joblib", compress=3)
    joblib.dump({
        "models": ["LinearSVC_C1", "HistGradientBoosting_stylometry"],
        "weights": [.50, .50],
        "validation_threshold": validation_threshold,
    }, MODELS / "ensemble_metadata.joblib", compress=3)


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    split = pd.read_csv(DATA / "splits/shared_validation_split.csv")
    assert train["id"].astype(str).tolist() == split["id"].astype(str).tolist()
    threshold = validate(train, split)
    fit_all_and_predict(train, test, threshold)


if __name__ == "__main__":
    main()
