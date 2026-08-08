"""Search from-scratch TF-IDF feature spaces for AI-text detection."""

from pathlib import Path
import re
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
    ScratchTfidfVectorizer,
    combine,
    macro_f1_score,
    style_matrix,
)
from scratch_hyperparameter_search import tuned_macro_f1  # noqa: E402


OUTPUT = PROJECT / "results"


class RawCharacterTfidf(ScratchTfidfVectorizer):
    """Character n-grams across spaces/punctuation, implemented from scratch."""

    def _terms(self, text):
        text = "" if pd.isna(text) else re.sub(r"\s+", " ", str(text).lower())
        low, high = self.ngram_range
        for n in range(low, high + 1):
            for index in range(len(text) - n + 1):
                yield text[index:index + n]


def make_features(texts, word_range, char_range, raw_char, max_features):
    word = ScratchTfidfVectorizer(
        "word", word_range, min_df=2, max_df=.98, max_features=max_features
    )
    char_class = RawCharacterTfidf if raw_char else ScratchTfidfVectorizer
    character = char_class(
        "char", char_range, min_df=2, max_df=1.0, max_features=max_features
    )
    texts = list(texts)
    return word, character, ScratchSparseMatrix.hstack([
        word.fit_transform(texts), character.fit_transform(texts)
    ])


def transform(word, character, texts):
    texts = list(texts)
    return ScratchSparseMatrix.hstack([
        word.transform(texts), character.transform(texts)
    ])


def main():
    data = pd.read_csv(PROJECT / "data/train.csv", usecols=["id", "text", "label"])
    split = pd.read_csv(PROJECT / "data/splits/shared_validation_split.csv")
    train_mask = split["split"].eq("train").to_numpy()
    validation_mask = split["split"].eq("validation").to_numpy()
    train_text = data.loc[train_mask, "text"]
    validation_text = data.loc[validation_mask, "text"]
    y_train = data.loc[train_mask, "label"].to_numpy()
    y_validation = data.loc[validation_mask, "label"].to_numpy()

    scaler = ScratchStandardScaler()
    train_style = scaler.fit_transform(style_matrix(train_text))
    validation_style = scaler.transform(style_matrix(validation_text))

    variants = [
        ("word12_charwb35", (1, 2), (3, 5), False, 100_000),
        ("word13_charwb35", (1, 3), (3, 5), False, 140_000),
        ("word12_charwb26", (1, 2), (2, 6), False, 140_000),
        ("word13_charwb26", (1, 3), (2, 6), False, 160_000),
        ("word12_rawchar35", (1, 2), (3, 5), True, 140_000),
        ("word13_rawchar35", (1, 3), (3, 5), True, 160_000),
    ]
    rows = []
    for name, word_range, char_range, raw_char, maximum in variants:
        start = time.perf_counter()
        word, character, train_tfidf = make_features(
            train_text, word_range, char_range, raw_char, maximum
        )
        validation_tfidf = transform(word, character, validation_text)
        # Round 2 showed that emphasizing word TF-IDF by 1.25 was strongest.
        word_columns = len(word.vocabulary_)
        for row in range(train_tfidf.shape[0]):
            mask = train_tfidf.row_indices[row] < word_columns
            train_tfidf.row_values[row][mask] *= 1.25
        for row in range(validation_tfidf.shape[0]):
            mask = validation_tfidf.row_indices[row] < word_columns
            validation_tfidf.row_values[row][mask] *= 1.25

        x_train = combine(train_tfidf, train_style, .10)
        x_validation = combine(validation_tfidf, validation_style, .10)
        model = ScratchAveragedHingeSGD(
            alpha=1e-4, epochs=150, batch_size=256,
            learning_rate=50.0, random_state=42, tolerance=1e-5
        )
        model.fit(x_train, y_train)
        scores = model.decision_function(x_validation)
        default_f1 = macro_f1_score(y_validation, scores >= 0)
        threshold, tuned_f1 = tuned_macro_f1(y_validation, scores)
        row = {
            "feature_space": name,
            "word_ngram_range": str(word_range),
            "character_ngram_range": str(char_range),
            "raw_character_ngrams": raw_char,
            "word_features": len(word.vocabulary_),
            "character_features": len(character.vocabulary_),
            "default_validation_macro_f1": default_f1,
            "best_threshold": threshold,
            "tuned_validation_macro_f1": tuned_f1,
            "total_seconds": time.perf_counter() - start,
        }
        rows.append(row)
        print(f"{name}: default={default_f1:.6f}, tuned={tuned_f1:.6f}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        "tuned_validation_macro_f1", ascending=False
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "scratch_feature_space_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
