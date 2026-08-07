"""From-scratch text feature extraction and linear model for Task 3.

Nothing here uses a pre-built model or vectorizer from scikit-learn. NumPy and
SciPy supply array and sparse-matrix storage only; the TF-IDF weighting, the
feature scaling, the stylometry features and the SGD training rule are all
written out below.

The final notebook embeds this file directly, so this is the single place these
implementations live.
"""

from collections import Counter
import math
import re

import numpy as np
import pandas as pd
from scipy import sparse


FUNCTION_WORD_GROUPS = {
    "first_person": {"i", "me", "my", "mine", "we", "us", "our", "ours"},
    "second_person": {"you", "your", "yours"},
    "third_person": {"he", "she", "it", "they", "him", "her", "them", "their"},
    "articles": {"a", "an", "the"},
    "conjunctions": {"and", "but", "or", "although", "because", "while", "whereas"},
    "prepositions": {"of", "to", "in", "for", "with", "on", "at", "from", "by"},
    "demonstratives": {"this", "that", "these", "those"},
    "modals": {"can", "could", "may", "might", "must", "shall", "should", "will", "would"},
    "negations": {"no", "not", "never", "neither", "nor", "none", "without"},
    "transitions": {
        "however", "therefore", "moreover", "furthermore", "additionally",
        "consequently", "nevertheless", "overall", "thus", "hence",
    },
}


class ScratchStandardScaler:
    """Column-wise standardization implemented with NumPy."""

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, values):
        return self.fit(values).transform(values)


class ScratchTfidfVectorizer:
    """Small TF-IDF vectorizer supporting the settings used by this project."""

    def __init__(self, analyzer, ngram_range, min_df=2, max_df=1.0,
                 max_features=100_000):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

    @staticmethod
    def _words(text):
        return re.findall(r"(?u)\b\w\w+\b", str(text).lower())

    def _terms(self, text):
        text = "" if pd.isna(text) else str(text)
        low, high = self.ngram_range
        if self.analyzer == "word":
            tokens = self._words(text)
            for n in range(low, high + 1):
                for index in range(len(tokens) - n + 1):
                    yield " ".join(tokens[index:index + n])
            return

        # Like analyzer="char_wb": make character n-grams inside padded words.
        for word in re.findall(r"\S+", text.lower()):
            padded = f" {word} "
            for n in range(low, high + 1):
                for index in range(len(padded) - n + 1):
                    yield padded[index:index + n]

    def fit(self, texts):
        documents = ["" if pd.isna(text) else str(text) for text in texts]
        document_frequency = Counter()
        term_frequency = Counter()
        for text in documents:
            terms = list(self._terms(text))
            term_frequency.update(terms)
            document_frequency.update(set(terms))

        n_documents = len(documents)
        maximum_df = (
            int(self.max_df * n_documents) if isinstance(self.max_df, float)
            else int(self.max_df)
        )
        candidates = [
            term for term, count in document_frequency.items()
            if count >= self.min_df and count <= maximum_df
        ]
        candidates.sort(key=lambda term: (-term_frequency[term], term))
        candidates = candidates[:self.max_features]
        self.vocabulary_ = {term: index for index, term in enumerate(candidates)}
        self.idf_ = np.asarray([
            math.log((1.0 + n_documents) / (1.0 + document_frequency[term])) + 1.0
            for term in candidates
        ], dtype=np.float32)
        return self

    def transform(self, texts):
        texts = list(texts)
        rows, columns, values = [], [], []
        for row, text in enumerate(texts):
            counts = Counter(
                self.vocabulary_[term]
                for term in self._terms(text)
                if term in self.vocabulary_
            )
            if not counts:
                continue
            indices = np.fromiter(counts.keys(), dtype=np.int32)
            data = 1.0 + np.log(np.fromiter(counts.values(), dtype=np.float32))
            data *= self.idf_[indices]
            norm = float(np.linalg.norm(data))
            if norm:
                data /= norm
            rows.extend([row] * len(indices))
            columns.extend(indices.tolist())
            values.extend(data.tolist())
        return sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(texts), len(self.vocabulary_)),
            dtype=np.float32,
        )

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)


class ScratchHybridTfidf:
    """Union of independently normalized word and character TF-IDF."""

    def __init__(self):
        self.word = ScratchTfidfVectorizer("word", (1, 2), 2, 0.98, 100_000)
        self.character = ScratchTfidfVectorizer("char_wb", (3, 5), 2, 1.0, 100_000)

    def fit_transform(self, texts):
        texts = list(texts)
        return sparse.hstack(
            [self.word.fit_transform(texts), self.character.fit_transform(texts)],
            format="csr",
        )

    def transform(self, texts):
        texts = list(texts)
        return sparse.hstack(
            [self.word.transform(texts), self.character.transform(texts)],
            format="csr",
        )


class ScratchAveragedHingeSGD:
    """Mini-batch SGD for a balanced, L2-regularized linear SVM."""

    def __init__(self, alpha=1e-4, epochs=100, batch_size=256,
                 learning_rate=20.0, random_state=42, tolerance=1e-5):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.tolerance = tolerance

    def fit(self, features, labels):
        features = sparse.csr_matrix(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int8)
        signed = np.where(labels == 1, 1.0, -1.0).astype(np.float32)
        counts = np.bincount(labels, minlength=2)
        class_weights = len(labels) / (2.0 * np.maximum(counts, 1))
        sample_weights = class_weights[labels].astype(np.float32)
        weights = np.zeros(features.shape[1], dtype=np.float32)
        bias = 0.0
        averaged_weights = np.zeros_like(weights)
        averaged_bias = 0.0
        rng = np.random.default_rng(self.random_state)
        previous_loss = np.inf
        averaged_epochs = 0
        update = 0

        for _epoch in range(self.epochs):
            order = rng.permutation(len(labels))
            epoch_hinge = 0.0
            for start in range(0, len(labels), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = features[indices]
                targets = signed[indices]
                importance = sample_weights[indices]
                margins = targets * (batch.dot(weights) + bias)
                active = margins < 1.0
                rate = self.learning_rate / math.sqrt(1.0 + update)
                weights *= max(0.0, 1.0 - rate * self.alpha)
                if np.any(active):
                    active_x = batch[active]
                    coefficients = importance[active] * targets[active]
                    gradient = np.asarray(active_x.T.dot(coefficients)).ravel()
                    weights += (rate / len(indices)) * gradient
                    bias += rate * float(coefficients.sum() / len(indices))
                epoch_hinge += float(np.maximum(0.0, 1.0 - margins).sum())
                update += 1

            averaged_weights += weights
            averaged_bias += bias
            averaged_epochs += 1
            loss = epoch_hinge / len(labels)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        self.coef_ = averaged_weights / averaged_epochs
        self.intercept_ = averaged_bias / averaged_epochs
        self.n_iter_ = averaged_epochs
        return self

    def decision_function(self, features):
        return np.asarray(features.dot(self.coef_) + self.intercept_).ravel()

    def predict(self, features):
        return (self.decision_function(features) >= 0.0).astype(int)


def build_vectorizer():
    return ScratchHybridTfidf()


def make_model(loss="hinge", alpha=1e-4):
    if loss != "hinge":
        raise ValueError("The from-scratch model currently supports hinge loss only.")
    return ScratchAveragedHingeSGD(alpha=alpha)


def macro_f1_score(y_true, y_pred):
    """Binary unweighted Macro F1 implemented from the definition."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    class_scores = []
    for label in (0, 1):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(class_scores))


def safe_stats(values):
    if not values:
        return [0.0] * 7
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std())
    return [
        mean,
        std,
        float(array.min()),
        float(array.max()),
        float(np.median(array)),
        std / (mean + 1e-8),
        float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    ]


def approximate_syllables(word):
    groups = re.findall(r"[aeiouy]+", word.lower())
    count = len(groups)
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def repetition_ratio(items, n):
    grams = list(zip(*(items[index:] for index in range(n))))
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def advanced_style_features(text):
    text = "" if pd.isna(text) else str(text)
    lower = text.lower()
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", lower)
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    word_count = max(len(words), 1)
    char_count = max(len(text), 1)
    sentence_count = max(len(sentences), 1)
    counts = Counter(words)
    unique_count = len(counts)

    sentence_lengths = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences]
    word_lengths = [len(word) for word in words]
    paragraph_lengths = [len(re.findall(r"\b\w+\b", paragraph)) for paragraph in paragraphs]

    frequency_values = np.asarray(list(counts.values()), dtype=np.float64)
    probabilities = frequency_values / frequency_values.sum() if frequency_values.size else np.array([])
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else 0.0
    hapax = sum(value == 1 for value in counts.values()) / word_count
    dis_legomena = sum(value == 2 for value in counts.values()) / word_count
    repeated_words = (len(words) - unique_count) / word_count
    most_common_rate = max(counts.values(), default=0) / word_count

    sentence_openings = [
        tuple(re.findall(r"\b[a-zA-Z]+\b", sentence.lower())[:2])
        for sentence in sentences
    ]
    opening_repetition = (
        1.0 - len(set(sentence_openings)) / len(sentence_openings)
        if sentence_openings else 0.0
    )

    syllables = sum(approximate_syllables(word) for word in words)
    syllables_per_word = syllables / word_count
    words_per_sentence = len(words) / sentence_count
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    flesch_kincaid = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    complex_ratio = sum(approximate_syllables(word) >= 3 for word in words) / word_count
    gunning_fog = 0.4 * (words_per_sentence + 100 * complex_ratio)

    features = [
        len(text), len(words), len(sentences), len(paragraphs),
        unique_count / word_count,
        unique_count / math.sqrt(word_count),
        unique_count / math.sqrt(2 * word_count),
        hapax, dis_legomena, repeated_words, most_common_rate, entropy,
        repetition_ratio(words, 2), repetition_ratio(words, 3), opening_repetition,
    ]
    features.extend(safe_stats(sentence_lengths))
    features.extend([
        sum(length <= 8 for length in sentence_lengths) / sentence_count,
        sum(length >= 30 for length in sentence_lengths) / sentence_count,
    ])
    features.extend(safe_stats(word_lengths))
    features.extend([
        sum(length <= 3 for length in word_lengths) / word_count,
        sum(4 <= length <= 6 for length in word_lengths) / word_count,
        sum(7 <= length <= 9 for length in word_lengths) / word_count,
        sum(length >= 10 for length in word_lengths) / word_count,
    ])
    features.extend(safe_stats(paragraph_lengths))

    for group in FUNCTION_WORD_GROUPS.values():
        features.append(sum(counts[word] for word in group) / word_count)

    features.extend([
        sum(character.isupper() for character in text) / char_count,
        sum(character.isdigit() for character in text) / char_count,
        sum(character.isspace() for character in text) / char_count,
        *(text.count(mark) / char_count for mark in [",", ";", ":", "?", "!", "-", "(", ")", '"', "'"]),
        text.count("\n") / char_count,
        len(re.findall(r"(?m)^\s*[-*•]\s+", text)) / sentence_count,
        len(re.findall(r"[!?.,]{2,}", text)) / sentence_count,
        len(re.findall(r"\([A-Z][A-Za-z-]+,?\s+\d{4}[a-z]?\)", text)) / sentence_count,
        len(re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", text)) / sentence_count,
        len(re.findall(r"\b\d+(?:\.\d+)?%", text)) / word_count,
        len(re.findall(r"\b[A-Z]{2,}\b", text)) / word_count,
        len(re.findall(r"\([^)]{3,}\)", text)) / sentence_count,
        syllables_per_word, flesch, flesch_kincaid, complex_ratio, gunning_fog,
    ])
    return np.nan_to_num(np.asarray(features, dtype=np.float32))


def style_matrix(texts):
    return np.vstack([advanced_style_features(text) for text in texts])


def combine(tfidf, styles, weight):
    """Glue the stylometry block onto the TF-IDF block.

    `weight` sets how loud the 78 stylometry columns are next to the 200k
    TF-IDF columns; without it they would be drowned out.
    """
    return sparse.hstack(
        [tfidf, sparse.csr_matrix(styles * weight)], format="csr"
    )
