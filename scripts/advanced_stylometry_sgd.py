"""From-scratch word/character TF-IDF, stylometry, and averaged SGD.

How the model works
-------------------
1. Word n-grams capture phrases, while character n-grams capture spelling,
   punctuation and formatting habits that may distinguish writing sources.
2. TF-IDF makes a term important when it is frequent in one document but
   uncommon across the training collection. Each document vector is L2
   normalized so long documents do not automatically receive larger scores.
3. Stylometric measurements describe writing *style* (sentence length,
   punctuation, function-word usage, vocabulary diversity, and similar cues).
4. A linear classifier learns one weight per feature. Positive weighted sums
   favour class 1 and negative sums favour class 0.
5. Mini-batch SGD minimizes balanced hinge loss with L2 regularization. Model
   weights from successive epochs are averaged to reduce noisy SGD variation.

The learning pipeline deliberately does not use scikit-learn or SciPy. NumPy
and pandas are used only for general numerical and tabular data handling.
"""

from collections import Counter
from pathlib import Path
import math
import re
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"

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


class ScratchSparseMatrix:
    """Row-oriented sparse matrix containing only NumPy index/value arrays.

    Text matrices contain hundreds of thousands of possible n-grams, but one
    document uses only a small fraction of them. Storing just the non-zero
    column indices and values avoids constructing an enormous dense matrix.
    """

    def __init__(self, row_indices, row_values, n_columns):
        if len(row_indices) != len(row_values):
            raise ValueError("Sparse row indices and values must have equal lengths.")
        self.row_indices = [np.asarray(row, dtype=np.int32) for row in row_indices]
        self.row_values = [np.asarray(row, dtype=np.float32) for row in row_values]
        self.shape = (len(self.row_indices), int(n_columns))

    @classmethod
    def hstack(cls, matrices):
        if not matrices:
            return cls([], [], 0)
        row_count = matrices[0].shape[0]
        if any(matrix.shape[0] != row_count for matrix in matrices):
            raise ValueError("All sparse matrices must have the same row count.")
        offsets = np.cumsum([0] + [matrix.shape[1] for matrix in matrices[:-1]])
        indices, values = [], []
        for row in range(row_count):
            row_indices = [
                matrix.row_indices[row] + offset
                for matrix, offset in zip(matrices, offsets)
                if matrix.row_indices[row].size
            ]
            row_values = [
                matrix.row_values[row]
                for matrix in matrices
                if matrix.row_values[row].size
            ]
            indices.append(
                np.concatenate(row_indices) if row_indices else np.empty(0, dtype=np.int32)
            )
            values.append(
                np.concatenate(row_values) if row_values else np.empty(0, dtype=np.float32)
            )
        return cls(indices, values, sum(matrix.shape[1] for matrix in matrices))

    def append_dense(self, dense):
        dense = np.asarray(dense, dtype=np.float32)
        if dense.ndim != 2 or dense.shape[0] != self.shape[0]:
            raise ValueError("Dense features must have the same number of rows.")
        new_indices, new_values = [], []
        for row, dense_row in enumerate(dense):
            nonzero = np.flatnonzero(dense_row).astype(np.int32)
            new_indices.append(
                np.concatenate([self.row_indices[row], nonzero + self.shape[1]])
            )
            new_values.append(
                np.concatenate([self.row_values[row], dense_row[nonzero]])
            )
        return ScratchSparseMatrix(
            new_indices, new_values, self.shape[1] + dense.shape[1]
        )

    def dot(self, weights, rows=None):
        weights = np.asarray(weights, dtype=np.float32)
        selected = range(self.shape[0]) if rows is None else np.asarray(rows, dtype=int)
        return np.asarray([
            np.dot(self.row_values[row], weights[self.row_indices[row]])
            for row in selected
        ], dtype=np.float32)

    def transpose_dot(self, rows, coefficients):
        gradient = np.zeros(self.shape[1], dtype=np.float32)
        for row, coefficient in zip(np.asarray(rows, dtype=int), coefficients):
            np.add.at(
                gradient,
                self.row_indices[row],
                self.row_values[row] * np.float32(coefficient),
            )
        return gradient


class ScratchStandardScaler:
    """Column-wise standardization implemented with NumPy.

    For each stylometric column this computes z = (x - mean) / standard
    deviation. This prevents a count with a large numerical range from
    dominating a rate or ratio simply because of its units.
    """

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
    """Small TF-IDF vectorizer supporting the settings used by this project.

    ``fit`` learns the vocabulary and smoothed inverse-document frequencies:
        idf(t) = log((1 + N) / (1 + df(t))) + 1
    ``transform`` uses sublinear term frequency ``1 + log(count)`` and then
    L2-normalizes every document. The vocabulary is learned on training text
    only, which prevents validation/test information leaking into training.
    """

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
        row_indices, row_values = [], []
        for text in texts:
            counts = Counter(
                self.vocabulary_[term]
                for term in self._terms(text)
                if term in self.vocabulary_
            )
            if not counts:
                row_indices.append(np.empty(0, dtype=np.int32))
                row_values.append(np.empty(0, dtype=np.float32))
                continue
            indices = np.fromiter(counts.keys(), dtype=np.int32)
            data = 1.0 + np.log(np.fromiter(counts.values(), dtype=np.float32))
            data *= self.idf_[indices]
            norm = float(np.linalg.norm(data))
            if norm:
                data /= norm
            row_indices.append(indices)
            row_values.append(data.astype(np.float32))
        return ScratchSparseMatrix(row_indices, row_values, len(self.vocabulary_))

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
        return ScratchSparseMatrix.hstack([
            self.word.fit_transform(texts), self.character.fit_transform(texts)
        ])

    def transform(self, texts):
        texts = list(texts)
        return ScratchSparseMatrix.hstack([
            self.word.transform(texts), self.character.transform(texts)
        ])


class ScratchAveragedHingeSGD:
    """Mini-batch SGD for a balanced, L2-regularized linear SVM.

    With labels converted to -1/+1, hinge loss is max(0, 1 - y*f(x)). A sample
    updates the separating hyperplane only when its margin y*f(x) is below 1.
    Inverse-frequency class weights give both classes equal total influence.
    L2 shrinkage discourages extreme coefficients, and the learning rate
    decays with the number of updates for increasingly stable steps.
    """

    def __init__(self, alpha=1e-4, epochs=100, batch_size=256,
                 learning_rate=20.0, random_state=42, tolerance=1e-5):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.tolerance = tolerance

    def fit(self, features, labels):
        if not isinstance(features, ScratchSparseMatrix):
            raise TypeError("features must be a ScratchSparseMatrix")
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
                targets = signed[indices]
                importance = sample_weights[indices]
                margins = targets * (features.dot(weights, indices) + bias)
                active = margins < 1.0
                rate = self.learning_rate / math.sqrt(1.0 + update)
                weights *= max(0.0, 1.0 - rate * self.alpha)
                if np.any(active):
                    coefficients = importance[active] * targets[active]
                    gradient = features.transpose_dot(indices[active], coefficients)
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
    return tfidf.append_dense(styles * weight)


def save_submission(filename, ids, predictions):
    frame = pd.DataFrame({"id": ids, "label": np.asarray(predictions, dtype=int)})
    assert frame.columns.tolist() == ["id", "label"]
    assert len(frame) == 6_999 and frame["id"].is_unique
    assert not frame.isna().any().any() and set(frame["label"]) <= {0, 1}
    path = SUBMISSIONS / filename
    frame.to_csv(path, index=False)
    print(f"Saved {path}; class-1 rate={frame['label'].mean():.4f}")


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
    y_val = train.loc[val_mask, "label"]

    vectorizer = build_vectorizer()
    train_tfidf = vectorizer.fit_transform(train_text)
    val_tfidf = vectorizer.transform(val_text)
    scaler = ScratchStandardScaler()
    train_styles = scaler.fit_transform(style_matrix(train_text))
    val_styles = scaler.transform(style_matrix(val_text))

    rows = []
    for weight in (0.02, 0.05, 0.10):
        x_train = combine(train_tfidf, train_styles, weight)
        x_val = combine(val_tfidf, val_styles, weight)
        start = time.perf_counter()
        model = make_model("hinge", 1e-4)
        model.fit(x_train, y_train)
        score = macro_f1_score(y_val, model.predict(x_val))
        rows.append({
            "loss": "hinge", "alpha": 1e-4, "style_weight": weight,
            "validation_macro_f1": float(score),
            "fit_seconds": time.perf_counter() - start,
        })
        print(f"style_weight={weight:.2f}: validation Macro F1={score:.6f}")

    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    results.to_csv(RESULTS / "advanced_stylometry_sgd_results.csv", index=False)
    best_weight = float(results.iloc[0]["style_weight"])

    final_vectorizer = build_vectorizer()
    all_tfidf = final_vectorizer.fit_transform(train["text"])
    test_tfidf = final_vectorizer.transform(test["text"])
    final_scaler = ScratchStandardScaler()
    all_styles = final_scaler.fit_transform(style_matrix(train["text"]))
    test_styles = final_scaler.transform(style_matrix(test["text"]))
    x_all = combine(all_tfidf, all_styles, best_weight)
    x_test = combine(test_tfidf, test_styles, best_weight)

    model = make_model("hinge", 1e-4)
    model.fit(x_all, train["label"])
    scores = model.decision_function(x_test)
    save_submission("Advanced_Stylometry_TFIDF_SGD_Prediction.csv", test["id"], scores >= 0)
    cutoff = np.quantile(scores, 0.45)
    save_submission("Advanced_Stylometry_TFIDF_SGD_Rank55_Prediction.csv", test["id"], scores >= cutoff)


if __name__ == "__main__":
    main()
