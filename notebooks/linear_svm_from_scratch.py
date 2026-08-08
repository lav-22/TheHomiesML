"""Linear SVM text classifier implemented without scikit-learn.

Allowed general-purpose libraries are used only for arrays, sparse storage, and CSV I/O.
TF-IDF, hinge-loss SVM training, prediction, and macro F1 are implemented here.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


class ScratchTfidfVectorizer:
    def __init__(self, min_df=2, max_features=100_000, ngram_range=(1, 2)):
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _terms(self, document):
        tokens = TOKEN_RE.findall(str(document).lower())
        terms = []
        if self.ngram_range[0] <= 1 <= self.ngram_range[1]:
            terms.extend(tokens)
        if self.ngram_range[0] <= 2 <= self.ngram_range[1]:
            terms.extend(a + " " + b for a, b in zip(tokens, tokens[1:]))
        return terms

    def fit(self, documents):
        document_frequency = Counter()
        n_documents = 0
        for document in documents:
            document_frequency.update(set(self._terms(document)))
            n_documents += 1
        eligible = ((term, df) for term, df in document_frequency.items() if df >= self.min_df)
        selected = sorted(eligible, key=lambda item: (-item[1], item[0]))[: self.max_features]
        self.vocabulary_ = {term: index for index, (term, _) in enumerate(selected)}
        dfs = np.asarray([df for _, df in selected], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_documents) / (1.0 + dfs)) + 1.0
        return self

    def transform(self, documents):
        if self.idf_ is None:
            raise RuntimeError("Vectorizer must be fitted before transform().")
        rows, columns, values = [], [], []
        for row, document in enumerate(documents):
            counts = Counter(self._terms(document))
            for term, count in counts.items():
                column = self.vocabulary_.get(term)
                if column is not None:
                    rows.append(row)
                    columns.append(column)
                    values.append(1.0 + np.log(count))
        matrix = sparse.csr_matrix(
            (np.asarray(values) * self.idf_[columns], (rows, columns)),
            shape=(row + 1 if "row" in locals() else 0, len(self.vocabulary_)),
            dtype=np.float64,
        )
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        norms[norms == 0.0] = 1.0
        return sparse.diags(1.0 / norms).dot(matrix).tocsr()

    def fit_transform(self, documents):
        documents = list(documents)
        return self.fit(documents).transform(documents)


class ScratchLinearSVM:
    """Mini-batch SGD on lambda/2||w||^2 + mean(max(0,1-yf(x)))."""

    def __init__(self, C=1.0, epochs=25, batch_size=256, learning_rate=0.5,
                 decay=0.02, class_weight=None, random_state=42):
        self.C, self.epochs, self.batch_size = C, epochs, batch_size
        self.learning_rate, self.decay = learning_rate, decay
        self.class_weight, self.random_state = class_weight, random_state
        self.weights_: np.ndarray | None = None
        self.bias_ = 0.0

    def fit(self, X, y, verbose=False):
        y = np.asarray(y, dtype=np.int8)
        signed_y = np.where(y == 1, 1.0, -1.0)
        sample_weight = np.ones(len(y))
        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=2)
            sample_weight = np.asarray([len(y) / (2 * counts[label]) for label in y])
        self.weights_ = np.zeros(X.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        regularization = 1.0 / (self.C * len(y))
        rng = np.random.default_rng(self.random_state)
        step = 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                indices = order[start:start + self.batch_size]
                X_batch, y_batch = X[indices], signed_y[indices]
                weights = sample_weight[indices]
                margins = y_batch * (X_batch @ self.weights_ + self.bias_)
                violating = margins < 1.0
                eta = self.learning_rate / (1.0 + self.decay * step)
                # Derivatives are lambda*w (L2) and -y*x (violating hinge terms).
                gradient_w = regularization * self.weights_
                gradient_b = 0.0
                if np.any(violating):
                    coefficients = weights[violating] * y_batch[violating]
                    gradient_w -= np.asarray(
                        X_batch[violating].T @ coefficients
                    ).ravel() / len(indices)
                    gradient_b = -coefficients.sum() / len(indices)
                self.weights_ -= eta * gradient_w
                self.bias_ -= eta * gradient_b
                step += 1
            if verbose:
                print(f"epoch={epoch + 1:02d} hinge_loss={self.hinge_loss(X, y):.6f}")
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.weights_ + self.bias_).ravel()

    def predict(self, X):
        return (self.decision_function(X) >= 0.0).astype(np.int8)

    def hinge_loss(self, X, y):
        signed_y = np.where(np.asarray(y) == 1, 1.0, -1.0)
        return np.maximum(0.0, 1.0 - signed_y * self.decision_function(X)).mean()


def macro_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for label in (0, 1):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def load_split(train, split_path):
    split = pd.read_csv(split_path)
    if len(split) != len(train) or not np.array_equal(split["row_index"], np.arange(len(train))):
        raise ValueError("Shared split does not align with train.csv rows.")
    return split["split"].eq("train").to_numpy(), split["split"].eq("validation").to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    args = parser.parse_args()
    train = pd.read_csv(args.project_root / "data" / "train.csv")
    test = pd.read_csv(args.project_root / "data" / "test.csv")
    train_mask, validation_mask = load_split(
        train, args.project_root / "data" / "splits" / "shared_validation_split.csv"
    )
    vectorizer = ScratchTfidfVectorizer()
    X_train = vectorizer.fit_transform(train.loc[train_mask, "text"].fillna(""))
    X_validation = vectorizer.transform(train.loc[validation_mask, "text"].fillna(""))
    selected_class_weight = None if args.class_weight == "none" else "balanced"
    model = ScratchLinearSVM(C=args.c, epochs=args.epochs,
                             learning_rate=args.learning_rate,
                             class_weight=selected_class_weight)
    model.fit(X_train, train.loc[train_mask, "label"], verbose=True)
    validation_predictions = model.predict(X_validation)
    print(f"validation_macro_f1={macro_f1(train.loc[validation_mask, 'label'], validation_predictions):.6f}")
    print(f"validation_prediction_counts={dict(zip(*np.unique(validation_predictions, return_counts=True)))}")

    # Refit both vocabulary and model using every labeled row before test prediction.
    vectorizer = ScratchTfidfVectorizer()
    X_full = vectorizer.fit_transform(train["text"].fillna(""))
    X_test = vectorizer.transform(test["text"].fillna(""))
    model = ScratchLinearSVM(C=args.c, epochs=args.epochs,
                             learning_rate=args.learning_rate,
                             class_weight=selected_class_weight)
    model.fit(X_full, train["label"], verbose=True)
    predictions = model.predict(X_test)
    submission = pd.DataFrame({"id": test["id"], "label": predictions})
    if len(submission) != len(test) or submission["label"].isna().any():
        raise RuntimeError("Submission validation failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"saved={args.output}")
    print(f"test_prediction_counts={dict(zip(*np.unique(predictions, return_counts=True)))}")


if __name__ == "__main__":
    main()
