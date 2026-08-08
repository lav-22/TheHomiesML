"""Final from-scratch feature, loss, and score-ensemble search.

This experiment asks three focused questions: how strongly should word and
stylometry features be weighted, which regularized linear loss works better,
and whether two independently trained score rankings improve one model. Every
configuration uses the same fixed train/validation rows, so the Macro-F1
comparison reflects model settings rather than a different random split.
"""

from pathlib import Path
import math
import re
import sys
import time

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from advanced_stylometry_sgd import (  # noqa: E402
    ScratchAveragedHingeSGD,
    ScratchStandardScaler,
    combine,
    macro_f1_score,
    style_matrix,
)
from scratch_feature_space_search import make_features, transform  # noqa: E402
from scratch_hyperparameter_search import tuned_macro_f1  # noqa: E402


OUTPUT = PROJECT / "results"


AI_MARKERS = {
    "delve", "tapestry", "multifaceted", "realm", "landscape", "underscore",
    "pivotal", "intricate", "comprehensive", "robust", "seamless", "foster",
    "leverage", "nuanced", "notably", "crucial", "moreover", "furthermore",
}


def extra_style(text):
    """Extract normalized structural and stylistic signals from one document.

    These signals target *how* text is written—formatting, discourse markers,
    pronoun usage, citations, passive constructions and line structure—rather
    than its subject. They complement TF-IDF when two documents discuss similar
    topics but have different authorship or generation styles.
    """
    text = "" if pd.isna(text) else str(text)
    lower = text.lower()
    words = re.findall(r"\b[a-z]+\b", lower)
    count = max(len(words), 1)
    lines = text.splitlines()
    patterns = [
        r"(?im)^\s*(abstract|introduction|background|methods?|results?|discussion|conclusions?|references)\b",
        r"\\[a-zA-Z]+|\$[^$]+\$", r"\b(fig(?:ure)?|table|equation)\s*\d+",
        r"https?://|www\.|\bdoi\b", r"\bet al\.\b", r"\[[0-9,\-\s]+\]",
        r"(?m)^\s*\d+(?:\.\d+)*[.)]?\s+", r"(?m)^\s*[-*•]\s+",
        r"\b(?:however|therefore|moreover|furthermore|additionally|consequently)\b",
        r"\b(?:I|me|my|we|us|our)\b", r"\b(?:is|are|was|were|be|been)\s+\w+ed\b",
    ]
    values = [len(re.findall(pattern, text, flags=0)) / count for pattern in patterns]
    values.extend([
        sum(word in AI_MARKERS for word in words) / count,
        len(set(words) & AI_MARKERS) / len(AI_MARKERS),
        sum(ord(character) > 127 for character in text) / max(len(text), 1),
        sum(not line.strip() for line in lines) / max(len(lines), 1),
        np.std([len(line) for line in lines]) if lines else 0.0,
    ])
    return values


class ScratchAveragedLogisticSGD(ScratchAveragedHingeSGD):
    """Balanced L2 logistic regression trained with project-owned SGD.

    Logistic loss produces a smooth gradient based on sigmoid probabilities,
    unlike hinge loss, which updates only margin-violating samples. Comparing
    the two under the same feature space tests whether a probabilistic loss or
    a maximum-margin boundary is more suitable for this dataset.
    """

    def fit(self, features, labels):
        labels = np.asarray(labels, dtype=np.int8)
        counts = np.bincount(labels, minlength=2)
        sample_weight = (len(labels) / (2.0 * np.maximum(counts, 1)))[labels]
        weights = np.zeros(features.shape[1], dtype=np.float32)
        bias = 0.0
        average_w = np.zeros_like(weights)
        average_b = 0.0
        rng = np.random.default_rng(self.random_state)
        previous = np.inf
        averaged = 0
        update = 0
        for _epoch in range(self.epochs):
            order = rng.permutation(len(labels))
            total_loss = 0.0
            for start in range(0, len(labels), self.batch_size):
                indices = order[start:start + self.batch_size]
                raw = np.clip(features.dot(weights, indices) + bias, -30, 30)
                probability = 1.0 / (1.0 + np.exp(-raw))
                error = sample_weight[indices] * (labels[indices] - probability)
                rate = self.learning_rate / math.sqrt(1.0 + update)
                weights *= max(0.0, 1.0 - rate * self.alpha)
                weights += rate * features.transpose_dot(indices, error) / len(indices)
                bias += rate * float(error.mean())
                total_loss += float(np.logaddexp(0, raw) .sum() - np.dot(labels[indices], raw))
                update += 1
            average_w += weights
            average_b += bias
            averaged += 1
            loss = total_loss / len(labels)
            if abs(previous - loss) < self.tolerance:
                break
            previous = loss
        self.coef_ = average_w / averaged
        self.intercept_ = average_b / averaged
        self.n_iter_ = averaged
        return self


def main():
    data = pd.read_csv(PROJECT / "data/train.csv", usecols=["id", "text", "label"])
    split = pd.read_csv(PROJECT / "data/splits/shared_validation_split.csv")
    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()
    train_text, val_text = data.loc[train_mask, "text"], data.loc[val_mask, "text"]
    y_train = data.loc[train_mask, "label"].to_numpy()
    y_val = data.loc[val_mask, "label"].to_numpy()

    word, character, train_tfidf = make_features(
        train_text, (1, 3), (3, 5), True, 160_000
    )
    val_tfidf = transform(word, character, val_text)
    word_columns = len(word.vocabulary_)

    raw_train_style = np.column_stack([
        style_matrix(train_text), np.asarray([extra_style(text) for text in train_text])
    ])
    raw_val_style = np.column_stack([
        style_matrix(val_text), np.asarray([extra_style(text) for text in val_text])
    ])
    scaler = ScratchStandardScaler()
    train_style = scaler.fit_transform(raw_train_style)
    val_style = scaler.transform(raw_val_style)

    # This is a deliberately small, interpretable search. Each row changes one
    # meaningful choice: feature weighting, step size, regularization, or loss.
    configs = [
        ("hinge_w100_lr50", "hinge", 1.00, .10, 50., 1e-4, 150),
        ("hinge_w125_lr50", "hinge", 1.25, .10, 50., 1e-4, 150),
        ("hinge_w150_lr50", "hinge", 1.50, .10, 50., 1e-4, 150),
        ("hinge_w125_lr75", "hinge", 1.25, .10, 75., 1e-4, 150),
        ("hinge_w125_lr100", "hinge", 1.25, .10, 100., 1e-4, 150),
        ("hinge_style15", "hinge", 1.25, .15, 50., 1e-4, 150),
        ("hinge_alpha01", "hinge", 1.25, .10, 50., 1e-5, 150),
        ("logistic_lr2", "logistic", 1.25, .10, 2., 1e-4, 150),
        ("logistic_lr5", "logistic", 1.25, .10, 5., 1e-4, 150),
        ("logistic_lr10", "logistic", 1.25, .10, 10., 1e-4, 150),
    ]
    original_train = [values.copy() for values in train_tfidf.row_values]
    original_val = [values.copy() for values in val_tfidf.row_values]
    rows, score_map = [], {}
    for name, loss, word_weight, style_weight, lr, alpha, epochs in configs:
        for matrix, originals in ((train_tfidf, original_train), (val_tfidf, original_val)):
            for index, values in enumerate(originals):
                matrix.row_values[index] = values.copy()
                mask = matrix.row_indices[index] < word_columns
                matrix.row_values[index][mask] *= word_weight
        x_train = combine(train_tfidf, train_style, style_weight)
        x_val = combine(val_tfidf, val_style, style_weight)
        cls = ScratchAveragedHingeSGD if loss == "hinge" else ScratchAveragedLogisticSGD
        model = cls(alpha=alpha, epochs=epochs, batch_size=256,
                    learning_rate=lr, random_state=42, tolerance=1e-5)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        scores = model.decision_function(x_val)
        threshold, tuned = tuned_macro_f1(y_val, scores)
        score_map[name] = scores
        rows.append({"trial": name, "loss": loss, "word_weight": word_weight,
                     "style_weight": style_weight, "learning_rate": lr,
                     "alpha": alpha, "best_threshold": threshold,
                     "validation_macro_f1": tuned,
                     "fit_seconds": time.perf_counter() - start})
        print(f"{name}: {tuned:.6f}", flush=True)

    # Blend standardized score rankings; rank normalization makes different
    # loss scales comparable without fitting an additional model.
    names = list(score_map)
    normalized = {}
    for name, scores in score_map.items():
        order = np.argsort(np.argsort(scores))
        normalized[name] = order / max(len(order) - 1, 1)
    top = sorted(rows, key=lambda row: row["validation_macro_f1"], reverse=True)[:5]
    top_names = [row["trial"] for row in top]
    for first_index in range(len(top_names)):
        for second_index in range(first_index + 1, len(top_names)):
            pair = (top_names[first_index], top_names[second_index])
            blend = (normalized[pair[0]] + normalized[pair[1]]) / 2
            threshold, tuned = tuned_macro_f1(y_val, blend)
            rows.append({"trial": f"blend:{pair[0]}+{pair[1]}", "loss": "rank_blend",
                         "word_weight": np.nan, "style_weight": np.nan,
                         "learning_rate": np.nan, "alpha": np.nan,
                         "best_threshold": threshold,
                         "validation_macro_f1": tuned, "fit_seconds": 0.0})

    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "scratch_final_ensemble_results.csv", index=False)
    print(results.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
