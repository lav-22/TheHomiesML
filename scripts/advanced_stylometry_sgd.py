"""Word/character TF-IDF plus advanced stylometry with averaged SGD."""

from collections import Counter
from pathlib import Path
import math
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from hybrid_tfidf_linear_svm import build_vectorizer
from hybrid_tfidf_sgd import make_model


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
    return sparse.hstack(
        [tfidf, sparse.csr_matrix(styles * weight)], format="csr"
    )


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
    scaler = StandardScaler()
    train_styles = scaler.fit_transform(style_matrix(train_text))
    val_styles = scaler.transform(style_matrix(val_text))

    rows = []
    for weight in (0.02, 0.05, 0.10):
        x_train = combine(train_tfidf, train_styles, weight)
        x_val = combine(val_tfidf, val_styles, weight)
        start = time.perf_counter()
        model = make_model("hinge", 1e-4)
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
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
    final_scaler = StandardScaler()
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
