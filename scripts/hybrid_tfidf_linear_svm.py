"""Compare hybrid TF-IDF and hybrid TF-IDF plus stylometric features."""

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions"
SEED = 42
STYLE_WEIGHT = 0.05


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def build_vectorizer():
    return FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=100_000,
                dtype=np.float32,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=100_000,
                dtype=np.float32,
            ),
        ),
    ])


def document_style(text):
    text = "" if pd.isna(text) else str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    characters = max(len(text), 1)
    word_count = max(len(words), 1)
    unique_words = len(set(words))
    repeated = len(words) - unique_words
    return [
        len(text),
        len(words),
        len(sentences),
        np.mean([len(word) for word in words]) if words else 0.0,
        len(words) / max(len(sentences), 1),
        unique_words / word_count,
        repeated / word_count,
        sum(char.isupper() for char in text) / characters,
        sum(char.isdigit() for char in text) / characters,
        sum(char.isspace() for char in text) / characters,
        sum(char in ",.;:!?" for char in text) / characters,
        text.count("\n"),
        text.count("?"),
        text.count("!"),
        text.count("(") + text.count(")"),
        text.count('"') + text.count("'"),
    ]


def style_matrix(texts):
    return np.asarray([document_style(text) for text in texts], dtype=np.float32)


def tune_model(name, x_train, y_train, x_val, y_val):
    rows = []
    for c_value in (0.25, 0.5, 1.0):
        start = time.perf_counter()
        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            random_state=SEED,
            max_iter=10_000,
        )
        model.fit(x_train, y_train)
        score = macro_f1(y_val, model.predict(x_val))
        rows.append({
            "model": name,
            "C": c_value,
            "validation_macro_f1": score,
            "fit_seconds": time.perf_counter() - start,
        })
        print(f"{name}, C={c_value}: validation Macro F1={score:.6f}")
    table = pd.DataFrame(rows)
    best_c = float(table.loc[table["validation_macro_f1"].idxmax(), "C"])
    return table, best_c


def save_submission(filename, ids, predictions):
    submission = pd.DataFrame({"id": ids, "label": np.asarray(predictions, dtype=int)})
    assert submission.columns.tolist() == ["id", "label"]
    assert len(submission) == 6_999 and submission["id"].is_unique
    assert not submission.isna().any().any()
    assert set(submission["label"]).issubset({0, 1})
    path = SUBMISSIONS / filename
    submission.to_csv(path, index=False)
    print(f"Saved {len(submission)} predictions to {path}")


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "text", "label"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "text"])
    split = pd.read_csv(DATA / "splits" / "shared_validation_split.csv")
    assert len(train) == len(split) == 20_000
    assert train["id"].astype("string").tolist() == split["id"].astype("string").tolist()
    train_mask = split["split"].eq("train").to_numpy()
    val_mask = split["split"].eq("validation").to_numpy()
    train_text = train.loc[train_mask, "text"]
    val_text = train.loc[val_mask, "text"]
    y_train = train.loc[train_mask, "label"]
    y_val = train.loc[val_mask, "label"]

    vectorizer = build_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(train_text)
    x_val_tfidf = vectorizer.transform(val_text)
    hybrid_results, hybrid_c = tune_model(
        "Word + character TF-IDF", x_train_tfidf, y_train, x_val_tfidf, y_val
    )

    scaler = StandardScaler()
    train_style = scaler.fit_transform(style_matrix(train_text)) * STYLE_WEIGHT
    val_style = scaler.transform(style_matrix(val_text)) * STYLE_WEIGHT
    x_train_style = sparse.hstack(
        [x_train_tfidf, sparse.csr_matrix(train_style)], format="csr"
    )
    x_val_style = sparse.hstack(
        [x_val_tfidf, sparse.csr_matrix(val_style)], format="csr"
    )
    style_results, style_c = tune_model(
        "Hybrid TF-IDF + stylometry", x_train_style, y_train, x_val_style, y_val
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.concat([hybrid_results, style_results], ignore_index=True).to_csv(
        RESULTS / "feature_engineering_results.csv", index=False
    )

    # Refit feature transformations on all labelled documents.
    final_vectorizer = build_vectorizer()
    x_all_tfidf = final_vectorizer.fit_transform(train["text"])
    x_test_tfidf = final_vectorizer.transform(test["text"])

    hybrid_model = LinearSVC(
        C=hybrid_c, class_weight="balanced", random_state=SEED, max_iter=10_000
    )
    hybrid_model.fit(x_all_tfidf, train["label"])
    save_submission(
        "Hybrid_Word_Char_TFIDF_Prediction.csv",
        test["id"],
        hybrid_model.predict(x_test_tfidf),
    )

    final_scaler = StandardScaler()
    all_style = final_scaler.fit_transform(style_matrix(train["text"])) * STYLE_WEIGHT
    test_style = final_scaler.transform(style_matrix(test["text"])) * STYLE_WEIGHT
    x_all_style = sparse.hstack(
        [x_all_tfidf, sparse.csr_matrix(all_style)], format="csr"
    )
    x_test_style = sparse.hstack(
        [x_test_tfidf, sparse.csr_matrix(test_style)], format="csr"
    )
    style_model = LinearSVC(
        C=style_c, class_weight="balanced", random_state=SEED, max_iter=10_000
    )
    style_model.fit(x_all_style, train["label"])
    save_submission(
        "Hybrid_TFIDF_Stylometry_Prediction.csv",
        test["id"],
        style_model.predict(x_test_style),
    )


if __name__ == "__main__":
    main()
