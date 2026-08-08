from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Constants
# =====================================================

LABEL_COLUMN = "label"
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
PCA_COMPONENTS = [2000, 1000, 500, 100]

# =====================================================
# Load Dataset
# =====================================================

train_features = pd.read_csv(DATA_DIR / "train_features.csv")
test_features = pd.read_csv(DATA_DIR / "test_features.csv")

X = train_features.drop(columns=["id", LABEL_COLUMN])
y = train_features[LABEL_COLUMN]

train_idx, val_idx = train_test_split(
    train_features.index,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

X_train = X.loc[train_idx]
X_val = X.loc[val_idx]

y_train = y.loc[train_idx]
y_val = y.loc[val_idx]

# =====================================================
# PCA Evaluation
# =====================================================

def evaluate_pca(n_components):
    """
    Train PCA + KNN and evaluate on validation set.

    Returns:
        pca model
        macro f1
        explained variance
        runtime
    """

    start = time.time()

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_SEED,
    )

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    knn = KNeighborsClassifier(
        n_neighbors=2
    )

    knn.fit(X_train_pca, y_train)

    predictions = knn.predict(X_val_pca)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    runtime = time.time() - start

    return (
        pca,
        macro_f1,
        explained_variance,
        runtime,
    )

# =====================================================
# Experiment Runner
# =====================================================

def run_experiments():

    results = []

    for n_components in PCA_COMPONENTS:

        print(f"\nRunning PCA ({n_components} components)")

        _, score, variance, runtime = evaluate_pca(
            n_components
        )

        results.append(
            {
                "Components": n_components,
                "MacroF1": score,
                "ExplainedVariance": variance,
                "Runtime": runtime,
            }
        )

        print(f"Macro F1 : {score:.4f}")
        print(f"Variance : {variance:.4f}")
        print(f"Runtime  : {runtime:.2f}s")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "pca_results.csv",
        index=False,
    )

    return results_df

# =====================================================
# Generate Kaggle Submissions
# =====================================================

def generate_submissions():

    X_all = train_features.drop(
        columns=["id", LABEL_COLUMN]
    )

    y_all = train_features[LABEL_COLUMN]

    X_test = test_features.drop(columns=["id"])

    for n_components in PCA_COMPONENTS:

        print(
            f"\nGenerating submission ({n_components} PCA)"
        )

        pca = PCA(
            n_components=n_components,
            random_state=RANDOM_SEED,
        )

        X_all_pca = pca.fit_transform(X_all)

        X_test_pca = pca.transform(X_test)

        knn = KNeighborsClassifier(
            n_neighbors=2
        )

        knn.fit(X_all_pca, y_all)

        predictions = knn.predict(X_test_pca)

        submission = pd.DataFrame(
            {
                "id": test_features["id"],
                "label": predictions,
            }
        )

        filename = (
            RESULTS_DIR
            / f"PCA{n_components}_KNN.csv"
        )

        submission.to_csv(
            filename,
            index=False,
        )

        print(f"Saved {filename.name}")

# =====================================================
# Plot Results
# =====================================================

def plot_results(results_df):

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["Components"],
        results_df["MacroF1"],
        marker="o",
    )

    plt.xlabel("Number of PCA Components")
    plt.ylabel("Validation Macro F1")
    plt.title("PCA Components vs Validation Macro F1")

    plt.grid(True)

    plt.savefig(
        RESULTS_DIR / "pca_macrof1.png",
        dpi=300,
    )

    plt.show()

# =====================================================
# Main
# =====================================================

def main():

    results_df = run_experiments()

    generate_submissions()

    plot_results(results_df)

    print("\nCompleted successfully.")

if __name__ == "__main__":
    main()