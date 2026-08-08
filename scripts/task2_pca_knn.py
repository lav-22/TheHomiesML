"""Task 2: PCA dimension reduction followed by KNN classification.

The task brief allows sklearn for this task, so PCA and KNN both come from
sklearn. The number of neighbours is fixed at 2 by the brief.

Run from the repository root:

    python3 scripts/task2_pca_knn.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import calculate_macro_f1
from src.submission import create_submission

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "extra" / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
FIGURES_DIR = PROJECT_ROOT / "extra" / "figures"

for directory in (RESULTS_DIR, SUBMISSIONS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
ID_COLUMN = "id"
LABEL_COLUMN = "label"
N_NEIGHBOURS = 2                       # fixed by the task brief
COMPONENT_SETTINGS = [2000, 1000, 500, 100]

# Task 1 and 2 must use the pre-processed 5000 TF-IDF features
train_features = pd.read_csv(DATA_DIR / "train_features.csv")
test_features = pd.read_csv(DATA_DIR / "test_features.csv")

FEATURES = [c for c in train_features.columns if c not in (ID_COLUMN, LABEL_COLUMN)]
assert len(FEATURES) == 5000, f"expected 5000 features, got {len(FEATURES)}"

X_all = train_features[FEATURES].to_numpy(dtype=np.float32)
y_all = train_features[LABEL_COLUMN].to_numpy()
X_test = test_features[FEATURES].to_numpy(dtype=np.float32)

# Use the shared validation split so the scores line up with the other tasks
split = pd.read_csv(DATA_DIR / "splits" / "shared_validation_split.csv")
assert len(split) == len(train_features)
assert split[ID_COLUMN].astype("string").tolist() == train_features[ID_COLUMN].astype("string").tolist()

is_train = (split["split"] == "train").to_numpy()
X_train, y_train = X_all[is_train], y_all[is_train]
X_val, y_val = X_all[~is_train], y_all[~is_train]

print(X_train.shape, X_val.shape, X_test.shape)
print(round(y_train.mean(), 4), round(y_val.mean(), 4))


def evaluate_pca(n_components):
    """Fit PCA and KNN on the training split and score on the validation split."""
    start = time.time()

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
    knn.fit(X_train_pca, y_train)
    predictions = knn.predict(X_val_pca)

    return {
        "components": n_components,
        "val_macro_f1": calculate_macro_f1(y_val, predictions),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pred_class1_rate": float(predictions.mean()),
        "runtime_s": round(time.time() - start, 1),
    }


# Validation experiments
results = []
for n_components in COMPONENT_SETTINGS:
    print(f"\nRunning PCA with {n_components} components...")
    result = evaluate_pca(n_components)
    results.append(result)
    print(f"  val Macro F1      : {result['val_macro_f1']:.4f}")
    print(f"  explained variance: {result['explained_variance']:.4f}")
    print(f"  runtime           : {result['runtime_s']}s")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "pca_knn_results.csv", index=False)
print("\n", results_df, sep="")

# Refit on all 20000 labelled rows and predict the test set, one file per setting
for n_components in COMPONENT_SETTINGS:
    print(f"\nTraining final model ({n_components} components)...")

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_all_pca = pca.fit_transform(X_all)
    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
    knn.fit(X_all_pca, y_all)
    test_preds = knn.predict(X_test_pca)

    submission_path = SUBMISSIONS_DIR / f"PCA{n_components}_KNN_Prediction.csv"
    create_submission(
        test_ids=test_features[ID_COLUMN],
        predictions=test_preds,
        output_path=submission_path,
        id_column=ID_COLUMN,
        label_column=LABEL_COLUMN,
    )
    print(f"  saved {submission_path.name}; class-1 rate={test_preds.mean():.4f}")

# Component analysis: how much variance the retained components actually keep.
# Fitting once at the largest setting gives the whole curve for free.
full_pca = PCA(n_components=max(COMPONENT_SETTINGS), random_state=RANDOM_SEED)
full_pca.fit(X_train)
cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

variance_rows = []
for n_components in sorted(COMPONENT_SETTINGS):
    variance_rows.append({
        "components": n_components,
        "share_of_original_features": n_components / len(FEATURES),
        "cumulative_explained_variance": float(cumulative_variance[n_components - 1]),
    })
variance_df = pd.DataFrame(variance_rows)
variance_df.to_csv(RESULTS_DIR / "pca_variance_analysis.csv", index=False)
print("\n", variance_df, sep="")

# Why the scores fall as components are added: with an even k, the two
# neighbours often disagree, and sklearn settles a tie by taking the lowest
# class index, which is class 0. Counting those ties explains the drop.
tie_rows = []
for n_components in COMPONENT_SETTINGS:
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
    knn.fit(X_train_pca, y_train)
    _, neighbour_idx = knn.kneighbors(X_val_pca)
    neighbour_labels = y_train[neighbour_idx]

    is_tie = neighbour_labels[:, 0] != neighbour_labels[:, 1]
    tie_rows.append({
        "components": n_components,
        "tie_rate": float(is_tie.mean()),
        "true_class1_rate_among_ties": float(y_val[is_tie].mean()),
        "val_macro_f1_k2": calculate_macro_f1(y_val, knn.predict(X_val_pca)),
        # the nearest neighbour alone, i.e. the same model without the tie rule
        "val_macro_f1_k1": calculate_macro_f1(y_val, neighbour_labels[:, 0]),
    })

tie_df = pd.DataFrame(tie_rows)
tie_df.to_csv(RESULTS_DIR / "pca_knn_tie_analysis.csv", index=False)
print("\n", tie_df, sep="")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, linewidth=2)
for n_components in COMPONENT_SETTINGS:
    axes[0].axvline(n_components, color="grey", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Number of components")
axes[0].set_ylabel("Cumulative explained variance")
axes[0].set_title("How much variance the components retain")
axes[0].grid(alpha=0.3)

axes[1].plot(results_df["components"], results_df["val_macro_f1"],
             marker="o", linewidth=2, color="darkorange")
axes[1].set_xlabel("Number of PCA components")
axes[1].set_ylabel("Validation Macro F1")
axes[1].set_title(f"Components vs Macro F1 (KNN, k={N_NEIGHBOURS})")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pca_knn_analysis.png", dpi=150)
print("\nSaved reports/figures/pca_knn_analysis.png")
