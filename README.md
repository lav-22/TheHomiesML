# TheHomiesML

50.007 Machine Learning course project — GenAI content detection.

Given a piece of text, classify whether it is human-authored (0) or
machine-generated (1). Scored by Macro F1.

Standalone learning algorithms are implemented from scratch. Existing machine-
learning libraries are used only where the project permits them, including PCA,
KNN and ensemble models such as Extra Trees and the final library ensemble.

## What to look at first

`notebooks/TheHomiesML_Final_Submission.ipynb` is the submitted notebook. It
covers Tasks 1–3 end to end, is stored with the outputs of a clean top-to-bottom
run, and writes all six graded prediction files.

## Repository structure

- `notebooks/TheHomiesML_Final_Submission.ipynb` — the submitted notebook, Tasks 1–3
- `submissions/` — the six graded prediction files, and nothing else
- `submissions/experiments/` — every other run we kept as evidence
- `data/splits/` — the shared fixed train/validation split
- `src/` — from-scratch models, features, evaluation and submission helpers
- `scripts/` — the individual experiment drivers and the notebook builder
- `results/` — validation scores and hyperparameter tables
- `figures/` — plots produced by the notebook and scripts
- `models/` — saved enhanced-scratch and library-ensemble artifacts
- `docs/` — experiment and implementation documentation
- `archive/` — superseded working notebooks, kept for the record only

See `docs/model_experiments.md` and `docs/task3_model_implementations.md` for
the retained model settings, implementation notes and result-file locations.

## Graded submission files

All six are written by the final notebook and live directly in `submissions/`.

| Submission file | Task | Written by |
|---|---|---|
| `LogReg_Prediction.csv` | 1 | Section 1.4 |
| `PCA2000_KNN_Prediction.csv` | 2 | Section 2.2 |
| `PCA1000_KNN_Prediction.csv` | 2 | Section 2.2 |
| `PCA500_KNN_Prediction.csv` | 2 | Section 2.2 |
| `PCA100_KNN_Prediction.csv` | 2 | Section 2.2 |
| `Final_Prediction.csv` | 3 | Section 3.7 |

`submissions/experiments/` holds the Extra Trees ensembles, the enhanced
from-scratch TF-IDF/stylometry SGD variants, the permitted library
SVM/stylometry ensemble and the rank-calibration variants. The two
`*_DecisionScores.csv` files there are raw scores kept for calibration
analysis, not `id,label` uploads.

## Results

Macro F1 on the shared validation split (`data/splits/shared_validation_split.csv`),
taken from the stored outputs of the final notebook. "Tuned" means the decision
threshold was chosen on validation as well.

| Task | Model | Val Macro F1 | Tuned |
|---|---|---|---|
| 3 | **Hybrid TF-IDF + stylometry + SGD, modified Huber (from scratch)** | **0.8513** | — |
| 3 | Linear classifier by SGD (from scratch) | 0.7446 | — |
| 3 | Soft vote (linear + Extra Trees + NB) | 0.7271 | 0.7462 |
| 1 | Logistic Regression (from scratch) | 0.7227 | 0.7440 |
| 3 | Extra Trees (sklearn ensemble, permitted) | 0.7189 | 0.7598 |
| 3 | Multinomial Naive Bayes (from scratch) | 0.6741 | — |
| 3 | Complement Naive Bayes (from scratch) | 0.6669 | — |
| 2 | PCA(100) + KNN (k=2) | 0.6556 | — |

Three headline findings:

- Swapping the provided 5000 TF-IDF features for our own word + character TF-IDF
  and 78 stylometry features lifted the *same* model from 0.7446 to 0.8513. The
  representation mattered far more than the algorithm.
- We had settled on modified Huber loss early, then built the hybrid pipeline
  with hinge loss and never re-tested. Re-testing was worth a further +0.02 —
  a hyperparameter validated on one representation does not carry to another.
- The random 80/20 split scores the final model near 0.85, but a cluster-holdout
  validation (Section 3.7) puts it near 0.80, which is where the leaderboard
  actually landed. The gap is domain shift, not a bug.

## Setup

```bash
pip install numpy pandas scipy scikit-learn matplotlib nbformat nbconvert ipykernel
```

Download `train.csv`, `test.csv`, `train_features.csv`, `test_features.csv` and
`sample_submission.csv` from Kaggle into `data/`. They are gitignored for size.

```bash
# Regenerate every graded submission
jupyter-nbconvert --to notebook --execute --inplace \
    notebooks/TheHomiesML_Final_Submission.ipynb

# Check the prediction files before uploading
python3 scripts/verify_submissions.py
```

`verify_submissions.py` checks that all six graded files are present, correctly
formatted, in test-set ID order, and free of Git conflict markers; it lists the
experiment files separately.

`random_state=42` throughout. See `scripts/` for the individual experiment
drivers, and `scripts/build_final_notebook.py` for the builder that assembles
the submitted notebook from `src/` — running it regenerates the notebook without
outputs, so re-execute the notebook afterwards.

## Rules compliance

Task 1 and Task 3 models are written out with NumPy — no sklearn estimator
learns the task, and the Macro F1 metric in `src/evaluation.py` is written out
from its definition rather than imported. sklearn appears in the final notebook
in exactly three places:

- **PCA and KNN** in Task 2, which the brief explicitly permits;
- **Extra Trees** in Task 3, permitted because the brief allows libraries for
  ensemble models;
- **KMeans** in Section 3.7, used only to group training rows into topic
  clusters so we could build a harder validation split. It never predicts a
  label.

The dedicated library SVM/stylometry ensemble under `scripts/library_ensemble.py`
is retained under the same ensemble exception; its predictions are in
`submissions/experiments/`.
