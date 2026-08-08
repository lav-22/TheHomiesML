# TheHomiesML

50.007 Machine Learning course project — GenAI content detection.

Given a piece of text, classify whether it is human-authored (0) or
machine-generated (1). Scored by Macro F1.

Standalone learning algorithms are implemented from scratch. Existing machine-
learning libraries are used only where the project permits them, including PCA,
KNN and ensemble models such as Extra Trees and the final library ensemble.

## Repository structure

- `notebooks/TheHomiesML_Final_Submission.ipynb` - consolidated Tasks 1–3
- `notebooks/logistic_regression_from_scratch.ipynb` - Task 1 experiment
- `data/splits/` - shared fixed train/validation split
- `scripts/` - scratch models, Extra Trees tuning and ensemble generation
- `results/` - validation scores and hyperparameter experiments
- `submissions/` - Kaggle-ready `id,label` prediction files
- `models/` - saved enhanced scratch and library-ensemble artifacts
- `src/` - shared evaluation and submission helpers
- `reports/` - figures and report assets
- `docs/` - experiment and implementation documentation

See `docs/model_experiments.md` and `docs/task3_model_implementations.md` for
the retained model settings, implementation notes and result-file locations.

## Core submission files

| Submission file | Task |
|---|---|
| `Final_Prediction.csv` | 3 |
| `LogReg_Prediction.csv` | 1 |
| `PCA2000_KNN_Prediction.csv` | 2 |
| `PCA1000_KNN_Prediction.csv` | 2 |
| `PCA500_KNN_Prediction.csv` | 2 |
| `PCA100_KNN_Prediction.csv` | 2 |

Additional experiments in `submissions/` include the Extra Trees ensembles,
enhanced from-scratch TF-IDF/stylometry SGD variants, and the permitted library
SVM/stylometry ensemble. Decision-score files support calibration analysis and
are not direct Kaggle `id,label` uploads.

## Results

Macro F1 on the shared validation split (`data/splits/shared_validation_split.csv`).

| Task | Model | Val Macro F1 |
|---|---|---|
| 3 | **Hybrid TF-IDF + stylometry + SGD, modified Huber (from scratch)** | **0.8491** |
| 3 | Extra Trees (tuned threshold) | 0.7621 |
| 3 | Linear classifier by SGD (from scratch) | 0.7446 |
| 1 | Logistic Regression (from scratch) | 0.7440 |
| 3 | Multinomial Naive Bayes (from scratch) | 0.6741 |
| 3 | Complement Naive Bayes (from scratch) | 0.6669 |
| 2 | PCA(100) + KNN (k=2) | 0.6556 |

Two headline findings:

- Swapping the provided 5000 TF-IDF features for our own word + character TF-IDF
  and 78 stylometry features lifted the *same* model from 0.7446 to 0.8491. The
  representation mattered far more than the algorithm.
- We had settled on modified Huber loss early, then built the hybrid pipeline
  with hinge loss and never re-tested. Re-testing was worth a further +0.02 —
  a hyperparameter validated on one representation does not carry to another.

## Setup

```bash
pip install numpy pandas scipy scikit-learn matplotlib nbformat nbconvert ipykernel
```

Download `train.csv`, `test.csv`, `train_features.csv`, `test_features.csv` and
`sample_submission.csv` from Kaggle into `data/`. They are gitignored for size.

```bash
# Regenerate every submission
jupyter-nbconvert --to notebook --execute --inplace \
    notebooks/TheHomiesML_Final_Submission.ipynb

# Check the prediction files before uploading
python3 extra/scripts/verify_submissions.py
```

`random_state=42` throughout. See `extra/scripts/` for the individual experiment
drivers.

## Rules compliance

Task 1 and Task 3 models are written out with NumPy — no sklearn estimator
learns anything. sklearn appears only where the brief permits it: PCA and KNN in
Task 2, Extra Trees in Task 3 (libraries are allowed for ensemble models), and
`f1_score` as the metric. The dedicated library ensemble is retained under the
agreed ensemble exception.
