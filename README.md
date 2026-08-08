# TheHomiesML

50.007 Machine Learning course project — GenAI content detection.

Given a piece of text, classify whether it is human-authored (0) or
machine-generated (1). Scored by Macro F1.


```
notebooks/TheHomiesML_Final_Submission.ipynb   Tasks 1-3, one notebook
submissions/                                   the six Kaggle prediction files
src/                                           helpers the notebook imports
data/splits/                                   the shared validation split
```

| Submission file | Task |
|---|---|
| `Final_Prediction.csv` | 3 |
| `LogReg_Prediction.csv` | 1 |
| `PCA2000_KNN_Prediction.csv` | 2 |
| `PCA1000_KNN_Prediction.csv` | 2 |
| `PCA500_KNN_Prediction.csv` | 2 |
| `PCA100_KNN_Prediction.csv` | 2 |

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
`f1_score` as the metric.
