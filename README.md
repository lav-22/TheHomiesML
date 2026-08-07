# TheHomiesML

50.007 Machine Learning course project — GenAI content detection.

Given a piece of text, classify whether it is human-authored (0) or
machine-generated (1). Scored by Macro F1.

## Deliverables

| Deliverable | Location |
|---|---|
| Notebook, Tasks 1–3 | `notebooks/TheHomiesML_Final_Submission.ipynb` |
| Final Kaggle prediction | `submissions/Final_Prediction.csv` |
| Report | `reports/final_report.md` |
| Task 1 prediction | `submissions/LogReg_Prediction.csv` |
| Task 2 predictions | `submissions/PCA{2000,1000,500,100}_KNN_Prediction.csv` |

## Results

Macro F1 on the shared validation split (`data/splits/shared_validation_split.csv`).

| Task | Model | Val Macro F1 |
|---|---|---|
| 3 | **Hybrid TF-IDF + stylometry + linear SVM (from scratch)** | **0.8307** |
| 3 | Extra Trees (tuned threshold) | 0.7621 |
| 3 | Linear SVM by SGD (from scratch) | 0.7446 |
| 1 | Logistic Regression (from scratch) | 0.7440 |
| 3 | Multinomial Naive Bayes (from scratch) | 0.6741 |
| 3 | Complement Naive Bayes (from scratch) | 0.6669 |
| 2 | PCA(100) + KNN (k=2) | 0.6556 |

The headline finding: swapping the provided 5000 TF-IDF features for our own
word + character TF-IDF and 78 stylometry features lifted the *same* model from
0.7446 to 0.8305. The representation mattered far more than the algorithm.

## Setup

```bash
pip install numpy pandas scipy scikit-learn matplotlib nbformat nbconvert ipykernel
```

Download `train.csv`, `test.csv`, `train_features.csv`, `test_features.csv` and
`sample_submission.csv` from Kaggle into `data/`. The two largest are gitignored.

```bash
# Regenerate everything
jupyter-nbconvert --to notebook --execute --inplace \
    notebooks/TheHomiesML_Final_Submission.ipynb
```

See `docs/model_experiments.md` for the individual experiment scripts and for
notes on which libraries the task rules permit.
