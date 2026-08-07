# Model experiments

Runnable code lives in `scripts/`, shared implementations in `src/`, experiment
output in `results/`, and Kaggle-ready prediction files in `submissions/`.

The large course CSVs belong in `data/` and are ignored by Git.

## The deliverable

`notebooks/TheHomiesML_Final_Submission.ipynb` is the single notebook covering
Tasks 1 to 3. It is assembled from `src/` so the notebook and the scripts can
never drift apart:

```bash
python3 scripts/build_final_notebook.py
jupyter-nbconvert --to notebook --execute --inplace \
    notebooks/TheHomiesML_Final_Submission.ipynb
```

Running it regenerates every submission file.

## Shared implementations

| File | Contents |
|---|---|
| `src/scratch_models.py` | Logistic regression, Multinomial NB, Complement NB and the SGD linear classifier — all from scratch, NumPy only |
| `src/text_features.py` | From-scratch TF-IDF vectorizer (word + character), standard scaler, 78 stylometry features, averaged hinge-loss SGD |
| `src/evaluation.py` | Macro F1 wrapper |
| `src/submission.py` | Submission writer with format checks |

## Experiment drivers

| Script | What it does | Output |
|---|---|---|
| `task1_logreg_tuning.py` | Task 1 tuning grid: shuffling, L2, class weight, threshold | `results/logreg_improved_results.csv` |
| `task2_pca_knn.py` | Task 2 PCA + KNN at 2000/1000/500/100 components, plus the variance and tie-break analysis | `results/pca_knn_*.csv`, 4 submissions |
| `task3_model_comparison.py` | Every Task 3 model on the provided 5000 features, one table | `results/task3_model_comparison.csv` |
| `task3_extra_trees_tuning.py` | One-factor-at-a-time Extra Trees tuning | `results/task3_extra_trees_tuning.csv` |
| `task3_final_model.py` | The final model: hybrid TF-IDF + stylometry, style weight and threshold tuned on validation | `submissions/Final_Prediction.csv` |
| `xgboost_classifier.py` | XGBoost experiment (needs `xgboost`, not installed in the final environment) | `results/xgboost_results.csv` |

Run everything from the repository root.

## Rules compliance

The brief requires Task 3 models to be implemented from scratch and only allows
existing libraries for **ensemble** models.

- **From scratch:** logistic regression, Multinomial NB, Complement NB, the SGD
  linear classifier (hinge / modified Huber / log loss), the TF-IDF vectorizer,
  the feature scaler and the stylometry features.
- **Library, permitted:** PCA and KNN in Task 2 (explicitly allowed by the
  brief), Extra Trees and XGBoost in Task 3 (ensembles), and `f1_score` as the
  evaluation metric.

Earlier versions of this project used `sklearn.svm.LinearSVC` and
`sklearn.svm.SVC` as Task 3 models. Neither is an ensemble, so both were
removed: the linear SVM is now the from-scratch hinge-loss SGD in
`src/text_features.py`, and the RBF SVM was dropped. The original code remains
on the `member5-extra-trees-ensemble` and `linear-svm-pipeline` branches.

Submissions that forced the test-set class-1 rate to a fixed quantile
("Rank55") were also removed — they fit the test distribution rather than
learning from the data.
