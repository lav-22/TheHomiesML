# Scratch model experiments

Model pipelines whose learning algorithms are implemented directly in the
project. NumPy and pandas are used for numerical and data operations; no
sklearn, SciPy or XGBoost estimators are used.

These are the supporting experiments. The graded implementations and the six
graded prediction files are all in `notebooks/TheHomiesML_Final_Submission.ipynb`.

## Logistic regression

Task 1 is implemented in Section 1 of
`notebooks/TheHomiesML_Final_Submission.ipynb`, which defines sigmoid, log loss,
gradients, mini-batch training and prediction directly and writes the graded
`submissions/LogReg_Prediction.csv`.

`archive/logistic_regression_from_scratch.ipynb` is the earlier working notebook
for the same task, kept for the record. It runs a plain learning-rate / epoch /
batch-size grid without the L2, shuffling and threshold search that the final
version adds, and writes only to `submissions/experiments/`.

## Advanced stylometry TF-IDF SGD

Run from the repository root:

```bash
python3 scripts/advanced_stylometry_sgd.py
```

The script implements word and character TF-IDF, sparse matrix operations,
feature scaling, advanced stylometry, class-balanced hinge-loss SGD, weight
averaging and Macro F1 directly. It uses Member 1's fixed split in
`data/splits/shared_validation_split.csv`.

Tune its stylometry weight and validation threshold with:

```bash
python3 scripts/tune_advanced_stylometry_sgd.py
```

The tuning output is
`results/advanced_stylometry_threshold_results.csv`.

## Library-based ensembles

Existing ML libraries are allowed for ensemble models on this branch.

The required Extra Trees and weighted ensemble experiment is implemented in:

```bash
python3 scripts/extra_trees_and_ensembles.py
```

Its Ensemble 2 submission can be regenerated with:

```bash
python3 scripts/generate_ensemble2_submission.py
```

The stronger SVM and nonlinear stylometry ensemble is implemented in:

```bash
python3 scripts/library_ensemble.py
```

It validates the two constituent models and their equal-weight rank ensemble,
refits on all labelled data, saves trained artifacts under
`models/library_ensemble/`, and writes its Kaggle submissions to
`submissions/experiments/`.

See `docs/task3_model_implementations.md` for the complete Task 3 model list,
implementation locations, key hyperparameters and validation results.
