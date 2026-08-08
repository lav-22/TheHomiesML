# Scratch model experiments

This branch contains only model pipelines whose learning algorithms are
implemented directly in the project. NumPy and pandas are used for numerical
and data operations; no sklearn, SciPy or XGBoost estimators are used.

## Logistic regression

`notebooks/logistic_regression_from_scratch.ipynb` implements sigmoid, log
loss, gradients, mini-batch training, prediction and Macro F1 directly. Its
Kaggle output is `submissions/LogReg_Prediction.csv`.

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
`submissions/`.

See `docs/task3_model_implementations.md` for the complete Task 3 model list,
implementation locations, key hyperparameters and validation results.
