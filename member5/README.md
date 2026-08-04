# Member 5: Extra Trees and ensembles

This folder contains only Member 5's work. It does not modify the notebooks or
outputs produced by other team members.

Run the complete validation experiment from the repository root:

```bash
python member5/run_experiments.py
```

The script:

1. Loads and validates Member 1's fixed train/validation split.
2. Runs the required Extra Trees baseline and one-factor-at-a-time tests.
3. Selects the Extra Trees configuration with the highest validation Macro F1.
4. Reproduces the completed Member 1 logistic-regression configuration and
   Member 4 Linear SVM configuration on the same split to obtain aligned scores.
5. Tests two-model and three-model equal/weighted soft-voting ensembles.
6. Writes all result tables and validation predictions to `member5/results/`.

The large course CSV files belong in `data/` and are ignored by Git.

Generate the final Kaggle submission after selecting the ensemble:

```bash
python3 member5/generate_submission.py
```

This writes `submissions/Member5_Ensemble2_Prediction.csv` with the required
`id,label` columns and all 6,999 test IDs in their original order.

Run the separate XGBoost experiment and create its Kaggle submission with:

```bash
brew install libomp  # one-time macOS prerequisite for XGBoost
.venv/bin/python member5/run_xgboost.py
```

This writes the validation comparison to `member5/results/xgboost_results.csv`
and the final test predictions to `submissions/XGBoost_Prediction.csv`.

Run the TF-IDF weighted RBF-kernel SVM experiment with:

```bash
python3 member5/run_rbf_svm.py
```

This writes `member5/results/rbf_svm_results.csv` and the Kaggle-ready
`submissions/TFIDF_RBF_SVM_Prediction.csv`.

Generate the two feature-engineered Linear SVM submissions with:

```bash
python3 member5/run_feature_engineering.py
```

The two outputs are `submissions/Hybrid_Word_Char_TFIDF_Prediction.csv` and
`submissions/Hybrid_TFIDF_Stylometry_Prediction.csv`.
