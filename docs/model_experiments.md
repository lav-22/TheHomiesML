# Model experiments and submissions

Runnable model code is stored in `scripts/`, experiment outputs in `results/`,
and Kaggle-ready prediction files in `submissions/`.

Run the complete validation experiment from the repository root:

```bash
python3 scripts/extra_trees_and_ensembles.py
```

The script:

1. Loads and validates Member 1's fixed train/validation split.
2. Runs the required Extra Trees baseline and one-factor-at-a-time tests.
3. Selects the Extra Trees configuration with the highest validation Macro F1.
4. Reproduces the completed Member 1 logistic-regression configuration and
   Member 4 Linear SVM configuration on the same split to obtain aligned scores.
5. Tests two-model and three-model equal/weighted soft-voting ensembles.
6. Writes all result tables and validation predictions to `results/`.

The large course CSV files belong in `data/` and are ignored by Git.

Generate the final Kaggle submission after selecting the ensemble:

```bash
python3 scripts/generate_ensemble2_submission.py
```

This writes `submissions/Ensemble2_Prediction.csv` with the required
`id,label` columns and all 6,999 test IDs in their original order.

Run the separate XGBoost experiment and create its Kaggle submission with:

```bash
brew install libomp  # one-time macOS prerequisite for XGBoost
.venv/bin/python scripts/xgboost_classifier.py
```

This writes the validation comparison to `results/xgboost_results.csv`
and the final test predictions to `submissions/XGBoost_Prediction.csv`.

Run the TF-IDF weighted RBF-kernel SVM experiment with:

```bash
python3 scripts/tfidf_rbf_svm.py
```

This writes `results/rbf_svm_results.csv` and the Kaggle-ready
`submissions/TFIDF_RBF_SVM_Prediction.csv`.

Generate the two feature-engineered Linear SVM submissions with:

```bash
python3 scripts/hybrid_tfidf_linear_svm.py
```

The two outputs are `submissions/Hybrid_Word_Char_TFIDF_Prediction.csv` and
`submissions/Hybrid_TFIDF_Stylometry_Prediction.csv`.

Run the hybrid word/character TF-IDF SGD experiment with:

```bash
python3 scripts/hybrid_tfidf_sgd.py
```

It creates a standard prediction file and an optional rank-thresholded file
whose class-1 rate is 55%, matching the strongest existing Kaggle SGD model.

Run the hybrid TF-IDF, stylometry and SGD experiment with:

```bash
python3 scripts/hybrid_tfidf_stylometry_sgd.py
```

It creates standard and Rank55 submissions in `submissions/` and writes the
validation comparison to `results/hybrid_stylometry_sgd_results.csv`.
