# Task 3 model implementations

This file lists every model family retained as a Task 3 experiment, its main
implementation file and its key hyperparameters. Logistic regression from Task
1 is documented separately and does not count toward the Task 3 model total.

## Standalone models implemented from scratch

| Model/experiment | Implementation | Key settings | Best fixed-validation Macro F1 |
|---|---|---|---:|
| Advanced TF-IDF + stylometry SGD | `scripts/advanced_stylometry_sgd.py` | Word 1-2 grams; character-within-word 3-5 grams; balanced averaged hinge SGD; alpha `1e-4`; stylometry weights `0.02`, `0.05`, `0.10` | 0.832403 after scratch rewrite |
| SGD/NBSVM hyperparameter search | `scripts/scratch_hyperparameter_search.py` | Learning rates `10-50`; alpha `1e-5-3e-4`; batch sizes `128-512`; 100-200 epochs; word/character/style weights; optional NB log-count ratio | 0.850147 focused search |
| TF-IDF feature-space search | `scripts/scratch_feature_space_search.py` | Word ranges `(1,2)` and `(1,3)`; character ranges `(3,5)` and `(2,6)`; raw and within-word characters; up to 160,000 features per family | 0.855148 |
| Enhanced stylometry SGD | `scripts/scratch_final_ensemble_search.py` | Word 1-3 grams; raw character 3-5 grams; 160,000 features per family; word weight `1.50`; style weight `0.10`; hinge loss; alpha `1e-4`; learning rate `50`; batch size `256`; 150 epochs | **0.859286** |
| Scratch logistic-loss SGD comparison | `scripts/scratch_final_ensemble_search.py` | Balanced L2 logistic loss; learning rates `2`, `5`, `10`; 150 epochs | 0.796243 |

The scratch pipelines directly implement TF-IDF, sparse matrix operations,
standardization, class balancing, hinge/logistic optimization, prediction and
Macro F1. They do not use sklearn or SciPy.

## Ensemble models allowed to use libraries

| Ensemble | Implementation | Key settings | Best fixed-validation Macro F1 |
|---|---|---|---:|
| Extra Trees ensembles | `scripts/extra_trees_and_ensembles.py` | Extra Trees counts `200`, `500`, `800`; multiple depths/features/class weights; equal and weighted two/three-model voting | 0.764140 for retained Ensemble 2 experiment |
| Linear SVM + nonlinear stylometry ensemble | `scripts/library_ensemble.py` | LinearSVC `C=1`, word 1-3 gram and raw character 3-5 gram TF-IDF; HistGradientBoosting with 300 iterations, learning rate `0.05`, 15 leaves and L2 `1.0`; equal rank weights | **0.890104** |

## Enhanced scratch prediction generation

Run:

```bash
python3 scripts/generate_enhanced_stylometry_submissions.py
python3 scripts/generate_enhanced_calibration_variants.py
```

The first command refits the enhanced scratch model, creates decision scores
and base submissions, and saves the fitted artifact to
`models/enhanced_from_scratch/enhanced_stylometry_sgd.pkl.gz`. The second
command creates the source-aware rank variants, including
`submissions/Enhanced_FromScratch_GroupRank_N55_U62.5.csv`.

Full experiment tables are stored in:

- `results/scratch_hyperparameter_results.csv`
- `results/scratch_hyperparameter_results_focused.csv`
- `results/scratch_feature_space_results.csv`
- `results/scratch_final_ensemble_results.csv`
- `results/library_ensemble_results.csv`
