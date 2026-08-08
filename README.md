# TheHomiesML

Machine-learning project for classifying human-authored and machine-generated
text without deep learning. Standalone models are implemented from scratch;
ensemble pipelines may use existing machine-learning libraries.

## Repository structure

- `data/splits/` - shared fixed train/validation split
- `notebooks/` - from-scratch logistic regression notebook
- `scripts/` - from-scratch standalone models and library-based ensembles
- `results/` - validation and tuning results for retained models
- `src/` - shared from-scratch evaluation and submission helpers
- `submissions/` - Kaggle-ready `id,label` files
- `reports/` - figures and report assets
- `docs/` - experiment instructions

See `docs/model_experiments.md` for the remaining scratch implementations.
