"""Regenerate the shared train/validation split.

Every member scores against this one split so the numbers in the report are
comparable. It is committed to `data/splits/`, so this script only needs
running if the file is lost — and it reproduces it exactly, because the seed
and the stratification are fixed.

Run from the repository root:

    python3 scripts/make_shared_split.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = DATA_DIR / "splits" / "shared_validation_split.csv"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
ID_COLUMN, LABEL_COLUMN = "id", "label"

train_df = pd.read_csv(DATA_DIR / "train.csv")

# Split the row indices rather than the frame, so the original row order is kept
train_indices, validation_indices = train_test_split(
    train_df.index,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=train_df[LABEL_COLUMN],
)

split_labels = pd.Series("train", index=train_df.index, name="split")
split_labels.loc[validation_indices] = "validation"

shared_split = pd.DataFrame({
    "row_index": train_df.index,
    ID_COLUMN: train_df[ID_COLUMN],
    "split": split_labels,
})

assert len(train_indices) + len(validation_indices) == len(train_df)
assert set(train_indices).isdisjoint(set(validation_indices))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# If the committed split already exists, confirm we reproduce it exactly rather
# than quietly writing a different one
if OUT_PATH.exists():
    existing = pd.read_csv(OUT_PATH)
    matches = existing["split"].tolist() == shared_split["split"].tolist()
    print("reproduces the committed split exactly:", matches)
    if not matches:
        raise SystemExit("refusing to overwrite: the regenerated split differs")

shared_split.to_csv(OUT_PATH, index=False)

print(f"wrote {OUT_PATH}")
print("train rows:", (shared_split['split'] == 'train').sum(),
      " validation rows:", (shared_split['split'] == 'validation').sum())
print("class-1 rate — train:",
      round(train_df.loc[train_indices, LABEL_COLUMN].mean(), 4),
      " validation:", round(train_df.loc[validation_indices, LABEL_COLUMN].mean(), 4))
