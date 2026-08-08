"""Check every file in submissions/ against the Kaggle format.

Catches the things that silently cost a submission: wrong column names, a
missing or extra row, IDs reordered by a merge, labels outside {0, 1}, or a
predicted class rate so far from the training rate that something has gone
wrong upstream.

Run from the repository root:

    python3 scripts/verify_submissions.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

ID_COLUMN, LABEL_COLUMN = "id", "label"
TRAIN_CLASS1_RATE = 0.6252

expected_ids = (pd.read_csv(DATA_DIR / "test_features.csv", usecols=[ID_COLUMN])[ID_COLUMN]
                .astype("string")
                .tolist())
print(f"expecting {len(expected_ids)} rows, IDs in test_features.csv order\n")

failures = []
for path in sorted(SUBMISSIONS_DIR.glob("*.csv")):
    problems = []
    frame = pd.read_csv(path, dtype={ID_COLUMN: "string"})

    if frame.columns.tolist() != [ID_COLUMN, LABEL_COLUMN]:
        problems.append(f"columns are {frame.columns.tolist()}, expected ['id', 'label']")
    if len(frame) != len(expected_ids):
        problems.append(f"{len(frame)} rows, expected {len(expected_ids)}")
    if LABEL_COLUMN in frame and frame[LABEL_COLUMN].isnull().any():
        problems.append("labels contain missing values")
    if LABEL_COLUMN in frame and not set(frame[LABEL_COLUMN].dropna().unique()) <= {0, 1}:
        problems.append(f"labels outside {{0, 1}}: {sorted(set(frame[LABEL_COLUMN]))}")
    if ID_COLUMN in frame:
        if not frame[ID_COLUMN].is_unique:
            problems.append("duplicate IDs")
        if len(frame) == len(expected_ids) and frame[ID_COLUMN].tolist() != expected_ids:
            problems.append("IDs do not match the test set order")

    rate = float(frame[LABEL_COLUMN].mean()) if LABEL_COLUMN in frame else float("nan")
    status = "FAIL" if problems else "ok  "
    print(f"{status} {path.name:<42} class-1 rate={rate:.4f}")

    # Not a failure, just worth flagging: a rate far from the training rate
    # usually means a threshold or tie-break problem rather than a real signal
    if not problems and abs(rate - TRAIN_CLASS1_RATE) > 0.25:
        print(f"     ^ note: far from the training rate of {TRAIN_CLASS1_RATE}")

    for problem in problems:
        print(f"     - {problem}")
        failures.append((path.name, problem))

print()
if failures:
    print(f"{len(failures)} problem(s) found")
    sys.exit(1)
print("all submission files are valid")
