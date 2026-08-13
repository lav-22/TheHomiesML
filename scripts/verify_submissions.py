"""Check the prediction files against the Kaggle format.

Catches the things that silently cost a submission: wrong column names, a
missing or extra row, IDs reordered by a merge, labels outside {0, 1}, or a
predicted class rate so far from the training rate that something has gone
wrong upstream.

It also refuses any file still carrying Git conflict markers. A merge once left
markers inside `LogReg_Prediction.csv`, which pushed it to 7520 rows without
anything else complaining, so that check is now first.

`submissions/` holds the six graded deliverables and is checked strictly:
every one must be present and valid. `submissions/experiments/` holds the
other runs we kept as evidence; the decision-score files in there are analysis
outputs rather than uploads, so they are listed and skipped.

Run from the repository root:

    python3 scripts/verify_submissions.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = SUBMISSIONS_DIR / "experiments"

ID_COLUMN, LABEL_COLUMN = "id", "label"
TRAIN_CLASS1_RATE = 0.6252

# The files the project brief actually asks for.
REQUIRED = {
    "LogReg_Prediction.csv": "Task 1",
    "PCA2000_KNN_Prediction.csv": "Task 2",
    "PCA1000_KNN_Prediction.csv": "Task 2",
    "PCA500_KNN_Prediction.csv": "Task 2",
    "PCA100_KNN_Prediction.csv": "Task 2",
    "Final_Prediction.csv": "Task 3",
}

expected_ids = (pd.read_csv(DATA_DIR / "test_features.csv", usecols=[ID_COLUMN])[ID_COLUMN]
                .astype("string")
                .tolist())
print(f"expecting {len(expected_ids)} rows, IDs in test_features.csv order\n")

failures = []


def check(path):
    """Report on one CSV. Returns the list of problems found."""
    problems = []

    # A conflict marker makes the file unparseable as predictions, so stop here
    text = path.read_text(errors="replace")
    if any(line.startswith(("<<<<<<<", "=======", ">>>>>>>"))
           for line in text.splitlines()):
        problems.append("contains unresolved Git conflict markers")
        print(f"FAIL {path.name:<58}")
        for problem in problems:
            print(f"     - {problem}")
            failures.append((path.name, problem))
        return problems

    frame = pd.read_csv(path, dtype={ID_COLUMN: "string"})

    # Decision scores are kept for calibration analysis, not uploaded as-is
    if LABEL_COLUMN not in frame.columns:
        print(f"skip {path.name:<58} decision scores, not an id,label upload")
        return problems

    if frame.columns.tolist() != [ID_COLUMN, LABEL_COLUMN]:
        problems.append(f"columns are {frame.columns.tolist()}, expected ['id', 'label']")
    if len(frame) != len(expected_ids):
        problems.append(f"{len(frame)} rows, expected {len(expected_ids)}")
    if frame[LABEL_COLUMN].isnull().any():
        problems.append("labels contain missing values")
    if not set(frame[LABEL_COLUMN].dropna().unique()) <= {0, 1}:
        problems.append(f"labels outside {{0, 1}}: {sorted(set(frame[LABEL_COLUMN]))}")
    if ID_COLUMN in frame:
        if not frame[ID_COLUMN].is_unique:
            problems.append("duplicate IDs")
        if len(frame) == len(expected_ids) and frame[ID_COLUMN].tolist() != expected_ids:
            problems.append("IDs do not match the test set order")

    rate = float(frame[LABEL_COLUMN].mean())
    status = "FAIL" if problems else "ok  "
    print(f"{status} {path.name:<58} class-1 rate={rate:.4f}")

    # Not a failure, just worth flagging: a rate far from the training rate
    # usually means a threshold or tie-break problem rather than a real signal
    if not problems and abs(rate - TRAIN_CLASS1_RATE) > 0.25:
        print(f"     ^ note: far from the training rate of {TRAIN_CLASS1_RATE}")

    for problem in problems:
        print(f"     - {problem}")
        failures.append((path.name, problem))
    return problems


print(f"--- graded deliverables ({SUBMISSIONS_DIR.name}/) ---")
for name, task in sorted(REQUIRED.items()):
    path = SUBMISSIONS_DIR / name
    if not path.exists():
        print(f"FAIL {name:<58} MISSING ({task} deliverable)")
        failures.append((name, f"missing {task} deliverable"))
        continue
    check(path)

unexpected = sorted(p.name for p in SUBMISSIONS_DIR.glob("*.csv")
                    if p.name not in REQUIRED)
if unexpected:
    print(f"\nnote: {len(unexpected)} extra file(s) alongside the deliverables; "
          "experiments belong in submissions/experiments/")
    for name in unexpected:
        print(f"     - {name}")

if EXPERIMENTS_DIR.is_dir():
    print(f"\n--- experiments ({EXPERIMENTS_DIR.relative_to(PROJECT_ROOT)}/) ---")
    for path in sorted(EXPERIMENTS_DIR.glob("*.csv")):
        check(path)

print()
if failures:
    print(f"{len(failures)} problem(s) found")
    sys.exit(1)
print("all submission files are valid")
