"""Create conservative calibration variants from saved enhanced-model scores."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "submissions" / "experiments" / "Enhanced_Stylometry_FromScratch_DecisionScores.csv"
OUTPUT = ROOT / "submissions" / "experiments"
SUMMARY = ROOT / "results" / "enhanced_calibration_variants_summary.csv"


def rank_labels(scores, positive_fraction):
    cutoff = float(np.quantile(scores, 1.0 - positive_fraction))
    return (scores >= cutoff).astype(int)


def grouped_labels(frame, numeric_fraction, uuid_fraction):
    numeric = frame["id"].astype(str).str.fullmatch(r"\d+").to_numpy()
    labels = np.zeros(len(frame), dtype=int)
    labels[numeric] = rank_labels(
        frame.loc[numeric, "decision_score"].to_numpy(), numeric_fraction
    )
    labels[~numeric] = rank_labels(
        frame.loc[~numeric, "decision_score"].to_numpy(), uuid_fraction
    )
    return labels


def save(frame, name, labels):
    submission = pd.DataFrame({"id": frame["id"], "label": labels})
    if len(submission) != 6_999 or submission["id"].nunique() != 6_999:
        raise ValueError("Invalid submission IDs.")
    path = OUTPUT / name
    submission.to_csv(path, index=False)
    numeric = submission["id"].astype(str).str.fullmatch(r"\d+")
    return {
        "file": name,
        "overall_positive_rate": submission["label"].mean(),
        "numeric_positive_rate": submission.loc[numeric, "label"].mean(),
        "uuid_positive_rate": submission.loc[~numeric, "label"].mean(),
    }


def main():
    frame = pd.read_csv(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fraction in (0.525, 0.575, 0.60):
        rows.append(save(
            frame,
            f"Enhanced_FromScratch_GlobalRank{fraction * 100:g}.csv",
            rank_labels(frame["decision_score"].to_numpy(), fraction),
        ))
    for numeric_fraction, uuid_fraction in (
        (0.55, 0.55),
        (0.50, 0.625),
        (0.55, 0.625),
        (0.60, 0.625),
    ):
        rows.append(save(
            frame,
            f"Enhanced_FromScratch_GroupRank_N{numeric_fraction * 100:g}_U{uuid_fraction * 100:g}.csv",
            grouped_labels(frame, numeric_fraction, uuid_fraction),
        ))
    table = pd.DataFrame(rows)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUMMARY, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
