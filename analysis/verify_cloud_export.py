"""Compare the included sample's downloaded predictions with the recorded model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="?", type=Path, default=Path.home() / "Downloads/predictions.csv")
    args = parser.parse_args()
    path = args.predictions.expanduser()
    actual = pd.read_csv(path)
    expected = pd.read_csv(root / "demo/expected_predictions.csv")
    required = {"source_id", "model_prediction", "model_attack_score"}
    if not required.issubset(actual.columns):
        raise ValueError("Select the predictions.csv downloaded after running the included sample.")
    if len(actual) != len(expected) or actual.source_id.duplicated().any() or set(actual.source_id) != set(expected.source_id):
        raise ValueError("The downloaded file does not contain exactly the included 90 sample observations.")
    joined = expected.merge(actual, on="source_id", validate="one_to_one")
    numeric_prediction = joined.model_prediction.map({"benign": 0, "attack": 1})
    matches = bool(np.array_equal(numeric_prediction, joined.expected_prediction))
    max_difference = float(np.max(abs(joined.model_attack_score - joined.expected_attack_score)))
    scores_match = bool(np.allclose(joined.model_attack_score, joined.expected_attack_score, atol=1e-12, rtol=0))
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(actual),
        "all_predicted_classes_match": matches,
        "attack_scores_match_with_absolute_tolerance_1e_minus12": scores_match,
        "maximum_attack_score_difference": max_difference,
        "scope": "Downloaded CSV comparison. Record the actual cloud URL and access evidence separately.",
    }
    (root / "verification/latest_export_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not (matches and scores_match):
        raise SystemExit("FAIL: inspect the package version, feature schema and deployed model.")
    print("PASS: all 90 predictions match the packaged reference.")


if __name__ == "__main__":
    main()
