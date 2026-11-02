"""Verify a downloaded RQ2 decision and optionally its matching CSV.

This checks export consistency against the locally saved policy. It does not
establish where the files were downloaded or validate simulated incident facts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rq2_core import SCHEMA_VERSION, record_csv
from rq2_policy_v4_hardened import Event, decide


def verify(record: dict, csv_text: str | None = None) -> dict:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unrecognised decision-record schema")
    expected_digest = hashlib.sha256((ROOT / "rq2_policy_v4_hardened.py").read_bytes()).hexdigest()
    if record.get("source_identity", {}).get("policy_sha256") != expected_digest:
        raise ValueError("The record belongs to a different policy; use the matching source version")
    expected = decide(Event.from_scenario(record["policy_inputs"])).as_dict()
    if record.get("recommendation") != expected or expected["execution_authorised"] is not False:
        raise ValueError("The recorded obligations differ from the saved policy replay")
    if csv_text is not None:
        from io import StringIO
        rows = list(csv.DictReader(StringIO(csv_text)))
        if len(rows) != 1:
            raise ValueError("The decision CSV must contain exactly one record")
        row = rows[0]
        if row.get("record_id") != record["record_id"]:
            raise ValueError("CSV and JSON refer to different records")
        if json.loads(row["policy_inputs_json"]) != record["policy_inputs"]:
            raise ValueError("CSV and JSON contain different evaluated inputs")
        if json.loads(row["reason_codes_json"]) != expected["reason_codes"]:
            raise ValueError("CSV and JSON contain different reasons")
        for key in ("request_source_evidence", "SOC_review", "human_approval_required", "auto_collect_evidence", "no_case", "execution_authorised"):
            if row.get(key) != str(expected[key]):
                raise ValueError(f"CSV and JSON differ on {key}")
        expected_row = next(csv.DictReader(StringIO(record_csv(record).decode())))
        if row != expected_row:
            raise ValueError("CSV and JSON differ in their prediction or record metadata")
    return {"record_id": record["record_id"], "policy_sha256": expected_digest,
            "policy_replay_matches": True, "csv_matches_json": csv_text is not None,
            "execution_authorised": False,
            "scope": "Export consistency and saved-policy replay only; live deployment, incident truth and independent V4 validation are not established."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_json", type=Path)
    parser.add_argument("decision_csv", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, default=ROOT / "verification/latest_rq2_export_comparison.json")
    args = parser.parse_args()
    try:
        result = verify(json.loads(args.decision_json.read_text()), args.decision_csv.read_text() if args.decision_csv else None)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
