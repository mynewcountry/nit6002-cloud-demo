from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.verify_rq2_export import verify
from rq2_core import evaluate_decision, record_csv


class DownloadVerificationChecks(unittest.TestCase):
    def record(self):
        item = next(x for x in json.loads((ROOT / "demo/rq2_examples.json").read_text())["examples"] if x["id"] == "N27")
        return evaluate_decision(item["evidence"], item["context"], "illustrative_scenario", {"illustrative_case": "N27"})

    def test_matching_downloads_replay_successfully(self):
        record = self.record()
        result = verify(record, record_csv(record).decode())
        self.assertTrue(result["csv_matches_json"])
        self.assertFalse(result["execution_authorised"])

    def test_changed_approval_execution_or_policy_digest_is_rejected(self):
        record = self.record()
        for key, value in [("human_approval_required", False), ("execution_authorised", True)]:
            changed = deepcopy(record); changed["recommendation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify(changed)
        changed = deepcopy(record); changed["source_identity"]["policy_sha256"] = "0" * 64
        with self.assertRaises(ValueError): verify(changed)

    def test_mismatched_or_multiple_csv_records_are_rejected(self):
        record = self.record()
        csv_text = record_csv(record).decode()
        rows = list(csv.DictReader(io.StringIO(csv_text))); rows[0]["record_id"] = "wrong-record"
        output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
        with self.assertRaises(ValueError): verify(record, output.getvalue())
        writer.writerows(rows)
        with self.assertRaises(ValueError): verify(record, output.getvalue())


if __name__ == "__main__": unittest.main(verbosity=2)
