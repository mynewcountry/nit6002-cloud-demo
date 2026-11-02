from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_core import infer, load_packaged_model, read_csv_bytes
from rq2_core import build_event, evaluate_decision, prediction_evidence, record_csv, record_json
from rq2_policy_v4_hardened import Event, decide


class DecisionIntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = json.loads((ROOT / "demo/rq2_examples.json").read_text())["examples"]
        cls.output, _ = infer(read_csv_bytes((ROOT / "demo/sample_inputs.csv").read_bytes()), load_packaged_model())

    def test_adapter_matches_reference_for_all_90_real_predictions(self):
        context = {"critical_asset": False, "privileged_access": False,
                   "secondary_signal": True, "additional_evidence": {"requested_action": "disable_privileged_identity"}}
        for _, row in self.output.iterrows():
            evidence = prediction_evidence(row)
            record = evaluate_decision(evidence, context, "included_model_sample", {"input_row": int(row.input_row)})
            reference = decide(Event.from_scenario(build_event(evidence, context))).as_dict()
            self.assertEqual(record["recommendation"], reference)
            self.assertTrue(record["recommendation"]["human_approval_required"])
            self.assertFalse(record["recommendation"]["execution_authorised"])

    def test_dataset_labels_and_identifiers_do_not_change_policy_inputs(self):
        row = self.output.iloc[0].to_dict()
        expected = prediction_evidence(row)
        row.update(supplied_label="attack", source_id="benign_data/changes", known_injected_flip=True)
        self.assertEqual(prediction_evidence(row), expected)
        evidence = self.examples[0]["evidence"]
        for bad in [{"supplied_label": "attack"}, {"source_id": "attack_data/a"}, {"all_zero": False}, {"nonfinite_features": False}]:
            with self.subTest(bad=bad):
                context = deepcopy(self.examples[0]["context"])
                context["additional_evidence"].update(bad)
                with self.assertRaises(ValueError): build_event(evidence, context)

    def test_untrusted_prediction_fields_are_rejected(self):
        row = self.output.iloc[0].to_dict()
        for field, value in [("model_prediction", "unknown"), ("model_attack_score", float("nan")),
                             ("model_attack_score", 1.1), ("model_attack_score", True),
                             ("all_zero_predictors", "False")]:
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                prediction_evidence({**row, field:value})

    def test_csv_and_json_exports_reconcile_every_example(self):
        for example in self.examples:
            with self.subTest(example=example["id"]):
                record = evaluate_decision(example["evidence"], example["context"], "illustrative_scenario", {"illustrative_case": example["id"]})
                document = json.loads(record_json(record))
                row = next(csv.DictReader(io.StringIO(record_csv(record).decode())))
                self.assertEqual(json.loads(row["policy_inputs_json"]), document["policy_inputs"])
                self.assertEqual(json.loads(row["reason_codes_json"]), document["recommendation"]["reason_codes"])
                for key in ["request_source_evidence", "SOC_review", "human_approval_required", "auto_collect_evidence", "no_case", "execution_authorised"]:
                    self.assertEqual(row[key], str(document["recommendation"][key]))
                self.assertEqual(row["record_id"], document["record_id"])
                self.assertEqual(document["source_identity"]["policy_sha256"], hashlib.sha256((ROOT / "rq2_policy_v4_hardened.py").read_bytes()).hexdigest())

    def test_quality_evidence_cannot_be_overridden_by_context(self):
        example = deepcopy(next(e for e in self.examples if e["id"] == "N28"))
        for value in (False, True):
            context = deepcopy(example["context"])
            context["additional_evidence"]["nonfinite_features"] = value
            with self.assertRaises(ValueError): build_event(example["evidence"], context)
        record = evaluate_decision(example["evidence"], example["context"], "illustrative_scenario", {"illustrative_case": "N28"})
        self.assertTrue(record["recommendation"]["request_source_evidence"])
        self.assertTrue(record["recommendation"]["human_approval_required"])

    def test_fingerprint_changes_with_context_but_not_evaluation_time(self):
        case = self.examples[0]
        args = [case["evidence"], case["context"], "illustrative_scenario", {"illustrative_case": case["id"]}]
        first = evaluate_decision(*args, evaluated_at="2026-11-02T00:00:00+00:00")
        again = evaluate_decision(*args, evaluated_at="2026-11-02T00:01:00+00:00")
        self.assertEqual(first["input_fingerprint"], again["input_fingerprint"])
        changed = deepcopy(case["context"]); changed["secondary_signal"] = True
        last = evaluate_decision(case["evidence"], changed, args[2], args[3])
        self.assertNotEqual(first["input_fingerprint"], last["input_fingerprint"])


if __name__ == "__main__": unittest.main(verbosity=2)
