from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest


class DecisionViewChecks(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.radio(key="page").set_value("RQ2 decision support").run()
        self.assertEqual(len(app.exception), 0)
        return app

    def evaluate(self, app):
        next(button for button in app.button if button.label == "Evaluate decision").click().run()
        self.assertEqual(len(app.exception), 0)
        return app.session_state["rq2_decision_record"]

    def example(self, app, ident):
        app.radio(key="rq2_source").set_value("Illustrative scenario").run()
        app.selectbox(key="rq2_example").set_value(ident).run()
        self.assertEqual(len(app.exception), 0)
        return self.evaluate(app)

    def test_real_inference_can_generate_exportable_obligations(self):
        app = self.app()
        record = self.evaluate(app)
        self.assertEqual(record["evidence_origin"], "included_model_sample")
        self.assertIn("input_sha256", record["provenance"])
        self.assertNotIn("supplied_label", record["policy_inputs"])
        self.assertFalse(record["recommendation"]["execution_authorised"])
        labels = [element.proto.label for element in app.get("download_button")]
        self.assertIn("Download decision JSON", labels)
        self.assertIn("Download decision CSV", labels)

    def test_benign_prediction_cannot_suppress_high_impact_approval(self):
        app = self.app(); record = self.example(app, "N27")
        self.assertFalse(record["policy_inputs"]["predicted_attack"])
        self.assertTrue(record["recommendation"]["SOC_review"])
        self.assertTrue(record["recommendation"]["human_approval_required"])
        self.assertFalse(record["recommendation"]["execution_authorised"])

    def test_zero_features_retain_source_and_approval_obligations(self):
        app = self.app(); record = self.example(app, "N28")
        self.assertTrue(record["recommendation"]["request_source_evidence"])
        self.assertTrue(record["recommendation"]["SOC_review"])
        self.assertTrue(record["recommendation"]["human_approval_required"])
        self.assertFalse(record["recommendation"]["auto_collect_evidence"])

    def test_ready_read_only_collection_remains_a_recommendation(self):
        app = self.app(); record = self.example(app, "N12")
        self.assertTrue(record["recommendation"]["auto_collect_evidence"])
        self.assertFalse(record["recommendation"]["execution_authorised"])
        self.assertTrue(any("No response action has run" in message.value for message in app.info))

    def test_latest_run_requires_a_completed_prediction_then_uses_it(self):
        app = self.app()
        app.radio(key="rq2_source").set_value("Latest prediction run").run()
        self.assertTrue(any("Run a valid CSV" in message.value for message in app.info))
        app.radio(key="page").set_value("Predict a CSV").run()
        app.button(key="predict_run").click().run()
        expected_hash = app.session_state["prediction_run"]["input_hash"]
        app.radio(key="page").set_value("RQ2 decision support").run()
        app.radio(key="rq2_source").set_value("Latest prediction run").run()
        record = self.evaluate(app)
        self.assertEqual(record["evidence_origin"], "latest_prediction_run")
        self.assertEqual(record["provenance"]["input_sha256"], expected_hash)

    def test_changing_the_selected_record_hides_stale_exports(self):
        app = self.app(); self.evaluate(app)
        record_widget = next(widget for widget in app.selectbox if widget.label == "Record")
        record_widget.set_value(1).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(any("generate a matching decision record" in message.value for message in app.info))


if __name__ == "__main__": unittest.main(verbosity=2)
