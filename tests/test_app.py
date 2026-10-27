from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_core import ROOT
from streamlit.testing.v1 import AppTest


class AppChecks(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    def test_navigation_and_real_sample_inference(self):
        app = self.app()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(app.title[0].value, "Predict a CSV")
        self.assertEqual(len(app.exception), 0)
        app.button(key="predict_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe[-1].value), 90)
        app.checkbox(key="flagged_only").check().run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.dataframe[-1].value), 0)

    def test_review_recorded_results_and_evidence_pages(self):
        app = self.app()
        for page in ["Review experiment", "Recorded evaluation", "Deployment evidence"]:
            app.radio(key="page").set_value(page).run()
            self.assertEqual(len(app.exception), 0, page)
        app.radio(key="page").set_value("Review experiment").run()
        app.select_slider(key="review_budget").set_value(25).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "42,687")
        self.assertEqual(app.metric[1].value, "16,298")


if __name__ == "__main__":
    unittest.main(verbosity=2)
