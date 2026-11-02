"""Regression and strict-input validation for offline V4 response policy."""
import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rq2_policy_v4_hardened import Event, decide
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests'/'fixtures'/'predeclared_scenarios.json'
BASE=dict(predicted_attack=True,all_zero=False,critical_asset=False,privileged_access=False,secondary_signal=False)
class PolicyHardenedTests(unittest.TestCase):
    def test_all_34_frozen_cases(self):
        cases=json.loads(FIXTURE.read_text())['cases']
        self.assertEqual(len(cases),34)
        for row in cases:
            with self.subTest(id=row['id']):
                if row['expect_rejection']:
                    with self.assertRaises(ValueError): Event.from_scenario(row['input'])
                else:
                    outcome=decide(Event.from_scenario(row['input'])).as_dict()
                    for key in row['must']: self.assertTrue(outcome[key],(row['id'],key))
                    for key in row['forbid']: self.assertFalse(outcome[key],(row['id'],key))
                    self.assertFalse(outcome['execution_authorised'])
    def test_unknown_top_level_action_fails_closed(self):
        with self.assertRaises(ValueError): Event.from_scenario({**BASE,'requested_action':'block_production_subnet'})
    def test_bad_evidence_type_fails_closed(self):
        for invalid in ([],0,'',None,False):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):Event.from_scenario({**BASE,'additional_evidence':invalid})
    def test_unknown_and_label_fields_rejected(self):
        for bad in ({**BASE,'source_id':'secret'},{**BASE,'additional_evidence':{'known_injected_flip':True}},{**BASE,'additional_evidence':{'playbok_authorised':True}}):
            with self.subTest(bad=bad),self.assertRaises(ValueError): Event.from_scenario(bad)
    def test_high_impact_requires_approval_even_with_bad_features(self):
        for attack,signal,zero in [(True,False,False),(True,False,True),(False,True,False),(True,True,True)]:
            dec=decide(Event.from_scenario({**BASE,'predicted_attack':attack,'secondary_signal':signal,'all_zero':zero,'additional_evidence':{'requested_action':'disable_privileged_identity'}}))
            self.assertTrue(dec.human_approval_required)
            self.assertTrue(dec.SOC_review)
            self.assertFalse(dec.execution_authorised)
    def test_low_risk_auto_collection_is_recommendation_only(self):
        x={**BASE,'secondary_signal':True,'additional_evidence':{'requested_action':'read_only_log_collection','signal_verified':True,'signal_fresh':True,'signal_provenance_independent':True,'playbook_authorised':True,'privacy_authorised':True,'access_audit_enabled':True}}
        y=decide(Event.from_scenario(x))
        self.assertTrue(y.auto_collect_evidence)
        self.assertFalse(y.execution_authorised)
        x['all_zero']=True
        z=decide(Event.from_scenario(x))
        self.assertFalse(z.auto_collect_evidence)
        self.assertTrue(z.request_source_evidence)
if __name__=='__main__':unittest.main(verbosity=2)
