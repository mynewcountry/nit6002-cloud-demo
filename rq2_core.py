"""Typed adapter and exports for the exploratory RQ2 policy demonstration.

The saved V4 policy stays unchanged. Dataset labels and identifiers never enter
its decision contract. Provenance is recorded separately from policy inputs.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping

import numpy as np

from rq2_policy_v4_hardened import Event, KNOWN_ACTIONS, decide

ROOT = Path(__file__).resolve().parent
POLICY_VERSION = "V4 exploratory, post-hoc repair"
APP_RELEASE = "rq2-cloud-view-2026-11-02.1"
SCHEMA_VERSION = "nit6002.rq2.decision.v1"
ORIGINS = {"included_model_sample", "latest_prediction_run", "illustrative_scenario"}
DETECTION_FIELDS = {"predicted_attack", "all_zero", "nonfinite_features", "model_attack_score"}
CONTEXT_FIELDS = {"critical_asset", "privileged_access", "secondary_signal", "additional_evidence"}
CONTEXT_EVIDENCE_FIELDS = set(Event.__dataclass_fields__) - {
    "predicted_attack", "all_zero", "nonfinite_features", "critical_asset",
    "privileged_access", "secondary_signal",
}
TRACE_FIELDS = {"input_sha256", "input_row", "device_name", "illustrative_case"}

REASON_TEXT = {
    "UNRELIABLE_MODEL_FEATURES": "Request original source evidence because model features are all zero or unusable.",
    "SIGNAL_PROVENANCE_FRESHNESS_OR_VERIFICATION_UNCONFIRMED": "Review the secondary signal because its verification, freshness or independence is unconfirmed.",
    "INDEPENDENT_SIGNAL_REQUIRES_SEPARATE_TRIAGE": "Investigate the secondary signal separately from unreliable model features.",
    "BENIGN_MODEL_DISAGREES_WITH_SECURITY_SIGNAL": "The model's benign prediction does not resolve the secondary security signal.",
    "NO_SPECIFIC_APPROVED_ACTION": "An attack prediction needs triage; no specific action has been proposed.",
    "UNKNOWN_ACTION_IMPACT": "Review is needed because the proposed action's impact is unknown.",
    "HIGH_IMPACT_REQUIRES_HUMAN_APPROVAL": "A high-impact proposal requires SOC review and separate human approval.",
    "MEDIUM_RISK_ROLLBACK_NOT_VERIFIED": "The proposed restriction lacks a verified rollback plan or an acceptable illustrative time limit.",
    "MEDIUM_RISK_SOC_REVIEW_REQUIRED": "SOC review remains required for the proposed temporary restriction.",
    "READ_ONLY_COLLECTION_RECOMMENDED": "Read-only evidence collection is recommended under the simulated conditions; it is not performed.",
    "READ_ONLY_AUTHORISATION_OR_EVIDENCE_UNCONFIRMED": "The evidence or authorisation conditions for read-only collection are incomplete.",
    "ACTION_BLOCKED_BY_UNRELIABLE_MODEL_EVIDENCE": "Unreliable model features prevent an action recommendation and require investigation.",
    "CRITICAL_ASSET_OR_PRIVILEGED_ACCESS": "Criticality or privileged access adds a SOC review obligation when a threat signal is present.",
    "PROPOSED_HIGH_IMPACT_ACTION_NEEDS_SEPARATE_APPROVAL": "A high-impact proposal still requires approval when another signal conflicts with a benign or unreliable model.",
    "MEDIUM_ACTION_PLAYBOOK_AUTHORITY_NOT_CONFIRMED": "Approval is required because the medium-impact proposal lacks confirmed playbook authority.",
    "NO_QUALIFYING_EVIDENCE": "These inputs trigger no follow-up obligation in this policy; this is not a finding that the event is safe.",
}


def _boolean(value, field):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a Boolean")
    return bool(value)


def prediction_evidence(row: Mapping) -> dict:
    """Select only observed prediction/quality fields from an inference row."""
    predicted = row.get("model_prediction")
    if predicted not in ("benign", "attack"):
        raise ValueError("The inference row must contain a recognised model prediction")
    score = row.get("model_attack_score")
    if not isinstance(score, Real) or isinstance(score, (bool, np.bool_)):
        raise ValueError("The inference row must contain a numeric model score")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The model score must be finite and between zero and one")
    return {
        "predicted_attack": predicted == "attack",
        "all_zero": _boolean(row.get("all_zero_predictors"), "all_zero_predictors"),
        "nonfinite_features": False,  # ml_core.infer rejects invalid feature rows.
        "model_attack_score": score,
    }


def build_event(evidence: dict, context: dict) -> dict:
    if type(evidence) is not dict or set(evidence) != DETECTION_FIELDS:
        raise ValueError("Detection evidence must contain exactly the recognised fields")
    if type(context) is not dict or set(context) != CONTEXT_FIELDS:
        raise ValueError("Context must contain exactly the recognised fields")
    extra = context["additional_evidence"]
    if type(extra) is not dict or set(extra) - CONTEXT_EVIDENCE_FIELDS:
        raise ValueError("Unrecognised context evidence; detection quality and labels cannot be overridden")
    score = evidence["model_attack_score"]
    if score is not None and (not isinstance(score, Real) or isinstance(score, bool)
                              or not math.isfinite(float(score)) or not 0 <= score <= 1):
        raise ValueError("A supplied model score must be finite and between zero and one")
    payload = {
        "predicted_attack": _boolean(evidence["predicted_attack"], "predicted_attack"),
        "all_zero": _boolean(evidence["all_zero"], "all_zero"),
        **{key: _boolean(context[key], key) for key in ("critical_asset", "privileged_access", "secondary_signal")},
        "additional_evidence": {
            **extra,
            "nonfinite_features": _boolean(evidence["nonfinite_features"], "nonfinite_features"),
        },
    }
    Event.from_scenario(payload)
    return payload


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def input_fingerprint(evidence: dict, context: dict, origin: str, trace: dict) -> str:
    payload = build_event(evidence, context)
    if origin not in ORIGINS:
        raise ValueError("Unrecognised evidence origin")
    if type(trace) is not dict or set(trace) - TRACE_FIELDS:
        raise ValueError("Unrecognised provenance fields")
    implementation = [APP_RELEASE, file_digest("rq2_policy_v4_hardened.py"),
                      file_digest("rq2_core.py"), file_digest("rq2_view.py")]
    return hashlib.sha256(_json([payload, evidence["model_attack_score"], origin, trace, implementation]).encode()).hexdigest()


def file_digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def evaluate_decision(evidence: dict, context: dict, origin: str, trace: dict,
                      evaluated_at: str | None = None) -> dict:
    fingerprint = input_fingerprint(evidence, context, origin, trace)
    payload = build_event(evidence, context)
    result = decide(Event.from_scenario(payload)).as_dict()
    if result["execution_authorised"] is not False:
        raise RuntimeError("The demonstration cannot authorise execution")
    unknown = set(result["reason_codes"]) - set(REASON_TEXT)
    if unknown:
        raise RuntimeError("The policy returned an undocumented reason code")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": fingerprint[:16],
        "input_fingerprint": fingerprint,
        "evaluated_at_utc": evaluated_at or datetime.now(timezone.utc).isoformat(),
        "app_release": APP_RELEASE,
        "policy_version": POLICY_VERSION,
        "source_identity": {
            "policy_sha256": file_digest("rq2_policy_v4_hardened.py"),
            "adapter_sha256": file_digest("rq2_core.py"),
            "view_sha256": file_digest("rq2_view.py"),
            "model_manifest_sha256": file_digest("model_manifest.json"),
        },
        "evidence_origin": origin,
        "context_origin": "simulated user inputs; not observed in the dataset or retrieved from security systems",
        "detector_evidence": evidence,
        "provenance": trace,
        "policy_inputs": payload,
        "recommendation": result,
        "reason_explanations": [REASON_TEXT[code] for code in result["reason_codes"]],
        "scope": "Decision-support demonstration only. No action is executed, no approval is granted, and natural labels remain unadjudicated. V4 repairs known cases and is not independently validated.",
    }


def record_json(record: dict) -> bytes:
    return json.dumps(record, indent=2, allow_nan=False).encode("utf-8")


def record_csv(record: dict) -> bytes:
    event = record["policy_inputs"]
    result = record["recommendation"]
    row = {
        "schema_version": record["schema_version"], "record_id": record["record_id"],
        "evaluated_at_utc": record["evaluated_at_utc"], "app_release": record["app_release"],
        "policy_version": record["policy_version"], "policy_sha256": record["source_identity"]["policy_sha256"],
        "evidence_origin": record["evidence_origin"],
        "model_prediction": "attack" if event["predicted_attack"] else "benign",
        "uncalibrated_attack_score": record["detector_evidence"]["model_attack_score"],
        "all_zero": event["all_zero"],
        "nonfinite_features": event["additional_evidence"]["nonfinite_features"],
        "critical_asset": event["critical_asset"], "privileged_access": event["privileged_access"],
        "secondary_signal": event["secondary_signal"],
        "requested_action": event["additional_evidence"].get("requested_action"),
        "action_impact": event["additional_evidence"].get("action_impact"),
        **{key: result[key] for key in ("request_source_evidence", "SOC_review", "human_approval_required", "auto_collect_evidence", "no_case", "execution_authorised")},
        "reason_codes_json": json.dumps(result["reason_codes"]),
        "policy_inputs_json": _json(event),
    }
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(row))
    writer.writeheader(); writer.writerow(row)
    return output.getvalue().encode("utf-8")
