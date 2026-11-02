"""Streamlit decision-support view for the unchanged exploratory V4 policy."""
from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import streamlit as st

from ml_core import ROOT, digest, infer, load_packaged_model, read_csv_bytes
from rq2_core import (
    APP_RELEASE, POLICY_VERSION, evaluate_decision, input_fingerprint,
    prediction_evidence, record_csv, record_json,
)
from rq2_policy_v4_hardened import KNOWN_ACTIONS, MAX_CONTAINMENT_MINUTES

ACTIONS = {
    None: "No action proposed",
    "read_only_log_collection": "Read-only log collection",
    "temporary_endpoint_isolation": "Temporary endpoint isolation",
    "block_production_subnet": "Block production subnet",
    "disable_privileged_identity": "Disable privileged identity",
}
CONDITIONS = {
    "signal_verified": "Signal verified",
    "signal_fresh": "Signal fresh",
    "signal_provenance_independent": "Signal source independent",
    "playbook_authorised": "Playbook authorised",
    "privacy_authorised": "Collection privacy conditions met",
    "access_audit_enabled": "Access auditing enabled",
    "rollback_plan": "Rollback plan available",
    "rollback_validated": "Rollback tested",
}


@st.cache_data
def included_predictions():
    raw = (ROOT / "demo/sample_inputs.csv").read_bytes()
    output, _ = infer(read_csv_bytes(raw), load_packaged_model())
    return output, digest(raw)


def _reset_editor(token, evidence, context):
    if st.session_state.get("rq2_editor_token") == token:
        return
    st.session_state["rq2_editor_token"] = token
    for key in ("critical_asset", "privileged_access", "secondary_signal"):
        st.session_state["rq2_" + key] = context[key]
    extra = context.get("additional_evidence", {})
    st.session_state["rq2_action"] = extra.get("requested_action")
    st.session_state["rq2_minutes"] = extra.get("time_limit_minutes") or 30
    for key in CONDITIONS:
        st.session_state["rq2_" + key] = extra.get(key)
    st.session_state["rq2_manual_prediction"] = "attack" if evidence["predicted_attack"] else "benign"
    st.session_state["rq2_manual_zero"] = evidence["all_zero"]
    st.session_state["rq2_manual_invalid"] = evidence["nonfinite_features"]


def _tri_state(key, label):
    return st.selectbox(label, [None, True, False],
                        format_func=lambda value: {None: "Unconfirmed", True: "Yes", False: "No"}[value],
                        key="rq2_" + key)


def render_rq2_view():
    st.title("RQ2 decision support")
    st.write("Explore how detection evidence and data quality change the investigation and approval obligations in the four-layer Adaptive Cyber Resilience Architecture.")
    st.info("Simulation only. This view recommends follow-up and records approval requirements. It does not collect logs, isolate devices, block networks, disable accounts or grant approval.")
    st.caption(f"{POLICY_VERSION} · {APP_RELEASE}. V4 repairs known cases; independent policy validation remains outstanding.")
    source = st.radio("Detection evidence", ["Included model sample", "Latest prediction run", "Illustrative scenario"],
                      key="rq2_source", horizontal=True)
    context = {"critical_asset": False, "privileged_access": False,
               "secondary_signal": False, "additional_evidence": {}}

    if source == "Illustrative scenario":
        data = json.loads((ROOT / "demo/rq2_examples.json").read_text())
        examples = {item["id"]: item for item in data["examples"]}
        ident = st.selectbox("Illustrative example", list(examples),
                             format_func=lambda key: f"{key}: {examples[key]['title']}", key="rq2_example")
        item = deepcopy(examples[ident]); evidence = item["evidence"]; context = item["context"]
        st.caption("These authored examples were available during V4 repair. Editing or replaying them is a demonstration, not an independent evaluation.")
        origin = "illustrative_scenario"; trace = {"illustrative_case": ident}
        token = f"example:{ident}"
    else:
        if source == "Latest prediction run":
            latest = st.session_state.get("prediction_run")
            if not latest:
                st.info("Run a valid CSV in Predict a CSV first, then return here. The included sample can also be used directly.")
                return
            output = latest["output"]
            input_hash = latest["input_hash"]
            origin = "latest_prediction_run"
        else:
            output, input_hash = included_predictions()
            origin = "included_model_sample"
        selected = st.selectbox("Record", list(range(len(output))),
                                format_func=lambda index: f"Row {int(output.iloc[index]['input_row'])}: {output.iloc[index]['model_prediction']}",
                                key="rq2_record_" + input_hash[:12])
        row = output.iloc[selected]
        evidence = prediction_evidence(row)
        trace = {"input_sha256": input_hash, "input_row": int(row["input_row"])}
        if "device_name" in row:
            trace["device_name"] = str(row["device_name"])
        token = f"{origin}:{input_hash}:{selected}"
        columns = st.columns(3)
        columns[0].metric("Recorded model prediction", row["model_prediction"])
        columns[1].metric("Uncalibrated attack score", f"{evidence['model_attack_score']:.4f}")
        columns[2].metric("All-zero model features", "Yes" if evidence["all_zero"] else "No")
        st.caption("Prediction and quality flags come from inference. Supplied labels and source identifiers are excluded from policy inputs. The score is displayed for traceability and does not change the policy's rules.")

    _reset_editor(token, evidence, context)
    st.subheader("Simulated operational context")
    st.caption("These assumptions are entered for this demonstration. They are not measured in DataSense or retrieved from a SIEM. Press Evaluate decision to apply changes.")
    with st.form("rq2_context_form"):
        if source == "Illustrative scenario":
            columns = st.columns(3)
            with columns[0]:
                prediction = st.selectbox("Hypothetical model prediction", ["benign", "attack"], key="rq2_manual_prediction")
            with columns[1]:
                zero = st.checkbox("All-zero model features", key="rq2_manual_zero")
            with columns[2]:
                invalid = st.checkbox("Unusable or nonfinite features", key="rq2_manual_invalid")
            evidence = {"predicted_attack": prediction == "attack", "all_zero": zero,
                        "nonfinite_features": invalid, "model_attack_score": None}
        columns = st.columns(3)
        with columns[0]:
            critical = st.checkbox("Critical asset", key="rq2_critical_asset")
        with columns[1]:
            privileged = st.checkbox("Privileged access", key="rq2_privileged_access")
        with columns[2]:
            signal = st.checkbox("Secondary security signal", key="rq2_secondary_signal")
        action = st.selectbox("Proposed action", list(ACTIONS), format_func=ACTIONS.get, key="rq2_action")
        st.caption("A proposal is not an authorised action. Known action impacts are fixed by the policy.")
        extra = {"requested_action": action, "action_impact": KNOWN_ACTIONS.get(action)}
        with st.expander("Evidence and authorisation assumptions", expanded=True):
            columns = st.columns(3)
            for index, (key, label) in enumerate(list(CONDITIONS.items())[:6]):
                with columns[index % 3]:
                    extra[key] = _tri_state(key, label)
        with st.expander("Rollback and time-limit assumptions"):
            columns = st.columns(3)
            with columns[0]:
                extra["rollback_plan"] = _tri_state("rollback_plan", CONDITIONS["rollback_plan"])
            with columns[1]:
                extra["rollback_validated"] = _tri_state("rollback_validated", CONDITIONS["rollback_validated"])
            with columns[2]:
                minutes = st.number_input("Proposed duration (minutes)", min_value=1, max_value=1440, step=1, key="rq2_minutes")
            st.caption(f"The {MAX_CONTAINMENT_MINUTES}-minute bound is illustrative, not an organisationally validated threshold.")
        extra["time_limit_minutes"] = int(minutes) if action in ("temporary_endpoint_isolation", "block_production_subnet", "disable_privileged_identity") else None
        context = {"critical_asset": critical, "privileged_access": privileged,
                   "secondary_signal": signal, "additional_evidence": extra}
        submitted = st.form_submit_button("Evaluate decision", type="primary")

    fingerprint = input_fingerprint(evidence, context, origin, trace)
    if submitted:
        try:
            st.session_state["rq2_decision_record"] = evaluate_decision(evidence, context, origin, trace)
        except (ValueError, RuntimeError) as exc:
            st.session_state.pop("rq2_decision_record", None)
            st.error(str(exc))
            return
    record = st.session_state.get("rq2_decision_record")
    if not record or record["input_fingerprint"] != fingerprint:
        st.info("Evaluate the selected record and context to generate a matching decision record.")
        return

    result = record["recommendation"]
    st.subheader("Last evaluated decision")
    st.caption(f"Record {record['record_id']} · {record['evaluated_at_utc']} · {record['policy_version']}")
    flags = [
        ("Source evidence", "request_source_evidence"), ("SOC review", "SOC_review"),
        ("Human approval", "human_approval_required"), ("Read-only collection", "auto_collect_evidence"),
    ]
    for col, (label, key) in zip(st.columns(4), flags):
        col.metric(label, ("Recommended" if key == "auto_collect_evidence" else "Required") if result[key] else "Not flagged")
    st.info("Execution authorised: False. Approval requirements are recorded, not granted. No response action has run.")
    if result["no_case"]:
        st.write("These inputs trigger no follow-up obligation in this policy. This does not establish that the event is benign or safe.")
    st.subheader("Reasons")
    for explanation in record["reason_explanations"]:
        st.write("• " + explanation)
    st.subheader("Proposed SIEM, SOC and SOAR handoff")
    st.dataframe(pd.DataFrame([
        ["SIEM", "Retain prediction, quality flags and provenance; correlate source evidence and secondary signals."],
        ["SOC", "Investigate this record." if result["SOC_review"] else "No SOC review flag under the current policy inputs."],
        ["SOAR", "Record the proposed playbook and obligations; wait for separately established approval and execution authority."],
    ], columns=["Process", "Demonstrated responsibility"]), hide_index=True, width="stretch")
    st.caption("These are proposed workflow handoffs. The app has no live SIEM/SOAR connector or response executor. Quality-driven follow-up does not change detector recall.")
    left, right = st.columns(2)
    with left:
        st.download_button("Download decision JSON", record_json(record), "rq2_decision.json", "application/json", key="rq2_json", on_click="ignore")
    with right:
        st.download_button("Download decision CSV", record_csv(record), "rq2_decision.csv", "text/csv", key="rq2_csv", on_click="ignore")
    with st.expander("Inspect evaluated inputs and policy identity"):
        st.json(record)
