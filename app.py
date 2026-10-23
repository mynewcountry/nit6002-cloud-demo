from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ml_core import ROOT, digest, features, infer, load_packaged_model, prepare_inputs, read_csv_bytes, replay_review, runtime_versions

st.set_page_config(page_title="NIT6002 Cybersecurity Data Review", layout="wide")


@st.cache_resource
def model():
    return load_packaged_model()


@st.cache_data
def recorded_csv(name):
    return pd.read_csv(ROOT / "results" / name)


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def pct(value):
    return "n/a" if value is None else f"{value:.2f}%"


def show_metrics(metrics):
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("Accuracy", "accuracy_pct"), ("Attack recall", "attack_recall_pct"),
        ("Attack F1", "attack_f1_pct"), ("False-positive rate", "false_positive_rate_pct")
    ]):
        col.metric(label, pct(metrics[key]))


def open_predictions():
    st.session_state["page"] = "Predict a CSV"


st.sidebar.title("NIT6002")
st.sidebar.caption("Cybersecurity data review")
page = st.sidebar.radio("Workspace", ["Overview", "Predict a CSV", "Review experiment", "Recorded evaluation", "Deployment evidence"], key="page")
st.sidebar.divider()
st.sidebar.caption("Recorded Random Forest model\n\nDataSense one-second representation\n\n71 numeric predictors")
st.sidebar.caption("Research prototype. Original dataset labels have not been independently verified.")

if page == "Overview":
    st.title("Cybersecurity data review")
    st.write("Inspect the data behind a prediction and explore how limited review budgets affect the recorded experiment.")
    source = read_json("results/source_audit.json")
    repeated = recorded_csv("repeated_baseline_results.csv")
    cols = st.columns(4)
    cols[0].metric("Benchmark records", f"{source['rows']:,}")
    cols[1].metric("Mean accuracy", pct(repeated.accuracy_pct.mean()))
    cols[2].metric("Mean attack recall", pct(repeated.attack_recall_pct.mean()))
    cols[3].metric("Mean attack F1", pct(repeated.attack_f1_pct.mean()))
    st.caption("Means are unweighted across the five recorded device holdouts. Test sets can overlap across runs.")
    left, right = st.columns(2)
    with left:
        st.subheader("Run a prediction")
        st.write("Use the included held-out sample or upload a compatible CSV. The app checks the 71 model fields before inference.")
        st.button("Open prediction workspace", type="primary", on_click=open_predictions)
    with right:
        st.subheader("Replay the review experiment")
        st.write("Choose one of four recorded review rules and a budget. The app rebuilds the queue from saved out-of-group scores.")
        st.caption("Known synthetic label changes are used for evaluation after selection. No original labels are automatically corrected.")
    st.subheader("Benchmark audit")
    rows = [
        ("Selected-member checksums", "Matched in the recorded audit"),
        ("Missing cells", str(source["missing_cells"])),
        ("All-zero predictor rows", f"{source['all_zero_rows']:,} ({100 * source['all_zero_rows'] / source['rows']:.2f}%)"),
        ("Device names", str(source["devices"])),
        ("Original label accuracy", "Not independently verified"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Measure", "Recorded result"]), hide_index=True, width="stretch")
    st.info("The prototype supports the dataset-verification and label-review questions. It is an offline research model served through a web interface; live incident response is outside this demonstration.")

elif page == "Predict a CSV":
    st.title("Predict a CSV")
    st.write("Run the fixed Split 1 model on processed DataSense observations. Raw packet captures and unrelated CSV schemas need a separate feature pipeline.")
    source_choice = st.radio("Input", ["Included demonstration sample", "Upload a CSV"], horizontal=True, key="input_choice")
    if source_choice == "Included demonstration sample":
        data = (ROOT / "demo/sample_inputs.csv").read_bytes()
        name = "sample_inputs.csv"
        st.download_button("Download this sample", data, name, "text/csv")
        st.caption("This purposive sample contains held-out devices, errors and all-zero cases. Its metrics describe only this sample, not the full benchmark.")
    else:
        upload = st.file_uploader("CSV with all 71 numeric model fields", type=["csv"], key="csv_upload")
        st.caption("Use at most 20 MiB and 20,000 rows. Optional columns: label1, device_name and source_id. Labels and identifiers are excluded from inference.")
        if upload is None:
            st.info("Choose a CSV or switch to the included sample.")
            st.stop()
        data, name = upload.getvalue(), upload.name
    input_hash = digest(data)
    try:
        frame = read_csv_bytes(data)
        _, quality = prepare_inputs(frame)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    cols = st.columns(4)
    cols[0].metric("Input rows", f"{len(frame):,}")
    cols[1].metric("Model fields", quality["model_features"])
    cols[2].metric("Invalid feature rows", quality["invalid_rows_after_float32_conversion"])
    cols[3].metric("All-zero rows", quality["all_zero_predictor_rows"])
    if quality["all_zero_predictor_rows"]:
        st.warning("All-zero predictors are retained and flagged for source review. Identical inputs cannot explain different supplied labels.")
    if quality["invalid_rows_after_float32_conversion"]:
        st.error("Prediction is blocked until missing, non-numeric, infinite or out-of-range feature values are resolved. The app does not silently impute them.")
        st.json(quality)
        st.stop()
    if st.button("Run model", type="primary", key="predict_run"):
        try:
            with st.spinner("Running the recorded model"):
                output, run_audit = infer(frame, model())
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
            st.stop()
        evidence = {
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_name": name, "input_sha256": input_hash,
            "runtime": runtime_versions(), "model_manifest": read_json("model_manifest.json"),
            "input_audit": run_audit,
            "scope": "CSV inference using the packaged fixed model; not a fresh training run or independent label adjudication",
        }
        st.session_state["prediction_run"] = {"input_hash": input_hash, "output": output, "evidence": evidence}
    stored = st.session_state.get("prediction_run")
    if stored and stored["input_hash"] == input_hash:
        output, evidence = stored["output"], stored["evidence"]
        run_audit = evidence["input_audit"]
        st.subheader("Model output")
        st.caption(f"Model inference: {run_audit['prediction_ms_excluding_parsing_loading_and_rendering']:.2f} ms. This excludes CSV parsing, model loading and page rendering.")
        overlap = run_audit.get("rows_with_device_name_used_in_model_training")
        if overlap:
            st.warning(f"{overlap:,} rows name devices present in model training. These uploaded-batch metrics do not establish performance on unseen devices.")
        elif overlap is None:
            st.caption("Device training overlap cannot be checked because device_name was not supplied.")
        if run_audit["supplied_label_metrics"]:
            st.write("Agreement with the supplied label1 values, across every input row:")
            show_metrics(run_audit["supplied_label_metrics"])
            m = run_audit["supplied_label_metrics"]
            st.dataframe(pd.DataFrame([[m['TN'], m['FP']], [m['FN'], m['TP']]], index=["Supplied benign", "Supplied attack"], columns=["Predicted benign", "Predicted attack"]), width="stretch")
        else:
            st.info(run_audit["label_message"])
        st.caption("Model attack scores are uncalibrated ranking scores. Neither a score nor a disagreement proves that an original label is wrong.")
        flagged = st.checkbox("Show only records flagged for review", key="flagged_only")
        visible = output[output.review_flag.ne("")] if flagged else output
        st.dataframe(visible.head(300), hide_index=True, width="stretch")
        st.caption(f"Showing up to 300 of {len(visible):,} selected rows. The metrics above always use the full input batch.")
        st.download_button("Download all predictions", output.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        st.download_button("Download this run record", json.dumps(evidence, indent=2).encode(), "inference_run_record.json", "application/json")

elif page == "Review experiment":
    st.title("Review experiment")
    st.write("Rebuild a review queue for the recorded Split 1 training set, using its saved out-of-group scores and fixed selection rules.")
    st.info("This controlled pilot deliberately changed 17,075 training labels. Recovery is measured against those known changes. It does not estimate the accuracy of the original annotations.")
    method = st.selectbox("Review method", ["Random", "Disagreement first", "Zero first", "Stratified review"], index=3, key="review_method")
    budget = st.select_slider("Training records reviewed (%)", options=[1, 5, 10, 25], value=5, key="review_budget")
    candidates = recorded_csv("training_review_candidates.csv.gz")
    queue, stats = replay_review(method, budget, candidates)
    cols = st.columns(4)
    cols[0].metric("Queue rows", f"{stats['reviewed_rows']:,}")
    cols[1].metric("Known flips found", f"{stats['recovered_injected_flips']:,}")
    cols[2].metric("Known-flip recovery", pct(stats["injected_flip_recovery_pct"]))
    cols[3].metric("Known-flip yield", pct(stats["injected_flip_yield_pct"]))
    comparison = recorded_csv("review_budget_results.csv")
    chart = comparison.pivot(index="budget_pct", columns="method", values="injected_flip_recovery_pct")
    chart.index.name = "Training records reviewed (%)"
    st.line_chart(chart, x_label="Training records reviewed (%)", y_label="Known flips recovered (%)")
    st.caption("The injected-flip mask is evaluated only after queue selection. The downloadable queue excludes that hidden evaluation mask and original unperturbed labels.")
    st.dataframe(queue.head(200), hide_index=True, width="stretch")
    st.download_button("Download selected review queue", queue.to_csv(index=False).encode(), "selected_review_queue.csv", "text/csv")
    st.subheader("Interpretation")
    st.write("Disagreement-first recovered more known changes at 1% and 5%. Stratified review recovered more at 10% and 25%. The current stratified and disagreement rules also use different tie ranking, so a controlled ablation is still needed.")
    st.caption("Changing the budget here replays selection from fixed recorded scores. It does not retrain the detector or execute a new corruption experiment.")

elif page == "Recorded evaluation":
    st.title("Recorded evaluation")
    st.write("Results from the fixed Random Forest across five recorded DataSense device holdouts. Source identity, model settings and software versions are retained in the deployment evidence.")
    repeated = recorded_csv("repeated_baseline_results.csv")
    show_metrics(repeated[["accuracy_pct", "attack_recall_pct", "attack_f1_pct", "false_positive_rate_pct"]].mean().to_dict())
    display = repeated.rename(columns={
        "split": "Split", "test_rows": "Test records", "accuracy_pct": "Accuracy (%)",
        "attack_recall_pct": "Attack recall (%)", "attack_f1_pct": "Attack F1 (%)",
        "false_positive_rate_pct": "False-positive rate (%)"
    })
    st.dataframe(display[["Split", "Test records", "Accuracy (%)", "Attack recall (%)", "Attack F1 (%)", "False-positive rate (%)"]].round(2), hide_index=True, width="stretch")
    st.line_chart(display.set_index("Split")[["Accuracy (%)", "Attack recall (%)"]])
    st.caption("Five repeated device holdouts, not five-fold cross-validation. Their test sets can overlap within one collection.")
    st.subheader("Detection after the fixed 5% restoration budget")
    post = recorded_csv("post_review_detection.csv")
    st.dataframe(post[["model", "accuracy_pct", "attack_recall_pct", "attack_f1_pct", "false_positive_rate_pct"]].round(4), hide_index=True, width="stretch")
    st.write("The recorded changes in detector performance are small. These results do not establish a dependable detection improvement after review.")
    st.download_button("Download the five-split results", repeated.to_csv(index=False).encode(), "repeated_baseline_results.csv", "text/csv")

else:
    st.title("Deployment evidence")
    st.write("Keep the research run, prototype checks and later cloud demonstration identifiable as separate records.")
    manifest = read_json("model_manifest.json")
    replay = read_json("verification/inference_replay.json")
    st.subheader("Original-record inference replay")
    st.metric("Held-out predictions checked", f"{replay['held_out_predictions_checked']:,}")
    st.metric("Differences from saved predictions", replay["held_out_prediction_mismatches"])
    st.caption("This replay loaded the saved model and original held-out inputs. It did not retrain the model or independently verify natural labels.")
    st.subheader("Current execution environment")
    st.json(runtime_versions())
    st.subheader("Model and source identity")
    st.json(manifest)
    deployment_url = st.text_input("URL you are demonstrating", placeholder="Paste the actual deployed app URL after deployment", key="deployment_url")
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_url_entered_by_user": deployment_url,
        "external_availability_automatically_verified": False,
        "current_runtime": runtime_versions(), "model_manifest": manifest,
        "original_record_inference_replay": replay,
    }
    st.download_button("Download deployment record", json.dumps(record, indent=2).encode(), "deployment_record.json", "application/json")
    st.download_button("Download cloud test sheet", (ROOT / "verification/cloud_test_record.csv").read_bytes(), "cloud_test_record.csv", "text/csv")
    st.write("After deploying, run the sample in the cloud, export its run record and compare predictions with demo/expected_predictions.csv. Test missing fields and non-finite values, then record actual outcomes and the URL in the cloud test sheet.")
    st.caption("A cloud URL by itself does not establish model quality. The app serves the recorded prototype; published-method comparisons, repeated-noise experiments and ablation remain separate research tasks.")
