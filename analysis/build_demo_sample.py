"""Replay the fixed model on its original held-out records and build a small demo CSV."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile

import joblib
import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    audit = json.loads((results / "source_audit.json").read_text())
    recorded_manifest = json.loads((results / "run_manifest.json").read_text())
    observed_sha = sha(args.archive)
    if observed_sha != audit["archive_sha256"]:
        raise ValueError("The source archive digest differs from the recorded run.")
    names = pd.read_csv(results / "feature_schema.csv")["feature"].tolist()
    recorded = pd.read_csv(results / "split1_test_predictions.csv.gz").set_index("source_id")
    matrices, sources, labels, devices = [], [], [], []
    observed_rows = 0
    observed_missing = 0
    observed_zero = 0
    observed_categories = Counter()
    with tarfile.open(args.archive, "r:xz") as outer:
        for member in audit["selected_members"]:
            raw = outer.extractfile(member["member"]).read()
            if hashlib.sha256(raw).hexdigest() != member["compressed_member_sha256"]:
                raise ValueError("A selected archive member changed.")
            offset = 0
            with tarfile.open(fileobj=BytesIO(raw), mode="r:xz") as inner:
                stream = inner.extractfile(member["csv"])
                for chunk in pd.read_csv(stream, chunksize=5000, low_memory=False):
                    ids = np.asarray([
                        f"{member['member']}::{member['csv']}::row{k:07d}"
                        for k in range(offset + 1, offset + len(chunk) + 1)
                    ])
                    X = chunk[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
                    if not np.isfinite(X).all():
                        raise ValueError("The source contains an invalid model input.")
                    observed_rows += len(chunk)
                    observed_missing += int(chunk.isna().to_numpy().sum())
                    observed_zero += int(np.all(X == 0, axis=1).sum())
                    observed_categories.update(chunk["label2"].astype(str).str.lower().value_counts().to_dict())
                    take = pd.Index(ids).isin(recorded.index)
                    if take.any():
                        y = chunk.loc[take, "label1"].str.lower().map({"benign": 0, "attack": 1}).to_numpy(dtype=np.int8)
                        matrices.append(X[take])
                        sources.extend(ids[take])
                        labels.extend(y)
                        devices.extend(chunk.loc[take, "device_name"].astype(str))
                    offset += len(chunk)
            print(f"Checked source member: {member['member']} ({offset:,} rows)", flush=True)
    if observed_rows != audit["rows"] or observed_missing != audit["missing_cells"] or observed_zero != audit["all_zero_rows"]:
        raise ValueError("Source counts do not match the audit.")
    X = np.vstack(matrices)
    y = np.asarray(labels, dtype=np.int8)
    sources = np.asarray(sources)
    devices = np.asarray(devices)
    if len(sources) != len(recorded) or len(set(sources)) != len(recorded):
        raise ValueError("Original held-out source identifiers did not reconcile.")
    expected = recorded.loc[sources]
    if not np.array_equal(y, expected["supplied_label"].to_numpy()):
        raise ValueError("Supplied labels changed.")
    model_path = results / "split1_baseline_forest.joblib"
    model = joblib.load(model_path)
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    mismatch = int(np.sum(prediction != expected["original_label_prediction"].to_numpy()))
    if mismatch:
        raise ValueError(f"Recorded-model replay differs on {mismatch} observations.")
    zero = np.all(X == 0, axis=1)
    buckets = Counter()
    selected = []
    for i, (device, label, pred, is_zero) in enumerate(zip(devices, y, prediction, zero)):
        key = (str(device), int(label), int(pred), bool(is_zero))
        if buckets[key] < 2:
            selected.append(i)
            buckets[key] += 1
    sample = pd.DataFrame(X[selected], columns=names)
    sample["label1"] = np.where(y[selected] == 1, "attack", "benign")
    sample["device_name"] = devices[selected]
    sample["source_id"] = sources[selected]
    demo = root / "demo"
    demo.mkdir(exist_ok=True)
    sample.to_csv(demo / "sample_inputs.csv", index=False, float_format="%.9g")
    pd.DataFrame({
        "source_id": sources[selected], "supplied_label": y[selected],
        "expected_prediction": prediction[selected], "expected_attack_score": probability[selected],
        "all_zero": zero[selected],
    }).to_csv(demo / "expected_predictions.csv", index=False)
    source_details = {
        "source": "DataSense 2025 one-second attack and benign files",
        "source_url": "https://www.unb.ca/cic/datasets/iiot-dataset-2025.html",
        "source_archive_sha256": observed_sha,
        "sample_rows": len(sample),
        "sampling": "First two held-out records per device, supplied class, model prediction and all-zero status; descriptive demonstration sample, not a random performance estimate.",
        "test_split": 1,
        "all_sample_devices_excluded_from_model_training": not bool(set(devices[selected]) & set(recorded_manifest["group_membership"][0]["training_devices"])),
        "independent_original_label_verification": False,
    }
    (demo / "provenance.json").write_text(json.dumps(source_details, indent=2) + "\n")
    manifest = {
        "model_description": "Recorded Split 1 original-label Random Forest",
        "feature_count": len(names),
        "training_versions": {k: recorded_manifest[k] for k in ["python", "numpy", "pandas", "scikit_learn", "joblib"]},
        "files_sha256": {str(path.relative_to(root)): sha(path) for path in [model_path, results / "feature_schema.csv"]},
        "source_archive_sha256": observed_sha,
    }
    (root / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verification = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Inference replay of the recovered saved model, not a fresh model-training run or independent label validation",
        "original_archive_digest_matches": True,
        "source_rows_checked": observed_rows,
        "source_all_zero_rows": observed_zero,
        "source_missing_cells": observed_missing,
        "source_category_counts": dict(observed_categories),
        "held_out_predictions_checked": len(sources),
        "held_out_prediction_mismatches": mismatch,
        "sample_rows": len(sample),
        "source_archive_sha256": observed_sha,
    }
    (root / "verification").mkdir(exist_ok=True)
    (root / "verification" / "inference_replay.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
