"""Validation and inference for the recorded NIT6002 DataSense model."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 20_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def features() -> list[str]:
    schema = pd.read_csv(ROOT / "results" / "feature_schema.csv")
    return schema.sort_values("feature_index")["feature"].tolist()


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def load_packaged_model():
    """Load only the fixed, checksummed model delivered with this project."""
    manifest = json.loads((ROOT / "model_manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"Packaged asset changed: {relative}. Restore a coherent project copy.")
    if sklearn.__version__ != manifest["training_versions"]["scikit_learn"]:
        raise RuntimeError("The model requires scikit-learn 1.8.0. Install requirements.txt.")
    model = joblib.load(ROOT / "results" / "split1_baseline_forest.joblib")
    if model.n_features_in_ != len(features()) or list(model.classes_) != [0, 1]:
        raise ValueError("Model classes or feature count do not match the packaged schema.")
    return model


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_BYTES:
        raise ValueError("Use a CSV smaller than 20 MiB for this demonstration.")
    try:
        text = data.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
    except (UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError("Select a non-empty UTF-8 CSV with a header row.") from exc
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not accepted.")
    if any(not h.strip() for h in header):
        raise ValueError("Every CSV column needs a name.")
    try:
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise ValueError("The file could not be read as a comma-separated table.") from exc
    if frame.empty:
        raise ValueError("The CSV needs at least one data row.")
    if len(frame) > MAX_ROWS:
        raise ValueError("Use at most 20,000 rows per demonstration upload.")
    return frame


def prepare_inputs(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    names = features()
    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    # Exact feature selection also prevents metadata and labels entering inference.
    numeric = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(frame[names].isna().to_numpy().sum())
    non_numeric = int((numeric.isna() & frame[names].notna()).to_numpy().sum())
    values64 = numeric.to_numpy(dtype=np.float64)
    infinite = int(np.isinf(values64).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = values64.astype(np.float32)
    invalid = ~np.isfinite(matrix).all(axis=1)
    all_zero = np.all(matrix == 0, axis=1) & ~invalid
    audit = {
        "rows": len(frame),
        "model_features": len(names),
        "missing_feature_cells": missing_cells,
        "non_numeric_feature_cells": non_numeric,
        "infinite_feature_cells": infinite,
        "invalid_rows_after_float32_conversion": int(invalid.sum()),
        "all_zero_predictor_rows": int(all_zero.sum()),
        "all_zero_predictor_pct": float(100 * all_zero.mean()),
        "excluded_columns": [c for c in frame.columns if c not in names],
        "original_label_accuracy": "Not independently verified",
    }
    return matrix, audit


def reference_labels(frame: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    if "label1" not in frame:
        return None, "No label1 column was supplied; label-agreement metrics are unavailable."
    labels = frame["label1"].astype(str).str.strip().str.lower().map(
        {"0": 0, "0.0": 0, "benign": 0, "1": 1, "1.0": 1, "attack": 1}
    )
    if labels.isna().any():
        return None, "Some label1 values are missing or unrecognised. No rows were discarded; metrics are unavailable."
    return labels.to_numpy(dtype=np.int8), None


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))

    def ratio(num, den):
        return float(100 * num / den) if den else None

    return {
        "rows": len(truth), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy_pct": ratio(tn + tp, len(truth)),
        "attack_precision_pct": ratio(tp, tp + fp),
        "attack_recall_pct": ratio(tp, tp + fn),
        "attack_f1_pct": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate_pct": ratio(fp, fp + tn),
    }


def infer(frame: pd.DataFrame, model) -> tuple[pd.DataFrame, dict]:
    matrix, audit = prepare_inputs(frame)
    if audit["invalid_rows_after_float32_conversion"]:
        raise ValueError("Prediction blocked: one or more feature rows are missing, non-numeric, infinite or outside float32 range.")
    begin = perf_counter()
    prediction = model.predict(matrix).astype(np.int8)
    score = model.predict_proba(matrix)[:, list(model.classes_).index(1)]
    elapsed = (perf_counter() - begin) * 1000
    zero = np.all(matrix == 0, axis=1)
    out = pd.DataFrame({"input_row": np.arange(1, len(frame) + 1)})
    for column in ["source_id", "device_name"]:
        if column in frame:
            out[column] = frame[column].to_numpy()
    out["model_prediction"] = np.where(prediction == 1, "attack", "benign")
    out["model_attack_score"] = score
    out["all_zero_predictors"] = zero
    truth, label_message = reference_labels(frame)
    if truth is not None:
        out["supplied_label"] = np.where(truth == 1, "attack", "benign")
        out["prediction_disagrees_with_label"] = prediction != truth
    out["review_flag"] = np.where(zero, "All-zero predictors: source context needed", "")
    if truth is not None:
        disagree = (prediction != truth) & ~zero
        out.loc[disagree, "review_flag"] = "Model and supplied label disagree"
    run_manifest = json.loads((ROOT / "results" / "run_manifest.json").read_text())
    training_devices = set(run_manifest["group_membership"][0]["training_devices"])
    if "device_name" in frame:
        known = frame["device_name"].astype(str).isin(training_devices)
        audit["rows_with_device_name_used_in_model_training"] = int(known.sum())
    else:
        audit["rows_with_device_name_used_in_model_training"] = None
    audit["prediction_ms_excluding_parsing_loading_and_rendering"] = elapsed
    audit["supplied_label_metrics"] = binary_metrics(truth, prediction) if truth is not None else None
    audit["label_message"] = label_message
    return out, audit


def replay_review(method: str, budget_pct: float, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Keep injected-flip truth outside every selector's inputs.
    from analysis.run_analysis import orders_for_review

    selectors = orders_for_review(
        candidates["source_id"].to_numpy(),
        candidates["all_zero"].to_numpy(dtype=bool),
        candidates["disagreement"].to_numpy(dtype=bool),
        candidates["uncertainty_heuristic"].to_numpy(dtype=float),
    )
    if method not in selectors or budget_pct not in (1, 5, 10, 25):
        raise ValueError("Select one recorded method and a budget of 1, 5, 10 or 25 percent.")
    size = math.ceil(float(budget_pct) / 100 * len(candidates))
    selected = candidates.iloc[selectors[method](size)].copy()
    # Only after selection is complete may evaluation inspect the injected mask.
    recovered = int(selected["known_injected_flip"].sum())
    total_flips = int(candidates["known_injected_flip"].sum())
    stats = {
        "method": method, "budget_pct": float(budget_pct), "reviewed_rows": size,
        "recovered_injected_flips": recovered,
        "injected_flip_recovery_pct": 100 * recovered / total_flips,
        "injected_flip_yield_pct": 100 * recovered / size,
    }
    display_columns = ["source_id", "device", "perturbed_label", "all_zero", "disagreement", "out_of_group_attack_score", "uncertainty_heuristic"]
    return selected.loc[:, display_columns], stats
