"""Reproduce the DataSense audit and a controlled review-priority pilot.

The supplied labels are the reference annotations, not adjudicated truth.
Only known, deliberately injected training-label flips are restored in the
synthetic review experiment. Test labels never enter queue construction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tarfile
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

MODEL = dict(n_estimators=100, max_depth=15, min_samples_leaf=5,
             class_weight="balanced_subsample", random_state=42, n_jobs=2)
NOISE_RATE = 0.10
NOISE_SEED = 2026
RANDOM_QUEUE_SEED = 4302
BUDGETS = (0.01, 0.05, 0.10, 0.25)
UPDATE_BUDGET = 0.05
MEMBERS = ("attack_data/attack_samples_1sec.csv.tar.xz",
           "benign_data/benign_samples_1sec.csv.tar.xz")


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy_pct": accuracy_score(y, pred) * 100,
            "balanced_accuracy_pct": balanced_accuracy_score(y, pred) * 100,
            "attack_precision_pct": precision_score(y, pred, zero_division=0) * 100,
            "attack_recall_pct": recall_score(y, pred, zero_division=0) * 100,
            "attack_f1_pct": f1_score(y, pred, zero_division=0) * 100,
            "false_positive_rate_pct": fp / (fp + tn) * 100,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def fit_predict(Xtr, ytr, Xte):
    begin = time.perf_counter()
    model = RandomForestClassifier(**MODEL).fit(Xtr, ytr)
    prediction = model.predict(Xte)
    return model, prediction, time.perf_counter() - begin


def load_data(archive, out):
    with archive.open("rb") as f:
        archive_hash = hashlib.file_digest(f, "sha256").hexdigest()
    audit = {"archive_name": archive.name, "archive_size_bytes": archive.stat().st_size,
             "archive_sha256": archive_hash, "selected_members": [],
             "missing_cells": 0, "infinite_numeric_cells": 0,
             "numeric_conversion_errors": 0, "repeated_full_record_hashes": 0,
             "invalid_timestamp_starts": 0}
    parts, labels, device_parts, source_ids, timestamps, categories = [], [], [], [], [], []
    whole_hashes = set()
    features, columns = None, None
    with tarfile.open(archive, "r:xz") as outer:
        inventory = outer.getmembers()
        audit["archive_member_inventory"] = [{"name": m.name, "size_bytes": m.size}
                                              for m in inventory if m.isfile()]
        for member_name in MEMBERS:
            raw = outer.extractfile(member_name).read()
            member_hash = hashlib.sha256(raw).hexdigest()
            parent, basename = member_name.rsplit("/", 1)
            checksum_member = f"{parent}/checksums/{basename}.sha256"
            checksum_text = outer.extractfile(checksum_member).read().decode("utf-8")
            expected_hashes = re.findall(r"\b[0-9a-fA-F]{64}\b", checksum_text)
            checksum_match = member_hash in [s.lower() for s in expected_hashes]
            if not checksum_match:
                raise ValueError(f"Packaged checksum does not match {member_name}")
            row_count = 0
            local_zero = 0
            first, last = None, None
            with tarfile.open(fileobj=BytesIO(raw), mode="r:xz") as inner:
                csvs = [m for m in inner.getmembers() if m.isfile() and m.name.endswith(".csv")]
                if len(csvs) != 1:
                    raise ValueError("Expected exactly one CSV in each selected member")
                with inner.extractfile(csvs[0]) as csv_file:
                    for chunk in pd.read_csv(csv_file, chunksize=5000, low_memory=False):
                        if columns is None:
                            columns = list(chunk.columns)
                            features = list(chunk.select_dtypes(include="number").columns)
                        elif list(chunk.columns) != columns:
                            raise ValueError("Inconsistent input schema")
                        numeric = chunk[features].apply(pd.to_numeric, errors="coerce")
                        audit["missing_cells"] += int(chunk.isna().to_numpy().sum())
                        audit["numeric_conversion_errors"] += int((numeric.isna() & chunk[features].notna()).to_numpy().sum())
                        audit["infinite_numeric_cells"] += int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum())
                        Xchunk = numeric.to_numpy(dtype=np.float32)
                        if not np.isfinite(Xchunk).all():
                            raise ValueError("Non-finite model input")
                        ychunk = chunk.label1.astype(str).str.lower().str.strip().map({"benign": 0, "attack": 1})
                        if ychunk.isna().any():
                            raise ValueError("Unknown target label")
                        for hashed in pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64):
                            key = int(hashed)
                            audit["repeated_full_record_hashes"] += int(key in whole_hashes)
                            whole_hashes.add(key)
                        dt = pd.to_datetime(chunk.timestamp_start, utc=True, errors="coerce")
                        audit["invalid_timestamp_starts"] += int(dt.isna().sum())
                        if dt.notna().any():
                            lo, hi = dt.min(), dt.max()
                            first = lo if first is None or lo < first else first
                            last = hi if last is None or hi > last else last
                        n = len(chunk)
                        parts.append(Xchunk)
                        labels.append(ychunk.to_numpy(dtype=np.int8))
                        device_parts.append(chunk.device_name.fillna("unknown").astype(str).to_numpy())
                        source_ids.extend(f"{member_name}::{csvs[0].name}::row{k:07d}"
                                          for k in range(row_count + 1, row_count + n + 1))
                        timestamps.extend(chunk.timestamp_start.astype(str))
                        categories.extend(chunk.label2.astype(str).str.lower().str.strip())
                        local_zero += int(np.all(Xchunk == 0, axis=1).sum())
                        row_count += n
            audit["selected_members"].append({"member": member_name, "csv": csvs[0].name,
                 "compressed_member_sha256": member_hash, "packaged_checksum_matches": checksum_match,
                 "rows": row_count, "all_zero_rows": local_zero,
                 "first_timestamp_start_utc": first.isoformat(),
                 "last_timestamp_start_utc": last.isoformat()})
            print(f"Loaded and checksum-verified {member_name}: {row_count:,} rows", flush=True)
    X, y, devices = np.vstack(parts), np.concatenate(labels), np.concatenate(device_parts)
    ids = np.asarray(source_ids)
    zero = np.all(X == 0, axis=1)
    audit.update(rows=len(y), columns=len(columns), numeric_predictors=len(features),
                 devices=len(np.unique(devices)), label_counts={str(k): int(v) for k,v in Counter(y).items()},
                 all_zero_rows=int(zero.sum()), all_zero_benign=int(np.sum(zero & (y == 0))),
                 all_zero_attack=int(np.sum(zero & (y == 1))), feature_names=features,
                 all_columns=columns, category_counts=dict(Counter(categories)))
    dump_json(out / "source_audit.json", audit)
    pd.DataFrame({"feature_index": np.arange(1, len(features) + 1), "feature": features}).to_csv(out / "feature_schema.csv", index=False)
    pd.DataFrame({"device": devices, "label": y, "all_zero": zero}).groupby(["device", "label"]).agg(records=("label", "size"), all_zero=("all_zero", "sum")).reset_index().to_csv(out / "device_class_counts.csv", index=False)
    return X, y, devices, ids, zero, audit


def orders_for_review(ids, zero, disagreement, uncertainty):
    n = len(ids)
    stable = np.argsort(np.argsort(ids, kind="stable"), kind="stable")
    random = np.random.default_rng(RANDOM_QUEUE_SEED).permutation(n)
    disagreement_first = np.lexsort((stable, -disagreement.astype(np.int8)))
    zero_first = np.lexsort((stable, -zero.astype(np.int8)))
    priority = np.lexsort((stable, -uncertainty, -disagreement.astype(np.int8)))
    strata = [priority[zero[priority]], priority[~zero[priority]]]

    def proposed(budget):
        quotas = [budget // 2, budget - budget // 2]
        selected = [s[:min(len(s), q)] for s, q in zip(strata, quotas)]
        chosen = np.concatenate(selected)
        if len(chosen) < budget:
            remaining = priority[~np.isin(priority, chosen)]
            chosen = np.concatenate([chosen, remaining[:budget-len(chosen)]])
        return chosen
    return {"Random": lambda k: random[:k],
            "Disagreement first": lambda k: disagreement_first[:k],
            "Zero first": lambda k: zero_first[:k],
            "Stratified review": proposed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
       "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
       "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
       "platform": platform.platform(), "cpu_count_visible": os.cpu_count(),
       "model_parameters": MODEL, "group_split_seed": 42, "test_device_fraction": 0.20,
       "noise_rate": NOISE_RATE, "noise_seed": NOISE_SEED, "random_queue_seed": RANDOM_QUEUE_SEED,
       "review_budgets": BUDGETS, "post_review_update_budget": UPDATE_BUDGET,
       "score_rounding_decimals": 12, "oof_method": "GroupKFold(5), shuffle=False",
       "interpretation": "Exploratory, one synthetic noise seed and fixed outer Split 1; supplied labels are not independently adjudicated truth."}
    dump_json(out / "run_manifest.json", manifest)
    X, y, devices, ids, zero, audit = load_data(args.archive.resolve(), out)
    splits = list(GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=42).split(X, y, groups=devices))
    manifest["group_membership"] = [{"split": i, "training_devices": sorted(np.unique(devices[tr]).tolist()),
           "testing_devices": sorted(np.unique(devices[te]).tolist()),
           "training_rows": len(tr), "testing_rows": len(te)} for i,(tr,te) in enumerate(splits, 1)]
    for tr, te in splits:
        assert set(devices[tr]).isdisjoint(devices[te])
        assert set(ids[tr]).isdisjoint(ids[te])
    train, test = splits[0]
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    np.savez_compressed(out / "split1_source_indices.npz", train_indices=train, test_indices=test)
    print("Computing exact float32 predictor-vector groups", flush=True)
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    n0 = np.bincount(inverse, weights=(y == 0)).astype(np.int64)
    n1 = np.bincount(inverse, weights=(y == 1)).astype(np.int64)
    shared = np.isin(inverse[test], np.unique(inverse[train]))
    ambiguity = {"unique_float32_vectors": len(n0), "contradictory_vector_groups": int(np.sum((n0 > 0) & (n1 > 0))),
        "minimum_in_sample_errors_all_vectors": int(np.minimum(n0, n1).sum()),
        "all_vector_in_sample_accuracy_ceiling_pct": (1 - np.minimum(n0,n1).sum()/len(y))*100,
        "test_rows_with_vector_seen_in_train": int(shared.sum()),
        "nonzero_test_rows_with_vector_seen_in_train": int(np.sum(shared & ~zero[test])),
        "zero_vector_accuracy_ceiling_pct": (1-min(audit["all_zero_benign"],audit["all_zero_attack"])/len(y))*100}
    dump_json(out / "representation_audit.json", ambiguity)
    print("Reproducing Split 1 forest", flush=True)
    baseline, base_pred, elapsed = fit_predict(Xtr, ytr, Xte)
    baseline_metrics = metrics(yte, base_pred)
    print(f"Split 1 fit and prediction completed in {elapsed:.1f}s: {baseline_metrics}", flush=True)
    joblib.dump(baseline, out / "split1_baseline_forest.joblib", compress=3)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    reference_results = [{"model": "Always benign", **metrics(yte,dummy.predict(Xte))},
                         {"model": "Original-label forest", **baseline_metrics}]
    test_predictions = pd.DataFrame({"source_id": ids[test], "device": devices[test],
             "supplied_label": yte, "all_zero": zero[test], "original_label_prediction": base_pred})
    # These labels are deliberately perturbed for a separate controlled pilot.
    noise_rng = np.random.default_rng(NOISE_SEED)
    flipped = noise_rng.choice(len(train), size=math.ceil(NOISE_RATE*len(train)), replace=False)
    injected = np.zeros(len(train), dtype=bool)
    injected[flipped] = True
    ynoisy = ytr.copy()
    ynoisy[flipped] = 1 - ynoisy[flipped]
    assert len(flipped) == int(np.sum(ytr != ynoisy))
    print(f"Controlled pilot: injected {len(flipped):,} known training-label flips", flush=True)
    oof_score = np.full(len(train), np.nan, dtype=np.float64)
    folds = list(GroupKFold(n_splits=5).split(Xtr, ynoisy, groups=devices[train]))
    fold_rows = []

    def oof_fold(fold, tr, val):
        assert set(devices[train][tr]).isdisjoint(devices[train][val])
        model = RandomForestClassifier(**MODEL).fit(Xtr[tr], ynoisy[tr])
        score = model.predict_proba(Xtr[val])[:, 1]
        return fold, val, score

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(oof_fold, i, tr, val):(tr,val) for i,(tr,val) in enumerate(folds,1)}
        for future in as_completed(jobs):
            fold, val, score = future.result()
            tr = jobs[future][0]
            oof_score[val] = score
            fold_rows.append({"fold": fold, "training_devices": sorted(np.unique(devices[train][tr]).tolist()),
                              "validation_devices": sorted(np.unique(devices[train][val]).tolist()),
                              "training_rows": len(tr), "validation_rows": len(val)})
            print(f"OOF fold {fold}/5 completed", flush=True)
    assert np.isfinite(oof_score).all()
    oof_score = np.round(oof_score, 12)
    disagreement = (oof_score > 0.5).astype(np.int8) != ynoisy
    uncertainty = 1 - np.abs(2*oof_score - 1)
    queues = orders_for_review(ids[train], zero[train], disagreement, uncertainty)
    recovery_rows = []
    update_selections = {}
    for name, selector in queues.items():
        for budget_fraction in BUDGETS:
            budget = math.ceil(budget_fraction * len(train))
            selected = selector(budget)
            assert len(selected) == budget and len(np.unique(selected)) == budget
            recovered = int(injected[selected].sum())
            recovery_rows.append({"method": name, "budget_pct": budget_fraction*100,
                 "reviewed_rows": budget, "recovered_injected_flips": recovered,
                 "injected_flip_recovery_pct": recovered/len(flipped)*100,
                 "injected_flip_yield_pct": recovered/budget*100,
                 "zero_rows_reviewed": int(zero[train][selected].sum()),
                 "disagreement_rows_reviewed": int(disagreement[selected].sum())})
            if budget_fraction == UPDATE_BUDGET:
                update_selections[name] = selected
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(out / "review_budget_results.csv", index=False)
    candidate = pd.DataFrame({"source_id": ids[train], "device": devices[train], "supplied_label": ytr,
       "perturbed_label": ynoisy, "known_injected_flip": injected, "all_zero": zero[train],
       "out_of_group_attack_score": oof_score, "disagreement": disagreement,
       "uncertainty_heuristic": uncertainty})
    for name, selection in update_selections.items():
        mask = np.zeros(len(train), dtype=bool); mask[selection] = True
        candidate["reviewed_5pct_"+name.lower().replace(" ","_")] = mask
    candidate.to_csv(out / "training_review_candidates.csv.gz", index=False, compression="gzip")
    print("Fitting perturbed-label forest and four equal-budget restorations", flush=True)
    labels_to_fit = {"Perturbed, no review": ynoisy}
    for name, selected in update_selections.items():
        repaired = ynoisy.copy()
        restore = selected[injected[selected]]
        repaired[restore] = ytr[restore]
        assert np.all(repaired[~injected] == ytr[~injected])
        labels_to_fit[name] = repaired

    def evaluate_update(name, labels):
        _, pred, seconds = fit_predict(Xtr, labels, Xte)
        return name, pred, seconds, metrics(yte,pred)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(evaluate_update, name, labels) for name,labels in labels_to_fit.items()]
        for future in as_completed(jobs):
            name, pred, seconds, measured = future.result()
            reference_results.append({"model": name, **measured})
            test_predictions[name.lower().replace(" ","_").replace(",","")+"_prediction"] = pred
            print(f"{name}: completed in {seconds:.1f}s, F1={measured['attack_f1_pct']:.4f}%", flush=True)
    pd.DataFrame(reference_results).to_csv(out / "post_review_detection.csv", index=False)
    test_predictions.to_csv(out / "split1_test_predictions.csv.gz", index=False, compression="gzip")
    auto = ~zero[test]
    selective = {"deferred_records": int(zero[test].sum()), "deferred_attacks": int(np.sum(zero[test] & (yte == 1))),
        "coverage_auto_pct": auto.mean()*100, "automatic_subset_metrics": metrics(yte[auto],base_pred[auto]),
        "note": "Deferred records are unresolved, not correct detections."}
    dump_json(out / "selective_prediction.json", selective)
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-begin,
                    source_sha256=audit["archive_sha256"], oof_folds=sorted(fold_rows,key=lambda r:r["fold"]),
                    injected_flip_count=len(flipped), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump_json(out / "run_manifest.json", manifest)
    summary = {"baseline": baseline_metrics, "representation": ambiguity, "selective": selective,
               "controlled_pilot_budget_results": recovery_rows, "post_review_detection": reference_results,
               "elapsed_seconds": manifest["elapsed_seconds"]}
    dump_json(out / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
