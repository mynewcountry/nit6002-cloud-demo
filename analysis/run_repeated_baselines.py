"""Complete the five device-partition replications in the current environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from run_analysis import load_data, fit_predict, metrics, dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    X, y, devices, ids, zero, audit = load_data(args.archive, out)
    splits = list(GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=42).split(X,y,groups=devices))
    first = pd.read_csv(out/'post_review_detection.csv').set_index('model').loc['Original-label forest'].to_dict()
    rows = [{'split':1, 'test_rows':len(splits[0][1]), **first}]
    def evaluate(i, tr, te):
        _, prediction, elapsed = fit_predict(X[tr],y[tr],X[te])
        return {'split':i,'test_rows':len(te),**metrics(y[te],prediction)}, elapsed
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(evaluate,i,tr,te) for i,(tr,te) in enumerate(splits[1:],2)]
        for job in as_completed(jobs):
            row, elapsed=job.result(); rows.append(row)
            print(f"Split {row['split']} finished in {elapsed:.1f}s: accuracy={row['accuracy_pct']:.6f}%",flush=True)
    repeated=pd.DataFrame(rows).sort_values('split')
    repeated.to_csv(out/'repeated_baseline_results.csv',index=False)
    columns=[c for c in repeated.columns if c.endswith('_pct')]
    summary=repeated[columns].agg(['mean','std','min','max']).T
    summary.to_csv(out/'repeated_baseline_summary.csv')
    predictions=pd.read_csv(out/'split1_test_predictions.csv.gz')
    device_rows=[]
    for name, data in predictions.groupby('device'):
        device_rows.append({'device':name, 'records':len(data), **metrics(data.supplied_label,data.original_label_prediction)})
    pd.DataFrame(device_rows).to_csv(out/'split1_device_performance.csv',index=False)
    manifest=json.loads((out/'run_manifest.json').read_text())
    manifest['repeated_baselines_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['repeated_baselines_completed']=5
    dump_json(out/'run_manifest.json',manifest)
    print('REPEATED RESULTS',repeated.to_json(orient='records'),flush=True)
    print('REPEATED SUMMARY',summary.to_json(orient='index'),flush=True)


if __name__=='__main__': main()
