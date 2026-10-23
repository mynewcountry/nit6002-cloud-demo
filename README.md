# NIT6002 Cybersecurity Data Review

This project wraps the recorded DataSense Random Forest in a Streamlit web app. It checks CSV model inputs, runs predictions and replays the recorded training-label review experiment. It is ready for local setup and cloud deployment; an actual cloud URL and cloud test outcomes must be recorded after deployment.

The source model was replayed against 56,443 original held-out observations with zero predicted-class differences. This was an inference replay of the saved model, not a fresh training run or independent verification of dataset labels. Details are in `verification/inference_replay.json`.

## 1 Download and extract

Extract `NIT6002_Cloud_Demo.zip` into Downloads. The project folder should be named `nit6002_cloud_demo`. Keep its folders together.

| Item | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `ml_core.py` | CSV checks, feature ordering, prediction and review-queue replay |
| `requirements.txt` | Pinned app and model dependencies |
| `model_manifest.json` | Model, schema and source identity |
| `results/` | Recorded model, scores, predictions, audits and comparison tables |
| `demo/` | A 90-row original-data sample, expected predictions and invalid-input examples |
| `analysis/` | Original analysis scripts and verification helpers |
| `tests/` | Functional checks for input validation, inference, queues and the app |
| `verification/` | Local verification and a blank cloud test sheet |

The app uses one coherent saved run: mean accuracy 91.79%, mean attack recall 84.88% and mean attack F1 90.30%. Standalone results from the earlier 91.85% run are not combined with it.

## 2 Create a separate Python environment

Open a Terminal with Anaconda available. Enter these commands in Terminal, rather than a Jupyter code cell. They create a separate environment for the app.

```bash
conda create -n nit6002-cloud python=3.12 pip -y
conda activate nit6002-cloud
cd "$HOME/Downloads/nit6002_cloud_demo"
python -m pip install -r requirements.txt
```

If that environment already exists, activate it and continue from the `cd` command. If the folder was extracted elsewhere, change the path in `cd` to the actual folder. Keep the pinned scikit-learn version: the packaged model was fitted with scikit-learn 1.8.0. The app validates the model and schema hashes before loading them.

## 3 Check the project

```bash
python -m unittest discover -s tests -v
```

The checks should finish with `OK`. They include predictions for the original sample, label exclusion, reordered columns, invalid values and all 16 recorded method-budget combinations. Streamlit can print a ScriptRunContext warning in its headless test mode; assess the final test result and any actual failure trace.

## 4 Start the local app

```bash
python -m streamlit run app.py
```

Open the local address printed in Terminal. It is normally `http://localhost:8501`. Keep the Terminal process running while you use the app. Press Control+C in that Terminal when finished. A local address demonstrates local execution; cloud access is established only after the later deployment steps.

## 5 Walk through the prototype

1. Open **Predict a CSV**, keep **Included demonstration sample** selected, and click **Run model**.
2. Inspect the all-zero flags and the disagreement cases. All-zero records remain in the prediction output and metric denominators.
3. Download **all predictions** and **this run record**. The run record includes the input digest, model identity, runtime versions, quality counts and inference-only timing.
4. Switch to **Upload a CSV** and upload `demo/invalid_schema.csv`. The missing model field should block prediction.
5. Upload `demo/invalid_values.csv`. The invalid feature value should block prediction.
6. Open **Review experiment**. With **Stratified review** and **5%**, the queue should contain 8,538 rows and recover 2,103 known injected flips.
7. Change the budget to **25%**. The queue should contain 42,687 rows and recover 16,298 known flips, or 95.45% of the injected changes.
8. Open **Recorded evaluation** to inspect the five device holdouts and the fixed 5% restoration comparison.

The demo sample was deliberately selected from Split 1 testing devices to include errors and all-zero cases. It contains 90 records and is not a random sample for estimating overall model performance. The actual five-split results are on **Recorded evaluation**.

The review workspace rebuilds queues from saved out-of-group scores. It does not retrain models when a slider changes. The known-flip mask is used after selection for evaluation and is excluded from downloaded queues. This controlled test does not establish which natural dataset labels are incorrect.

## 6 Create the GitHub repository

1. Sign in to GitHub and create a new repository named `nit6002-cloud-demo`.
2. Choose a private repository while developing it.
3. Use **Add file**, then **Upload files**. Upload the contents of `nit6002_cloud_demo`, retaining its subfolders.
4. Check that `app.py` and `requirements.txt` are directly at the repository root. Upload the extracted project files, rather than the ZIP as one file.
5. Commit the files to `main` and record the commit identifier.

The original 294 MB dataset archive stays separately in Downloads. Serving the existing model needs the packaged model, schema, recorded results and small demo inputs. Every packaged project file is below GitHub's 25 MiB browser-upload limit.

## 7 Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io and sign in.
2. Connect the GitHub account that owns the repository and grant access to this private repository.
3. Click **Create app**, then select the option for an existing app.
4. Select repository `nit6002-cloud-demo`, branch `main`, and entrypoint `app.py`.
5. In **Advanced settings**, choose **Python 3.12**. Keep `requirements.txt` at the root. This app does not require an API key or database credentials.
6. Click **Deploy**. Inspect the logs if a dependency or path error appears.
7. Save the actual URL once the app opens. It will use a `streamlit.app` subdomain.

The package contains a single root dependency file for deployment. Keep the exact model dependencies to avoid loading a scikit-learn model under an incompatible version.

## 8 Test the cloud version

Repeat Step 5 at the deployed URL. Then download the cloud sample predictions and compare them on your Mac:

```bash
conda activate nit6002-cloud
cd "$HOME/Downloads/nit6002_cloud_demo"
python analysis/verify_cloud_export.py "$HOME/Downloads/predictions.csv"
```

If the browser renamed the file, use its actual filename in the last command. The helper expects the 90-row included sample. It checks identical predicted classes and model-score agreement within an absolute tolerance of 0.000000000001. It records the result in `verification/latest_export_comparison.json`.

The app reports inference time, excluding parsing, model loading and page rendering. For deployment testing, record repeated measured times separately for local and cloud execution. Include batch size, environment and whether the first run loaded the model. Report the measurements you actually observe.

Open the app on another device or network and confirm that the sample can run. In **Deployment evidence**, enter the real URL and download the deployment record. That record labels the URL as user-entered; retain browser screenshots or other actual access evidence as well.

## 9 Set assessor access

A private GitHub repository produces a private Streamlit app by default. Use the app's **Sharing** settings to choose who can view it. For a private demonstration, give the intended assessor viewer access and test that access before the defence. Streamlit Community Cloud currently allows one private app at a time.

## 10 Keep the demonstration evidence

Fill in `verification/cloud_test_record.csv` with actual results, UTC execution times, the real URL and the GitHub commit. Its outcome fields are intentionally blank until you perform those tests.

Keep these items together:

- The source code commit and `requirements.txt`.
- The deployed URL, deployment date and tested access conditions.
- The cloud predictions, run record and export-comparison result.
- Screenshots showing the live address, sample predictions, all-zero review flags and invalid-input rejection.
- The 5% and 25% review-queue outputs.
- Recorded response-time measurements and any failures.

No cloud deployment is claimed by opening this ZIP or running the local app. A cloud demonstration is complete only when the remote app is accessible and the cloud tests have been performed.

## 11 Use the prototype in the final report

Describe its architecture and implemented components in Methodology. Describe compatible inputs and validation in Data Collection and Processing. Report prediction replay, deployment checks, measured timing and failed cases in Experimental Evaluation. Interpret what the prototype establishes and its limits in Discussion.

The app does not implement live SIEM ingestion, SOAR containment or independent natural-label adjudication. Original labels remain reference annotations. Cloud hosting is additional implementation evidence; the 2–3 relevant published-method comparisons, controlled ablation and broader experimental evaluation still need to be completed for the intended HD submission.

Refer to it as a cloud-hosted research prototype once actually deployed. The supplied assessment files did not establish a D633-specific hosting requirement.

## 12 Repeat the original analysis when needed

The original scripts are included unchanged. The saved model serves predictions without retraining. To repeat model fitting, keep the original DataSense archive in Downloads, then run the scripts separately:

```bash
python analysis/run_analysis.py --archive "$HOME/Downloads/all_attack_benign_samples.tar.xz" --output-dir rerun_results
python analysis/run_repeated_baselines.py --archive "$HOME/Downloads/all_attack_benign_samples.tar.xz" --output-dir rerun_results
```

This work can take several minutes and needs memory for the dataset and multiple forests. New results remain in `rerun_results`; the app continues serving its recorded bundle. Adopt a new run only after replacing its model, schema, summaries, row outputs and manifests as one consistent version.

## Common setup issues

| Message or symptom | Action |
|---|---|
| `conda: command not found` | Open a Terminal through your Anaconda installation, then repeat the setup commands. |
| `app.py` cannot be found | Change Terminal to the folder containing `app.py`. Check its name and location in Finder. |
| `No module named streamlit` | Activate `nit6002-cloud` and install `requirements.txt` in that environment. |
| Model-version error | Confirm Python 3.12 and scikit-learn 1.8.0 are installed from the supplied requirements. |
| Missing packaged asset | Restore the complete extracted folder, including `results` and `demo`. |
| Cloud repository is not listed | Check the connected GitHub account and repository access. |
| Assessor cannot open the app | Check the app's Sharing settings and their viewer access. |

## Sources

- DataSense dataset: https://www.unb.ca/cic/datasets/iiot-dataset-2025.html
- Firouzi et al. (2025), DataSense: https://doi.org/10.3390/electronics14204095
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Running the app: https://docs.streamlit.io/develop/concepts/architecture/run-your-app
- Streamlit sharing: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- GitHub uploads: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html
