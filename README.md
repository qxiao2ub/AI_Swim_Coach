# AI Swimming Coach - Streamlit Community Cloud Repository

A GitHub-ready Streamlit prototype that accepts a swimming video, extracts pose landmarks, produces an annotated video and machine-readable time series, generates coaching insights, trains supervised-learning baselines, and collects swimmer or coach feedback.

## Project team

- **Author:** Jasper Ding
- **Advisor:** Dr. Qingyang Xiao

## Important deployment requirement

**Deploy this repository with Python 3.12.**

The video-analysis stack uses MediaPipe 0.10.35 and binary scientific-computing wheels selected for Python 3.12. The app includes a runtime guard: when a deployment is accidentally created with Python 3.13 or 3.14, it opens a repair screen instead of importing the full computer-vision stack.

An existing Streamlit Community Cloud app cannot switch Python versions in place. If the build log says `Using Python 3.14...`, delete that deployment and create it again from the same GitHub repository, selecting **Python 3.12** in **Advanced settings**.

See `DEPLOYMENT_CHECKLIST.md` for the exact steps.

## Main functions

- Upload MP4, MOV, M4V, AVI, MKV, or WMV video containers.
- Extract 33 pose landmarks frame by frame with the MediaPipe Tasks Pose Landmarker.
- Generate an annotated video and a downloadable analysis bundle.
- Calculate joint angles, body-line deviation, head alignment, kick amplitude, movement speeds, and left-right symmetry features.
- Generate prioritized coach-style observations and training drills.
- Train cloud-friendly Logistic Regression, Random Forest, and Gradient Boosting baselines.
- Optionally train XGBoost and LightGBM in a local or Colab environment.
- Prepare sequence windows for an optional Conv1D plus bidirectional LSTM model.
- Collect ratings and demonstrate a reward-based recommendation-ranking update.

## Why the previous deployment appeared stuck

The screenshot showed:

```text
Using Python 3.14.7 environment
Resolved 67 packages
```

The earlier dependency set pinned compiled packages such as NumPy and MediaPipe for an unsupported or mismatched runtime. After dependency resolution, the installer could spend a long time looking for compatible artifacts or attempting builds. The repository now prevents that failure in four ways:

1. The full compiled AI/video stack is installed only when the runtime is Python 3.12.
2. `--only-binary=:all:` prevents slow source compilation on Community Cloud.
3. XGBoost and LightGBM were removed from the default cloud dependency set.
4. `packages.txt` is intentionally empty, so no Debian APT video or GUI packages are installed.

The default app still includes three supervised-learning models. Optional boosters remain available through `requirements-optional-boosters.txt` for local or Colab use.

## Repository structure

```text
.
|-- app.py                              Streamlit entry point and runtime guard
|-- ai_pipeline.py                      Pose, feature, recommendation, ML, and feedback pipeline
|-- deep_learning_template.py           Optional TensorFlow/Colab sequence-model template
|-- requirements.txt                    Python 3.12 Community Cloud dependencies
|-- requirements-optional-boosters.txt  Optional XGBoost and LightGBM extension
|-- packages.txt                        Intentionally empty; overwrites old APT package lists
|-- .python-version                     Python 3.12 hint for compatible hosts/tools
|-- runtime.txt                         Python 3.12 compatibility hint
|-- DEPLOYMENT_CHECKLIST.md             Exact GitHub and Community Cloud steps
|-- STREAMLIT_DEPLOYMENT_FIX.md         Technical explanation of both deployment fixes
|-- AUTHORS.md                          Project credits
|-- vendor/
|   `-- opencv_contrib_python-4.11.0.86-py3-none-any.whl
|-- .streamlit/config.toml              Upload limits and visual settings
|-- notebooks/
|   `-- Jasper_Ding_AI_Swim_Coach_Streamlit_Compatible.ipynb
|-- data/.gitkeep
|-- models/.gitkeep
|-- outputs/.gitkeep
|-- tests/
|   |-- test_pipeline_smoke.py
|   |-- test_deployment_dependencies.py
|   |-- test_repository_layout.py
|   `-- test_runtime_guard.py
`-- .github/workflows/streamlit-smoke.yml
```

## Deploy to Streamlit Community Cloud

1. Extract the ZIP file.
2. Replace the GitHub repository contents with the extracted files and folders.
3. Confirm that root-level `requirements.txt`, `.python-version`, `runtime.txt`, and the empty `packages.txt` were committed.
4. Delete the existing Community Cloud deployment if its log reports Python 3.13 or 3.14.
5. Create a new app from the updated GitHub repository and branch.
6. Set the main file path to `app.py`.
7. Open **Advanced settings** and select **Python 3.12**.
8. Deploy.
9. Confirm that the build log begins with a Python 3.12 environment and that the app opens the AI Swimming Coach page.

A reboot of the old Python 3.14 deployment is not sufficient because rebooting preserves its selected runtime.

## Run locally

Use Python 3.12 for the closest match to Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

For optional XGBoost and LightGBM support:

```bash
python -m pip install -r requirements-optional-boosters.txt
```

## Recommended video input

For a fast cloud demonstration, use a 5-20 second H.264 MP4 clip with the full swimmer visible. A stable side view generally gives the most interpretable body-line and joint-angle results. Underwater refraction, splashes, glare, and overlapping limbs can reduce pose accuracy.

## Important model limitations

- The coaching thresholds are prototype heuristics and must be validated by qualified swimming coaches.
- The supervised-learning tab creates KMeans-derived demonstration labels only to prove that the training pipeline works. They are not coach ground truth.
- TensorFlow is intentionally excluded from the cloud deployment because it is resource intensive. Use `deep_learning_template.py` or the included Colab notebook for Conv1D plus BiLSTM training.
- Community Cloud file storage is temporary. Download generated analysis bundles, model bundles, and feedback before a restart or redeployment.
- This educational prototype is not a medical device, injury assessment, or substitute for an in-person qualified coach.

## Moving from prototype to coach-grade AI

Replace demonstration labels with consented, coach-reviewed labels such as stroke phase, elbow-drop category, body-roll range, breathing event, kick-timing issue, and technique severity. Split data by swimmer rather than by frame to prevent information leakage, validate across camera views and pool conditions, and report confidence intervals in addition to accuracy.
