# AI Swimming Coach - Streamlit Prototype

A GitHub-ready Streamlit application converted from the uploaded Colab notebook. The app accepts a swimming video, extracts pose landmarks, creates an annotated video, generates machine-readable time-series features, produces coach-style recommendations, trains supervised learning baselines, and collects user or coach feedback.

## Main functions

- Upload MP4, MOV, M4V, AVI, MKV, or WMV video containers.
- Extract 33 MediaPipe pose landmarks frame by frame.
- Generate an annotated video and downloadable analysis bundle.
- Calculate joint angles, body-line deviation, head alignment, kick amplitude, movement speeds, and left-right symmetry features.
- Train Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM demo baselines.
- Prepare sequence windows for an optional Conv1D plus bidirectional LSTM model.
- Collect ratings and demonstrate a reward-based recommendation ranking update.

## Repository structure

```text
.
|-- app.py                         Streamlit entrypoint
|-- ai_pipeline.py                 Reusable computer-vision and AI pipeline
|-- deep_learning_template.py      Optional TensorFlow/Colab template
|-- requirements.txt               Python dependencies
|-- packages.txt                   Debian packages for Community Cloud
|-- .streamlit/config.toml         Upload and visual settings
|-- notebooks/
|   `-- Jasper_Ding_AI_Swim_Coach_Streamlit_Compatible.ipynb
|-- data/.gitkeep
|-- models/.gitkeep
|-- outputs/.gitkeep
`-- tests/test_pipeline_smoke.py
```

## Deploy to Streamlit Community Cloud

1. Unzip the downloaded repository package.
2. Create an empty GitHub repository.
3. Upload all files and folders inside the unzipped project folder to the root of the GitHub repository.
4. In Streamlit Community Cloud, create a new app and select the GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Open Advanced settings and select Python 3.12.
7. Deploy the app. The first video analysis downloads and caches the official MediaPipe Pose Landmarker Lite model.

`requirements.txt` and `packages.txt` are already placed in the repository root so Community Cloud can install the Python and Linux dependencies automatically.

## Run locally

Use Python 3.12 for the closest match to Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Recommended video input

For a fast cloud demo, use a 5-20 second H.264 MP4 clip with the full swimmer visible. A stable side view generally gives the most interpretable body-line and joint-angle results. Underwater refraction, splashes, glare, and overlapping limbs can reduce pose accuracy.

## Important model limitations

- The coaching thresholds are prototype heuristics and must be validated by qualified swimming coaches.
- The supervised-learning tab creates KMeans-derived demo labels only to prove that the training pipeline works. These labels are not coach ground truth.
- The default Community Cloud app does not install TensorFlow because it is resource intensive. Use `deep_learning_template.py` or the included Colab notebook for Conv1D plus BiLSTM training.
- Community Cloud file storage is temporary. Download generated analysis bundles, model bundles, and feedback before the app restarts or redeploys.
- This software is an educational prototype, not a medical device, injury assessment, or substitute for an in-person qualified coach.

## Moving from prototype to coach-grade AI

Replace demo labels with consented, coach-reviewed labels such as stroke phase, elbow-drop category, body-roll range, breathing event, kick-timing issue, and technique severity. Split data by swimmer rather than by frame to prevent information leakage, validate across camera views and pool conditions, and report confidence intervals in addition to accuracy.
