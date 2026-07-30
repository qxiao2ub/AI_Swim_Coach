# AI Swimming Coach - Streamlit Prototype

A GitHub-ready Streamlit application converted from the Colab notebook. The app accepts a swimming video, extracts pose landmarks, creates an annotated video, generates machine-readable time-series features, produces coach-style recommendations, trains supervised-learning baselines, and collects swimmer or coach feedback.

## Main functions

- Upload MP4, MOV, M4V, AVI, MKV, or WMV video containers.
- Extract 33 MediaPipe pose landmarks frame by frame.
- Generate an annotated H.264 video and downloadable analysis bundle.
- Calculate joint angles, body-line deviation, head alignment, kick amplitude, movement speeds, and left-right symmetry features.
- Train Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM demo baselines.
- Prepare sequence windows for an optional Conv1D plus bidirectional LSTM model.
- Collect ratings and demonstrate a reward-based recommendation-ranking update.

## Streamlit deployment fix

The former repository asked Debian APT to install `ffmpeg`, `libgl1`, and `libglib2.0-0`. The failed Community Cloud log mixed Debian releases and could not satisfy FFmpeg and GLib dependencies, so deployment stopped before Python or `app.py` ran.

This repository avoids that failure:

- `packages.txt` contains only `libgomp1` for the optional ML boosters.
- `imageio-ffmpeg` supplies a bundled FFmpeg executable through a Python wheel.
- OpenCV uses the headless contrib build, so desktop `libGL` and GLib packages are unnecessary.
- `vendor/opencv_contrib_python-4.11.0.86-py3-none-any.whl` is a metadata-only compatibility shim. It satisfies MediaPipe's package-name requirement while installing `opencv-contrib-python-headless`.

See `STREAMLIT_DEPLOYMENT_FIX.md` for details.

## Repository structure

```text
.
|-- app.py                         Streamlit entrypoint
|-- ai_pipeline.py                 Computer-vision and AI pipeline
|-- deep_learning_template.py      Optional TensorFlow/Colab template
|-- requirements.txt               Python dependencies
|-- packages.txt                   Only libgomp1; no apt FFmpeg/OpenCV stack
|-- STREAMLIT_DEPLOYMENT_FIX.md    Explanation and redeployment steps
|-- vendor/
|   `-- opencv_contrib_python-4.11.0.86-py3-none-any.whl
|-- .streamlit/config.toml         Upload and visual settings
|-- notebooks/
|   `-- Jasper_Ding_AI_Swim_Coach_Streamlit_Compatible.ipynb
|-- data/.gitkeep
|-- models/.gitkeep
|-- outputs/.gitkeep
`-- tests/
    |-- test_pipeline_smoke.py
    `-- test_deployment_dependencies.py
```

## Deploy to Streamlit Community Cloud

1. Unzip this package.
2. Replace the contents of the existing GitHub repository with all files and folders from the ZIP.
3. Confirm that the root `packages.txt` contains **only**:

   ```text
   libgomp1
   ```

4. Commit and push the changes.
5. In Streamlit Community Cloud, select the repository and branch.
6. Set the main file path to `app.py`.
7. In Advanced settings, select Python 3.12.
8. Reboot the app.

When replacing an existing repository through GitHub's web uploader, verify that the old `packages.txt` was overwritten. If Community Cloud still displays the old FFmpeg APT error, delete the old app deployment and create it again from the updated branch.

The first video analysis downloads and caches the official MediaPipe Pose Landmarker Lite model. The app sidebar includes a Deployment diagnostics panel showing the detected OpenCV, MediaPipe, imageio-ffmpeg, and FFmpeg executable information.

## Run locally

Use Python 3.12 for the closest match to Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
