# Streamlit Community Cloud deployment fix

## What failed

The previous `packages.txt` requested Debian `ffmpeg`, `libgl1`, and
`libglib2.0-0`. The Community Cloud build log showed packages from different
Debian releases, so APT could not satisfy FFmpeg's `libavcodec61` and GLib
requirements. The build stopped before Python packages or `app.py` were run.

## What changed

1. `packages.txt` now contains only `libgomp1`, which is used by LightGBM and
   XGBoost and was already available in the failed build log.
2. `imageio-ffmpeg` provides a bundled FFmpeg executable through Python, so the
   Debian FFmpeg package is no longer required.
3. OpenCV is installed as `opencv-contrib-python-headless`, avoiding desktop
   `libGL` and `libglib` requirements.
4. A metadata-only compatibility wheel in `vendor/` satisfies MediaPipe's
   package-name dependency on `opencv-contrib-python` while redirecting it to
   the headless wheel.
5. The Streamlit sidebar includes a deployment diagnostic panel.

## Redeployment

Replace all files in the GitHub repository with this fixed repository. In
particular, make sure the old `packages.txt` is overwritten. Commit and push,
then reboot the Streamlit app. If the old APT log still appears, delete the app
from Community Cloud and deploy it again from the updated branch.
