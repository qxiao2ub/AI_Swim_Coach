# Streamlit Community Cloud deployment fixes

## Current symptom: installation appears stuck after package resolution

The deployment log in the reported screenshot showed:

```text
Using Python 3.14.7 environment
Resolved 67 packages
```

The application was created with Python 3.14, while the computer-vision stack in this repository is designed and tested for Python 3.12. Several packages are compiled binary distributions. A mismatched runtime can leave the installer searching for compatible wheels or attempting expensive builds after resolution.

### Current fix

- The repository requires a new Community Cloud deployment using **Python 3.12**.
- `requirements.txt` includes `--only-binary=:all:` so Community Cloud never attempts slow source builds.
- Full AI/video dependencies use Python-version markers and install only on Python 3.12.
- On an accidental Python 3.13/3.14 deployment, only Streamlit is installed; `app.py` then displays an immediate runtime repair page.
- XGBoost and LightGBM are optional and are not installed in the default cloud build.
- `.python-version` and `runtime.txt` document the intended runtime, but the Community Cloud **Advanced settings** selector controls the actual deployment.

## Earlier symptom: APT dependency conflict

An older `packages.txt` requested Debian `ffmpeg`, `libgl1`, and `libglib2.0-0`. The build mixed incompatible Debian package versions and could not satisfy FFmpeg and GLib dependencies. The build stopped before Python or `app.py` ran.

### Earlier fix retained here

- `packages.txt` is now intentionally empty.
- `imageio-ffmpeg` supplies a bundled FFmpeg executable from a Python wheel.
- OpenCV uses a headless build, so desktop `libGL` and GLib packages are unnecessary.
- The small wheel in `vendor/` satisfies MediaPipe's declared OpenCV package name while redirecting installation to the headless contrib package.

## Required redeployment procedure

1. Push all files from this repository to GitHub.
2. Verify that the old contents of `requirements.txt` and `packages.txt` were replaced.
3. Open Streamlit Community Cloud and record any secrets or custom subdomain.
4. Delete the existing app whose logs show Python 3.14.
5. Create the app again from the same repository and branch.
6. Use `app.py` as the main file.
7. In **Advanced settings**, select **Python 3.12**.
8. Deploy and verify that the log says Python 3.12.

Rebooting the existing Python 3.14 deployment does not change its Python runtime.
