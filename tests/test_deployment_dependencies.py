"""Deployment checks for the Python 3.12 Community Cloud environment.

Run with: python tests/test_deployment_dependencies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_headless_opencv_imports() -> None:
    import cv2

    assert cv2.__version__
    assert hasattr(cv2, "VideoCapture")
    assert hasattr(cv2, "VideoWriter")


def test_bundled_ffmpeg_is_available() -> None:
    import imageio_ffmpeg

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    assert executable.is_file()


def test_mediapipe_tasks_api_is_available() -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    assert mp.__version__
    assert hasattr(vision, "PoseLandmarker")
    assert hasattr(vision, "PoseLandmarkerOptions")


def test_pipeline_exposes_ffmpeg_locator() -> None:
    from ai_pipeline import get_ffmpeg_executable

    executable = get_ffmpeg_executable()
    assert executable is not None
    assert Path(executable).is_file()


def main() -> None:
    test_headless_opencv_imports()
    test_bundled_ffmpeg_is_available()
    test_mediapipe_tasks_api_is_available()
    test_pipeline_exposes_ffmpeg_locator()
    print("Deployment dependency checks passed.")


if __name__ == "__main__":
    main()
