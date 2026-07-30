"""Small deployment checks for the Streamlit Community Cloud environment."""

from pathlib import Path


def test_headless_opencv_imports() -> None:
    import cv2

    assert cv2.__version__
    assert hasattr(cv2, "VideoCapture")
    assert hasattr(cv2, "VideoWriter")


def test_bundled_ffmpeg_is_available() -> None:
    import imageio_ffmpeg

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    assert executable.is_file()


def test_pipeline_exposes_ffmpeg_locator() -> None:
    from ai_pipeline import get_ffmpeg_executable

    executable = get_ffmpeg_executable()
    assert executable is not None
    assert Path(executable).is_file()
