"""Static checks for the GitHub and Streamlit Community Cloud package.

Run with: python tests/test_repository_layout.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist() -> None:
    required = [
        "app.py",
        "ai_pipeline.py",
        "requirements.txt",
        "requirements-optional-boosters.txt",
        "packages.txt",
        ".python-version",
        "runtime.txt",
        ".streamlit/config.toml",
        "vendor/opencv_contrib_python-4.11.0.86-py3-none-any.whl",
        "AUTHORS.md",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"Missing repository files: {missing}"


def test_python_runtime_hints() -> None:
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    assert (ROOT / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.12"


def test_apt_package_list_is_empty() -> None:
    contents = (ROOT / "packages.txt").read_text(encoding="utf-8").strip()
    assert contents == "", "packages.txt must remain empty to avoid APT dependency conflicts."


def test_cloud_requirements_are_binary_only_and_runtime_guarded() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "--only-binary=:all:" in text
    assert "streamlit==1.63.0" in text

    compiled_packages = [
        "mediapipe==0.10.35",
        "opencv_contrib_python-4.11.0.86",
        "imageio-ffmpeg==0.6.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.8.0",
        "joblib==1.5.3",
    ]
    for package in compiled_packages:
        matching_lines = [line for line in text.splitlines() if package in line]
        assert matching_lines, f"Missing dependency line for {package}"
        assert all('python_version >= "3.12"' in line for line in matching_lines)
        assert all('python_version < "3.13"' in line for line in matching_lines)


def test_optional_boosters_are_not_in_default_requirements() -> None:
    default_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    optional_text = (ROOT / "requirements-optional-boosters.txt").read_text(
        encoding="utf-8"
    ).lower()
    assert not re.search(r"^xgboost", default_text, flags=re.MULTILINE)
    assert not re.search(r"^lightgbm", default_text, flags=re.MULTILINE)
    assert re.search(r"^xgboost", optional_text, flags=re.MULTILINE)
    assert re.search(r"^lightgbm", optional_text, flags=re.MULTILINE)


def test_credits_are_present() -> None:
    combined = "\n".join(
        (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        for name in ["app.py", "ai_pipeline.py", "README.md", "AUTHORS.md"]
    )
    assert "Jasper Ding" in combined
    assert "Dr. Qingyang Xiao" in combined


def main() -> None:
    test_required_files_exist()
    test_python_runtime_hints()
    test_apt_package_list_is_empty()
    test_cloud_requirements_are_binary_only_and_runtime_guarded()
    test_optional_boosters_are_not_in_default_requirements()
    test_credits_are_present()
    print("Repository layout and deployment configuration checks passed.")


if __name__ == "__main__":
    main()
