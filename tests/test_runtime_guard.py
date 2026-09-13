"""Validate that a wrong Python runtime stops before AI imports.

Run with: python tests/test_runtime_guard.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


class StopCalled(RuntimeError):
    pass


class FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.messages: list[str] = []

    def set_page_config(self, **kwargs) -> None:
        self.messages.append(str(kwargs))

    def title(self, value) -> None:
        self.messages.append(str(value))

    def error(self, value) -> None:
        self.messages.append(str(value))

    def markdown(self, value, **kwargs) -> None:
        del kwargs
        self.messages.append(str(value))

    def code(self, value, **kwargs) -> None:
        del kwargs
        self.messages.append(str(value))

    def stop(self) -> None:
        raise StopCalled("Streamlit runtime guard stopped execution")


def test_wrong_runtime_stops_before_pipeline_import() -> None:
    if sys.version_info[:2] == (3, 12):
        print("Runtime guard branch skipped because this interpreter is Python 3.12.")
        return

    fake = FakeStreamlit()
    original_streamlit = sys.modules.get("streamlit")
    original_pipeline = sys.modules.pop("ai_pipeline", None)
    sys.modules["streamlit"] = fake
    try:
        try:
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        except StopCalled:
            pass
        else:
            raise AssertionError("The app did not stop on an unsupported Python runtime.")

        assert "ai_pipeline" not in sys.modules
        rendered = "\n".join(fake.messages)
        assert "requires Python 3.12" in rendered
        assert "Delete the current Community Cloud app" in rendered
    finally:
        if original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = original_streamlit
        if original_pipeline is not None:
            sys.modules["ai_pipeline"] = original_pipeline


def main() -> None:
    test_wrong_runtime_stops_before_pipeline_import()
    print("Runtime guard check passed.")


if __name__ == "__main__":
    main()
