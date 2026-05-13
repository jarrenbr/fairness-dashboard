"""Baseline import smoke tests for Run 1."""

from __future__ import annotations

import importlib
import runpy
from pathlib import Path


def test_import_src_metrics() -> None:
    module = importlib.import_module("src.metrics")
    assert module is not None


def test_import_src_match() -> None:
    module = importlib.import_module("src.match")
    assert module is not None


def test_import_src_dashboard_logic() -> None:
    module = importlib.import_module("src.dashboard_logic")
    assert module is not None


def test_import_src_app() -> None:
    module = importlib.import_module("src.app")
    assert module is not None


def test_run_app_by_file_path_imports_cleanly() -> None:
    app_path = Path(__file__).resolve().parent.parent / "src" / "app.py"
    namespace = runpy.run_path(str(app_path), run_name="streamlit_app_test")
    assert "main" in namespace
    assert callable(namespace["main"])
