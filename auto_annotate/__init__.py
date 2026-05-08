"""TUG auto-annotation library API."""

from pathlib import Path

from .elan_export import write_eaf
from .pipeline_adapter import analyze_and_export, extract_annotations

__all__ = ["analyze_and_export", "extract_annotations", "write_eaf", "Path"]
