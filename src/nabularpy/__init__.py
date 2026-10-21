"""Structured descriptive analysis of observed missing-data patterns."""

from .core import (
    MissingDataError,
    MissingReport,
    inspect,
    missing_by,
    missing_patterns,
    missing_summary,
    shadow,
)

__version__ = "0.1.0"
__all__ = [
    "MissingDataError",
    "MissingReport",
    "inspect",
    "missing_by",
    "missing_patterns",
    "missing_summary",
    "shadow",
]
