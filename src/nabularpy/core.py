"""Descriptive missingness structures, independent of optional plotting tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd


class MissingDataError(ValueError):
    """Invalid labels or an ambiguous missing-data analysis request."""


def _validate(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not frame.columns.is_unique or isinstance(frame.columns, pd.MultiIndex):
        raise MissingDataError("Input requires unique flat column labels")


def _summary(mask: pd.DataFrame) -> pd.DataFrame:
    counts = mask.sum(axis=0).to_numpy(dtype=np.int64)
    rows = len(mask)
    return pd.DataFrame(
        {
            "variable": pd.Series(list(mask.columns), dtype=object),
            "missing": counts,
            "present": rows - counts,
            "percentage": counts * (100.0 / rows) if rows else np.zeros(len(counts), dtype=float),
        }
    )


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Missing/present counts and percentages per column, in input column order."""
    _validate(data)
    return _summary(data.isna())


def _patterns(mask: pd.DataFrame) -> pd.DataFrame:
    # Exact packed bytes, not hashes: there is no collision assumption. Only
    # observed row patterns are counted; never enumerate the power set.
    packed = np.packbits(mask.to_numpy(dtype=np.bool_), axis=1, bitorder="little")
    counts: dict[bytes, tuple[int, int]] = {}
    for pos, bits in enumerate(packed):
        key = bits.tobytes()
        previous, first = counts.get(key, (0, pos))
        counts[key] = (previous + 1, first)
    patterns: list[tuple[Any, ...]] = []
    frequencies: list[int] = []
    for count, first in sorted(counts.values(), key=lambda entry: (-entry[0], entry[1])):
        patterns.append(
            tuple(
                column
                for column, absent in zip(mask.columns, mask.iloc[first], strict=True)
                if absent
            )
        )
        frequencies.append(count)
    return pd.DataFrame(
        {
            "pattern": pd.Series(patterns, dtype=object),
            "count": pd.Series(frequencies, dtype="int64"),
            "percentage": pd.Series(
                [count * 100.0 / len(mask) for count in frequencies], dtype=float
            ),
            "n_missing": pd.Series([len(pattern) for pattern in patterns], dtype="int64"),
        }
    )


def missing_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """Count observed missing-label tuples, by frequency then first occurrence.

    Empty-label tuple () denotes a complete record. For zero columns and n>0
    rows it occurs n times; an empty input has no observed patterns.
    """
    _validate(data)
    return _patterns(data.isna())


def shadow(data: pd.DataFrame, *, suffix: str = "_NA", bind: bool = False) -> pd.DataFrame:
    """Return boolean indicators (True means missing), optionally beside data.

    The original index, including duplicate labels, is preserved. Generated
    names must be unique, and binding must not overwrite any original column.
    """
    _validate(data)
    if not isinstance(suffix, str) or not suffix:
        raise MissingDataError("suffix must be a nonempty string")
    names = [f"{column}{suffix}" for column in data.columns]
    if len(set(names)) != len(names) or (bind and set(names).intersection(data.columns)):
        raise MissingDataError(
            "Generated shadow names collide; rename inputs or choose another suffix"
        )
    result = data.isna()
    result.columns = names
    return pd.concat([data.copy(deep=True), result], axis=1) if bind else result


def missing_by(
    data: pd.DataFrame,
    *,
    group: str | Sequence[str],
    include_group: bool = False,
) -> pd.DataFrame:
    """Long-form per-segment summaries; include null groups, exclude unused levels.

    Group order is first observation order; variable order follows input columns.
    Group column names may not collide with the output metadata column names.
    """
    _validate(data)
    keys = [group] if isinstance(group, str) else list(group)
    metadata = ["variable", "n_rows", "missing", "present", "percentage"]
    if (
        not keys
        or not all(isinstance(key, str) for key in keys)
        or len(set(keys)) != len(keys)
        or any(key not in data.columns for key in keys)
    ):
        raise MissingDataError("group must contain distinct existing string column names")
    if set(keys).intersection(metadata):
        raise MissingDataError("Group names collide with summary metadata; rename group columns")
    records: list[dict[str, Any]] = []
    null_group = object()  # An internal identity token cannot collide with data strings.
    segments: dict[tuple[Any, ...], list[int]] = {}
    try:
        for pos, values in enumerate(data.loc[:, keys].itertuples(index=False, name=None)):
            key = tuple(
                null_group if pd.api.types.is_scalar(value) and bool(pd.isna(value)) else value
                for value in values
            )
            segments.setdefault(key, []).append(pos)
        for positions in segments.values():
            segment = data.iloc[positions]
            # Pandas groupby can coerce an object-backed large integer/null key
            # index to float. Read original scalar labels, never inferred keys.
            labels = tuple(data[key].iloc[positions[0]] for key in keys)
            measured = segment if include_group else segment.drop(columns=keys)
            for summary in missing_summary(measured).to_dict("records"):
                records.append(
                    {
                        **dict(zip(keys, labels, strict=True)),
                        **cast(dict[str, Any], summary),
                        "n_rows": len(segment),
                    }
                )
    except TypeError as exc:
        raise MissingDataError(f"Group values must be hashable: {exc}") from exc
    result = pd.DataFrame(records, columns=[*keys, *metadata], dtype=object)
    return result.astype(
        {"n_rows": "int64", "missing": "int64", "present": "int64", "percentage": float}
    )


@dataclass(frozen=True)
class MissingReport:
    """Snapshot report. Returned tables can be changed without altering the report."""

    _data: pd.DataFrame = field(repr=False)
    _mask: pd.DataFrame = field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Return missing/present counts and percentages per variable."""
        return _summary(self._mask)

    def patterns(self) -> pd.DataFrame:
        """Return observed patterns, including complete records."""
        return _patterns(self._mask)

    def row_summary(self) -> pd.DataFrame:
        """Return row counts and percentages, preserving the input index."""
        counts = self._mask.sum(axis=1).astype("int64")
        width = len(self._mask.columns)
        return pd.DataFrame(
            {
                "missing": counts,
                "present": width - counts,
                "percentage": counts * 100.0 / width if width else counts.astype(float),
            },
            index=self._mask.index,
        )

    def cooccurrence(self) -> pd.DataFrame:
        """Count rows with both variables missing; diagonal equals missing counts.

        This is descriptive count association, not a hypothesis test or a
        missingness-mechanism classification. The output is quadratic in columns.
        """
        integer_mask = self._mask.astype("int64")
        return integer_mask.T.dot(integer_mask)

    def missing_by(
        self, group: str | Sequence[str], *, include_group: bool = False
    ) -> pd.DataFrame:
        """Summarize the captured data's missingness within explicit segments."""
        return missing_by(self._data, group=group, include_group=include_group)

    def plot(self, *, ax: Any = None) -> Any:
        """Plot variable missing percentages on optional Matplotlib Axes.

        Plotting imports Matplotlib only when called and returns the Axes.
        No figures are shown or saved automatically.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install nabularpy[plot] to use optional plotting") from exc
        if ax is None:
            _, ax = plt.subplots()
        summary = self.summary()
        ax.bar([str(variable) for variable in summary["variable"]], summary["percentage"])
        ax.set_ylabel("Missing (%)")
        ax.set_ylim(0, 100)
        return ax


def inspect(data: pd.DataFrame) -> MissingReport:
    """Capture a data snapshot for descriptive missingness exploration."""
    _validate(data)
    snapshot = data.copy(deep=True)
    return MissingReport(snapshot, snapshot.isna())
