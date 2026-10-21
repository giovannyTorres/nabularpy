"""Compare observed pattern aggregation with a manual Python/Pandas recipe."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from nabularpy import missing_patterns


def manual(frame: pd.DataFrame) -> pd.DataFrame:
    """Independent, explicit Python Counter recipe using observed label tuples."""
    columns = frame.columns.tolist()
    counts = Counter(
        tuple(column for column, missing in zip(columns, row, strict=True) if missing)
        for row in frame.isna().itertuples(index=False, name=None)
    )
    ordered = counts.most_common()
    return pd.DataFrame(
        {
            "pattern": pd.Series([pattern for pattern, _ in ordered], dtype=object),
            "count": [count for _, count in ordered],
            "percentage": [count * 100.0 / len(frame) for _, count in ordered],
            "n_missing": [len(pattern) for pattern, _ in ordered],
        }
    )


def measure(operation: Callable[[], object], repeats: int) -> dict[str, object]:
    operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "traced_peak_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    args = parser.parse_args()
    if args.rows < 12 or args.columns < 2 or args.repeat < 2:
        parser.error("rows must be >=12, columns >=2 and repeat >=2")
    rng = np.random.default_rng(20261021)
    templates = rng.random((12, args.columns)) < 0.25
    values = rng.normal(size=(args.rows, args.columns))
    values[templates[np.arange(args.rows) % 12]] = np.nan
    frame = pd.DataFrame(values, columns=[f"v{col}" for col in range(args.columns)])
    assert_frame_equal(missing_patterns(frame), manual(frame))
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "input_rows": args.rows,
        "input_columns": args.columns,
        "observed_patterns": len(missing_patterns(frame)),
        "repeats": args.repeat,
        "manual_counter": measure(lambda: manual(frame), args.repeat),
        "nabularpy": measure(lambda: missing_patterns(frame), args.repeat),
        "notes": "Same output checked before timing. Seed20261021,12repeated random missingness templates. Both enumerate observed patterns only. Separate tracemalloc peak excludes some native allocations and is not process RSS.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
