"""Compare complete high-cardinality grouped summaries with a slicing recipe."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark import measure
from pandas.testing import assert_frame_equal

from nabularpy import missing_by, missing_summary


def manual(frame: pd.DataFrame) -> pd.DataFrame:
    """Independent per-group slicing and summary recipe, including output assembly."""
    records = []
    for label, segment in frame.groupby("segment", sort=False):
        for summary in missing_summary(segment.drop(columns="segment")).to_dict("records"):
            records.append({"segment": label, **summary, "n_rows": len(segment)})
    return pd.DataFrame(
        records,
        columns=["segment", "variable", "n_rows", "missing", "present", "percentage"],
        dtype=object,
    ).astype({"n_rows": "int64", "missing": "int64", "present": "int64", "percentage": float})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--groups", type=int, default=2_000)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/grouped-results.json"))
    args = parser.parse_args()
    if args.rows < args.groups or args.groups < 1 or args.columns < 1 or args.repeat < 2:
        parser.error("rows must be >=groups, groups/columns >=1 and repeat >=2")
    rng = np.random.default_rng(20261021)
    values = rng.normal(size=(args.rows, args.columns))
    values[rng.random(values.shape) < 0.2] = np.nan
    frame = pd.DataFrame(values, columns=[f"v{column}" for column in range(args.columns)])
    frame.insert(0, "segment", np.arange(args.rows) % args.groups)
    assert_frame_equal(missing_by(frame, group="segment"), manual(frame))
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "input_rows": args.rows,
        "observed_groups": args.groups,
        "measured_columns": args.columns,
        "repeats": args.repeat,
        "manual_group_slicing": measure(lambda: manual(frame), args.repeat),
        "nabularpy": measure(lambda: missing_by(frame, group="segment"), args.repeat),
        "notes": "Full output equivalence checked before timing. Seed20261021; 20% independent missingness. Both measurements include long-form output construction. Separate tracemalloc runs exclude some native allocations and are not process RSS.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
