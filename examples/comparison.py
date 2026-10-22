"""Same observed missingness pattern counts, including stable tie ordering."""

from collections import Counter

import pandas as pd
from pandas.testing import assert_frame_equal

import nabularpy as nb


def baseline(data):
    counts = Counter(
        tuple(col for col, missing in zip(data.columns, row, strict=True) if missing)
        for row in data.isna().itertuples(index=False, name=None)
    )
    ordered = counts.most_common()
    return pd.DataFrame(
        {
            "pattern": pd.Series([p for p, _ in ordered], dtype=object),
            "count": [n for _, n in ordered],
            "percentage": [n * 100.0 / len(data) for _, n in ordered],
            "n_missing": [len(p) for p, _ in ordered],
        }
    )


def library(data):
    return nb.missing_patterns(data)


if __name__ == "__main__":
    data = pd.DataFrame({"age": [30, None, None], "income": [100, None, 200]})
    assert_frame_equal(baseline(data), library(data))
    print("Both return patterns (), ('age','income'), ('age',), each with count 1.")
