# Comparison: nabularpy and the underlying primitives

A form export has missing age and income values, but column totals do not show which fields disappear together or whether a form version is affected. nabularpy returns observed patterns, shadow columns and segment summaries.

## What is compared

| Dimension | Scope |
|---|---|
| Existing tools | Pandas `isna` and grouping provide primitives; missingno offers richer plots and pyampute provides additional missingness tools. |
| Shared result checked | All observed-pattern output fields: original-label tuples, counts, percentages, number missing and stable tie ordering. |
| Additional package behavior | A shared report interface, row summaries, explicit label/collision policies, observed/null segment handling and optional plotting. |
| Cost / when a manual recipe fits | Mask snapshots require O(rows × columns) memory and a requested cooccurrence matrix requires O(columns²). A simple `isna().sum()` is enough for column totals alone. |

[naniar](https://naniar.njtierney.com/) inspires shadow structures and tidy missingness summaries. These are descriptive workflows. Little's MCAR test and other inferential statistics are outside this release. The comparison below measures a Python workflow; it makes no claim of feature parity or numerical parity with an R package.

## Full executable example

Save this as `comparison.py` after installing the release, or run `python examples/comparison.py` from this repository. The assertions verify the stated shared result.

```python
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
```

## How to choose

Use the package when you need its named diagnostics, explicit policies or provenance repeatedly. Keep the underlying pandas/Python recipe when that smaller shared result meets the job. Review each recipe's data assumptions before applying it to another dtype or layout.

Mask snapshots require O(rows × columns) memory and a requested cooccurrence matrix requires O(columns²). A simple `isna().sum()` is enough for column totals alone.

## Evidence and boundaries

Complete pattern frames are compared, including counts, percentages, original labels, missing-variable counts and tie order.

Twelve repeated templates favor aggregation and do not represent near-unique patterns in a very wide table. Faster results here do not imply a win over every pandas implementation. Timing differences do not establish statistical accuracy or developer productivity. The benchmark checks exact fixture outputs and separately reports code-size counts as a limited usability proxy.

[Actual measured results](../benchmarks/RESULTS.md) · [Reproduce the benchmark](../benchmarks/README.md) · [Recipes](cookbook.md) · [API contracts](api.md)

## Sources and alternatives

- [pandas API documentation](https://pandas.pydata.org/docs/reference/index.html) documents the underlying grouping, masking and reshaping primitives.
- [naniar summaries and shadow matrices](https://naniar.njtierney.com/)
- [missingno upstream](https://github.com/ResidentMario/missingno)
- [pyampute exploration](https://rianneschouten.github.io/pyampute/build/html/pyampute.exploration.html)

Upstream references credit problem ideas and existing tools. This repository contains its own implementation and claims no upstream affiliation.
