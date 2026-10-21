# nabularpy

Turn missingness into inspectable data: observed patterns, boolean shadow
columns, per-segment summaries and integer cooccurrence counts. Plotting is
optional; structured tables are the primary output.

```python
import pandas as pd
import nabularpy as nb

data = pd.DataFrame({"age": [30, None, None], "income": [100, None, 200]})
report = nb.inspect(data)
print(report.summary())
print(report.patterns())
print(report.cooccurrence())
```

## Install

Python 3.11+, Pandas 2.2+ and NumPy 1.26+ are required. Until the initial PyPI
publication is verified, install from a checkout:

```bash
python -m pip install .
# After the first verified PyPI release:
python -m pip install nabularpy
# Optional plotting:
python -m pip install "nabularpy[plot]"
```

There are no runtime services, telemetry or sibling-project dependencies.
Importing the package does not import Matplotlib.

## API and output schemas

| API | Output |
|---|---|
| `missing_summary(data)` | `variable`, `missing`, `present`, `percentage` per column |
| `missing_patterns(data)` | `pattern`, `count`, `percentage`, `n_missing` for observed patterns |
| `shadow(data, suffix="_NA", bind=False)` | Boolean indicators; True means missing |
| `missing_by(data, group="region", include_group=False)` | Group fields, `variable`, `n_rows`, `missing`, `present`, `percentage` |
| `inspect(data)` | Snapshot `MissingReport` |
| `report.summary()` / `report.patterns()` | Same structured column/pattern summaries |
| `report.row_summary()` | `missing`, `present`, `percentage` per row; original index preserved |
| `report.cooccurrence()` | Integer jointly missing counts for every variable pair |
| `report.missing_by(group, include_group=False)` | Segments from the captured snapshot |
| `report.plot(ax=None)` | Optional column-percentage bar plot; returns Matplotlib Axes |

Missingness follows Pandas `isna`: None, NaN, pandas.NA and NaT count according
to their dtype semantics. Empty strings, literal `"NA"` and infinity are values;
this library does not silently reinterpret them as missing. Inputs require
unique flat column labels. Nonstring labels are retained in structured outputs;
duplicate or MultiIndex column labels raise `MissingDataError`.

The report captures data and mask snapshots. Changing original frames or
returned result tables does not change later report summaries. No operation
imputes, deletes or changes input values. Avoid concurrent mutation or editing
mutable objects inside cells while preparing a report.

## Observed patterns and count association

`pattern` is a tuple of absent original column labels in input column order;
`()` denotes a complete record. Rows sort by descending count, with ties ordered
by first observed occurrence. Percentages use total input rows. Only observed
row patterns are stored using exact packed bit vectors; the implementation
never enumerates 2**p theoretical combinations. Packed keys are exact bytes,
not probabilistic hashes. The mask costs O(n×p), and the number of returned
patterns is bounded by the number of rows.

`cooccurrence()` counts rows in which both variables are absent. Its diagonal
equals per-variable missing counts, and its values are integer counts rather
than boolean dot products. The output uses O(p²) space and is computed only
when requested. Cooccurrence counts are descriptive association, not correlation
coefficients or hypothesis-test results.

Empty inputs have zero counts/percentages and no observed patterns. A table
with n>0 rows and zero columns has one complete pattern `()` with count n;
row missing percentages are defined as zero. These conventions avoid undefined
percentages and preserve record counts.

## Shadows and segments

```python
bound = nb.shadow(data, bind=True)
print(bound.groupby("income_NA")["age"].mean())
```

This lets an analyst compare observed characteristics of records with and
without a missing variable. The comparison describes observed records; it does
not establish a cause for missingness.

Shadow names use the original label's string representation plus a nonempty
suffix. Generated-name duplicates are rejected, as are collisions with original
columns when `bind=True`. Binding preserves the original index, including
duplicate labels, and all original data columns. Choose another suffix or
rename ambiguous labels explicitly.

`missing_by` accepts a string or sequence of existing string column names.
It retains null-valued groups, includes only observed categorical combinations,
and preserves first-observed group order. No public sentinel string replaces
nulls, so a literal `"<missing>"` remains a separate group. Group identity and
labels preserve exact object-backed large integers. Group fields are excluded
from measured variables unless `include_group=True`.

Grouping names colliding with `variable`, `n_rows`, `missing`, `present` or
`percentage` raise an explicit error rather than overwrite metadata. Group
values must be hashable. A table containing only excluded group columns returns
an empty long-form summary.

## Optional visualization and statistical scope

With the `plot` extra installed, `report.plot()` draws missing percentages.
Matplotlib is imported lazily; a supplied `ax` is reused. No figure is displayed
or saved automatically. All analysis works without plotting dependencies.

Little's MCAR test is deliberately omitted from 0.1.0 because a new numerical
implementation has not been rigorously verified against independent references.
[naniar](https://naniar.njtierney.com/reference/mcar_test.html) and
[pyampute](https://rianneschouten.github.io/pyampute/build/html/pyampute.exploration.html)
already document such tests. Pattern counts and observed segment differences
alone cannot classify data as MCAR, MAR or MNAR. This library provides
description and observed association, without causal or inferential claims.

## Alternatives, limitations and examples

[naniar](https://github.com/njtierney/naniar) inspired shadow structures and tidy
missingness summaries. [missingno](https://github.com/residentmario/missingno)
already provides richer Python missing-data plots, and pyampute already analyzes
patterns. Pandas and Polars supply missingness primitives. nabularpy combines
the structured workflow, deterministic policies and explicit label preservation
in a small API; see [RESEARCH.md](RESEARCH.md). It claims no feature novelty or
affiliation with those projects.

This release supports in-memory Pandas data, with no Polars adapter, imputation,
inference, automatic sentinel conversion or persistent report storage. Dense
masks and pairwise matrices still require memory for large tables. See
[examples/quickstart.py](examples/quickstart.py) and
[examples/segment_association.py](examples/segment_association.py) for executable
workflows, including null-valued segments.

## Validation and benchmark

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov=nabularpy --cov-report=term-missing
python -m ruff check .
python -m mypy src
python examples/quickstart.py
python examples/segment_association.py
python benchmarks/benchmark.py --rows 20000 --columns 32 --repeat 5
```

Tests include independent Counter/pairwise references, generated patterns,
wide/empty tables, categorical and null groups, exact large integer labels,
shadow collisions, snapshot preservation and headless optional plotting.
[Raw measurements](benchmarks/results.json) compare equivalent pattern output
with a manual Pandas/Counter recipe. Five repetitions follow warmup; memory is
measured separately with tracemalloc. See [methodology](benchmarks/README.md)
for limitations and environment. No universal performance advantage is claimed.

See [CONTRIBUTING.md](CONTRIBUTING.md). Original code is MIT licensed. For 0.2.0,
priorities are bounded pattern output, conditional structured comparisons and
streamed mask aggregation; inferential tests require a separate validation plan.
