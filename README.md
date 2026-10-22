# nabularpy

**Make missing-data patterns and segment differences inspectable as tables.**

A form export has missing age and income values, but column totals do not show which fields disappear together or whether a form version is affected. nabularpy returns observed patterns, shadow columns and segment summaries.

## Try it in two minutes

Requires Python 3.11+ with pandas 2.2+. Install the [published release](https://pypi.org/project/nabularpy/0.1.1/):

```sh
python -m pip install "nabularpy==0.1.1"
```

Copy this complete example into a Python session or save it as `demo.py` and run `python demo.py`:

```python
import pandas as pd

import nabularpy as nb

data = pd.DataFrame(
    {"region": ["N", "N", "S"], "age": [30, None, None], "income": [100, None, 200]}
)
report = nb.inspect(data)
print(report.summary())
print(report.patterns())
print(report.cooccurrence())
assert report.cooccurrence().loc["age", "income"] == 1
assert nb.missing_by(data, group="region")["missing"].tolist() == [1, 1, 1, 0]
```

The example finds **one jointly missing age/income row** and retains all three observed missingness patterns.

## What you gain

A consistent structured workflow combines deterministic observed patterns, shadow indicators, null-inclusive segments and integer cooccurrence counts. Plotting dependencies remain optional.

| Existing approach | What this package adds | Tradeoff |
|---|---|---|
| Pandas `isna` and grouping provide primitives; missingno offers richer plots and pyampute provides additional missingness tools. | A shared report interface, row summaries, explicit label/collision policies, observed/null segment handling and optional plotting. | Mask snapshots require O(rows × columns) memory and a requested cooccurrence matrix requires O(columns²). A simple `isna().sum()` is enough for column totals alone. |

[Run the side-by-side comparison](docs/comparison.md): both implementations are included and their shared outputs are asserted equal. [naniar](https://naniar.njtierney.com/) inspires shadow structures and tidy missingness summaries. These are descriptive workflows. Little's MCAR test and other inferential statistics are outside this release.

## Solve a real workflow problem

These small, synthetic operational fixtures make the workflows reproducible without external services or private records. Each recipe includes expected results and a failure or interpretation boundary.

| Recipe | Verified outcome |
|---|---|
| [Find a form-version data-collection issue](docs/cookbook.md#compare-form-versions) | Compare per-field percentages by version; verify 50% missing income in v1 and retain the unknown-version group. |
| [Identify fields that disappear together](docs/cookbook.md#discover-jointly-missing-fields) | Inspect the three observed patterns and verify the joint count is 1. |
| [Compare observed records using shadow indicators](docs/cookbook.md#compare-observed-age) | Bind boolean missingness indicators and compare observed mean ages: 20 with income present, 35 with income absent. |

From a checkout, after installing the package:

```sh
python examples/comparison.py
python examples/cookbook.py
```

## Measured performance, accuracy and usability

Recorded on Windows-11-10.0.26200-SP0, CPython 3.13.5, with five rotating timed rounds. At the largest measured input size (20,000; see workload definition):

| Compared operation | Baseline median ms | Package median ms | Package / baseline |
|---|---:|---:|---:|
| Package / manual recipe | 68.223 | 9.628 | 0.14x |

Ratio = package median / baseline median; above 1 means higher latency. The package performs the extra diagnostic or provenance work described above. These single-host synthetic results have no universal speed guarantee. Allocation measurements and the smaller scales are in the full table.

[Measured table](benchmarks/RESULTS.md) · [Raw repeated samples](benchmarks/latest.json) · [Reproduction instructions](benchmarks/README.md)

Accuracy means agreement on the stated fixture outputs: Complete pattern frames are compared, including counts, percentages, original labels, missing-variable counts and tie order. Code-size counts are an inspectable usability proxy; no developer-time study was performed.

## Scope and limits

- Missingness follows pandas.isna; empty strings, the literal "NA" and infinity remain values.
- Inputs need unique flat column labels. Shadow name collisions raise an explicit error.
- Only observed patterns are counted. High-dimensional inputs can still have many distinct observed patterns.
- Observed associations do not establish a missingness mechanism, causality or statistical significance.

## Documentation

- [Start here](docs/index.md): choose a workflow and find the relevant guide.
- [Comparison and alternatives](docs/comparison.md): full code, shared behavior and added outputs.
- [Cookbook](docs/cookbook.md): three complete problem-solving recipes.
- [API and behavioral contracts](docs/api.md): exact semantics and errors.
- [Benchmark methodology](benchmarks/README.md): inputs, correctness checks, environment and limits.
- [Research and upstream credit](RESEARCH.md): R inspiration, existing Python tools and scope decisions.

## Develop and validate

```sh
python -m pip install -e ".[dev]"
python scripts/check_docs.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

MIT licensed. Contributions should include a reproducing example and the expected behavior. See [CONTRIBUTING.md](CONTRIBUTING.md).
