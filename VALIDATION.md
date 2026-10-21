# Local validation record — nabularpy 0.1.0

Recorded from completed local checks on 2026-10-21 (UTC).
These results are separate from GitHub CI and from public PyPI installation.

| Check | Observed result |
|---|---|
| Editable installation with development dependencies | Passed |
| Pytest and coverage threshold | Passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| Mypy, src/ | Passed |
| Wheel and sdist build | Passed |
| Twine distribution metadata checks | Passed |
| Independent clean wheel and sdist installations and shipped examples | Passed |
| Benchmark execution | Passed |

**Tests:** 40 passed.
**Statement coverage:** 100.00% (106 of 106 measured statements; 0 excluded lines). This is not a branch-coverage claim.

Wheel and sdist were installed separately in fresh virtual environments with declared runtime dependencies.
The shipped examples ran with Python's `-I` flag, isolating imports from the repository checkout.
These installations used local distribution files, not a public-index upload.

## Recorded benchmark

Environment: python: `3.13.5`; pandas: `3.0.6`; platform: `Windows-11-10.0.26200-SP0`.

numpy: **2.5.4**; input_rows: **20,000**; input_columns: **32**; observed_patterns: **12**; repeats: **5**.

The deterministic synthetic input has 20,000 rows and 32 variables with 12 repeated missingness templates, seeded with 20261021; this is a repeated-pattern workload rather than a high-cardinality missingness benchmark. Before timing, observed pattern labels, counts, percentages and n_missing are asserted identical to an independent Python Counter/Pandas reference. Both workflows enumerate observed patterns only. Each warms up once and records five perf_counter durations; traced peak allocation is measured in a separate untimed execution. No performance or missingness-mechanism inference is generalized beyond this workload.

| Workflow | Median seconds | Minimum–maximum seconds | Peak traced bytes |
|---|---:|---:|---:|
| `manual_counter` | 0.073938 | 0.070166–0.077324 | 680,752 |
| `nabularpy` | 0.012420 | 0.011532–0.015221 | 747,130 |

Same output checked before timing. Seed20261021,12repeated random missingness templates. Both enumerate observed patterns only. Separate tracemalloc peak excludes some native allocations and is not process RSS.

The [raw measurement file](benchmarks/results.json) is the source of these numbers.
See [benchmark code](benchmarks/benchmark.py) and its documented methodology for workload details.
`tracemalloc` figures, when present, are traced allocations rather than process RSS.
No universal performance advantage follows from a single workload.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov=nabularpy --cov-report=term-missing --cov-fail-under=95
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m build
python -m twine check dist/*
python scripts/check_artifacts.py nabularpy
python benchmarks/benchmark.py
```

See [RELEASE.md](RELEASE.md) for publication status and the Trusted Publisher configuration.
