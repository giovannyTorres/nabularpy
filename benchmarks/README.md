# Reproducible observed-pattern benchmark

```bash
python benchmarks/benchmark.py --rows 20000 --columns 32 --repeat 5
```

NumPy generator seed 20261021 creates twelve random missingness templates and
repeats them across 20,000 rows and 32 columns. Continuous observed values are
otherwise irrelevant to pattern counting. The script checks complete output
equivalence before measuring either implementation.

The manual reference uses Pandas isna, original missing-label tuples and Python
Counter.most_common. Both implementations count only observed patterns. The
library uses exact packed bit vectors and constructs label tuples once per
observed pattern. This is a narrow pattern-aggregation comparison, not a claim
that Python lacks a structured missingness implementation.

Each workflow warms up once; five independent perf_counter durations follow.
A separate tracemalloc run records peak traced Python allocation, excluding
some native allocations and not measuring process RSS. [results.json](results.json)
contains the latest environment, complete samples, median, minimum, maximum
and memory values, and is authoritative after rerunning.

Repeated templates favor efficient observed-pattern aggregation. Datasets where
almost every row has a unique pattern, very narrow tables, or different dtypes
can behave differently. Results describe one seeded synthetic workload; no
universal runtime or memory advantage is claimed. Cooccurrence and grouped
analysis are separate operations and are not timed by this benchmark.

## High-cardinality grouped summaries

```bash
python benchmarks/grouped.py --rows 20000 --groups 2000 --columns 8 --repeat 5
```

The grouped benchmark uses 2,000 first-observed integer segments and eight
measured columns with independently sampled 20% missingness. It checks the entire
long-form result against a per-group slicing and summary recipe before timing.
Both paths include output construction; the recipe represents the repeated
DataFrame work removed in 0.1.1. The optimized path factorizes keys once and sums
the missingness mask across integer group codes.

[grouped-results.json](grouped-results.json) records repeated timings and separate
traced allocation peaks. This synthetic case emphasizes many small groups;
different group sizes, column widths and dtypes can change the relative benefit.
No timing threshold is asserted in tests.
