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
