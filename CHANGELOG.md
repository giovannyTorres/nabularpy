# Changelog

## 0.1.1 — 2026-10-21

- Aggregate `missing_by` once across all observed groups, avoiding a DataFrame slice and summary construction for every group.
- Preserve first-observed group/variable ordering, null and categorical groups, exact large integer labels, nullable values and empty-output schemas.
- Add grouped property/regression tests and a reproducible benchmark that checks complete output equivalence before timing.

## 0.1.0 — 2026-10-21

- Initial independently installable release of nabularpy.
- Typed public API, original implementations, executable examples, tests, and reproducible benchmarks.

Publication state is recorded in RELEASE.md; this version number alone does not imply a PyPI upload.
