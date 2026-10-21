# Research and scope decision

Reviewed 2026-10-21. Original code only: no R code was copied or translated.

| Capability | R implementation | Python alternatives | Gap | Proposed solution |
|---|---|---|---|---|
| Shadow structures | [naniar](https://github.com/njtierney/naniar) bind_shadow/nabular | Pandas isna and explicit concatenation already supply boolean indicators | A consistent naming/collision policy and report workflow still requires assembly | Boolean shadow table with explicit suffix and binding validation |
| Observed missingness patterns | naniar summaries and pattern-oriented workflows | [pyampute mdPatterns](https://rianneschouten.github.io/pyampute/build/html/pyampute.exploration.html) already supplies structured missingness patterns | No novelty claim for counting patterns; a small dependency footprint and shared deterministic schema are useful | Exact packed observed patterns, count/percentage/label tuples |
| Missingness visualization | naniar plotting functions | [missingno](https://github.com/residentmario/missingno) offers missing-data visualization | Plotting is already well served | Structured outputs first; only a simple optional Matplotlib adapter |
| Column/row counts | naniar variable/case summaries | [Pandas isna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html), [Polars null_count](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.null_count.html) | Counting alone does not differentiate a library | Consistent summaries, segments and cross-variable count matrices |
| Segment associations | naniar tidy workflows | Pandas grouping and missing indicators | Empty/null/category rules and label preservation need careful handling | Observed-only groups including nulls; exact object-valued keys and original labels |
| Statistical testing | [naniar mcar_test](https://naniar.njtierney.com/reference/mcar_test.html) implements Little's test | pyampute also documents MCARTest | This is already implemented elsewhere and requires rigorous assumptions/numerical validation | Omit inferential tests from this descriptive 0.1.0 release |

## Value and evidence

Python already has suitable missingness tools. This package offers a small
Pandas/NumPy API that combines deterministic observed patterns, shadow data,
segment summaries and integer cooccurrence counts without requiring plotting
dependencies. It does not claim to supersede pyampute or missingno. Tests use
independent Python Counter and pairwise references, generated patterns and
wide tables. The benchmark compares equivalent observed-pattern output with a
manual Pandas/Counter recipe, including raw repeated timings and traced memory.

Only observed row bit patterns are packed and counted; no set of 2**p possible
patterns is generated. Cooccurrence is a separate on-demand p-by-p matrix.
It counts jointly absent rows, not correlation or a statistical significance
test. Segment keys use exact Python tuples and an internal identity-only null
marker, avoiding public-string sentinels and numeric group-index rounding.

## Little's MCAR test decision

Both naniar and pyampute document an MCAR test. A new implementation would need
independent validation of estimation, covariance singularities, degrees of
freedom and distributional assumptions. That validation is outside this
release, so no placeholder or unverified test is exposed. Observed patterns
and segment differences alone cannot classify data as MCAR, MAR or MNAR.
This library describes absence and observed association; it makes no causal
or inferential missingness-mechanism claim.

## Licenses and attribution

[naniar DESCRIPTION](https://github.com/njtierney/naniar/blob/master/DESCRIPTION)
declares MIT + license file. [missingno LICENSE](https://github.com/residentmario/missingno/blob/master/LICENSE.md)
is MIT. [pyampute's README](https://github.com/RianneSchouten/pyampute) declares
BSD 3-Clause (see its LICENSE.md). Pandas is BSD-3-Clause, NumPy is BSD-3-Clause,
and Polars is MIT. These references inform behavior and scope only; no source
was reused. nabularpy original code uses MIT and claims no upstream affiliation.

## Naming

On 2026-10-21, `https://pypi.org/pypi/nabularpy/json` returned HTTP 404 and
GitHub's public repository search `nabularpy in:name` returned zero results.
PEP 503 normalization leaves `nabularpy` unchanged. The name credits the nabular
missingness-structure concept without claiming to be an official naniar port.
These checks are not a name reservation or trademark clearance. Recheck before
publication; a missing PyPI page does not guarantee registration.
