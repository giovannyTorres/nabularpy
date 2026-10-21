"""Inspect structured missingness without importing plotting libraries."""

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
