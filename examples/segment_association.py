"""Compare observed groups and jointly missing variables, without causal claims."""

import pandas as pd

import nabularpy as nb

data = pd.DataFrame(
    {
        "region": ["N", "N", "S", "S", None],
        "age": [25, 35, 40, 50, 30],
        "income": [100, None, 200, None, None],
        "education": ["college", None, "school", None, "college"],
    }
)
bound = nb.shadow(data, bind=True)
comparison = bound.groupby("income_NA")["age"].agg(["count", "mean"])
assert comparison["count"].sum() == len(data)
print("Observed age differences by income missingness (descriptive):")
print(comparison)
print(nb.missing_by(data, group="region"))
print(nb.inspect(data).patterns())
