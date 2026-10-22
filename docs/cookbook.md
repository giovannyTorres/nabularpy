# Cookbook: nabularpy

Three common operational problems, with complete Python code and asserted outcomes. The records are **synthetic fixtures modeled on real workflows**, not measurements from a business or real customer data.

Install `python -m pip install "nabularpy==0.1.1"`. From a checkout, run all three with `python examples/cookbook.py`.

<a id="compare-form-versions"></a>
## Find a form-version data-collection issue

**Problem.** A survey export contains v1, v2 and an unknown form version.

**Expected result.** Compare per-field percentages by version; verify 50% missing income in v1 and retain the unknown-version group.

Copy the complete block into a fresh Python file and run it:

```python
import pandas as pd
import nabularpy as nb


def compare_form_versions():
    data = pd.DataFrame(
        {
            "form": ["v1", "v1", "v2", None],
            "age": [30, None, 40, None],
            "income": [100, None, None, 200],
        }
    )
    result = nb.missing_by(data, group="form")
    v1_income = result.loc[(result["form"] == "v1") & (result["variable"] == "income")]
    assert v1_income["percentage"].iloc[0] == 50
    assert result["form"].isna().sum() == 2
    return result


print(compare_form_versions())
```

**Apply it to your data.** The small counts illustrate the workflow. A production comparison needs enough records and domain context.

<a id="discover-jointly-missing-fields"></a>
## Identify fields that disappear together

**Problem.** A column-level summary hides that one row is missing both age and income.

**Expected result.** Inspect the three observed patterns and verify the joint count is 1.

Copy the complete block into a fresh Python file and run it:

```python
import pandas as pd
import nabularpy as nb


def discover_jointly_missing_fields():
    data = pd.DataFrame({"age": [30, None, None], "income": [100, None, 200]})
    report = nb.inspect(data)
    assert report.cooccurrence().loc["age", "income"] == 1
    assert report.patterns()["count"].sum() == 3
    assert report.patterns()["pattern"].tolist() == [(), ("age", "income"), ("age",)]
    return report.patterns()


print(discover_jointly_missing_fields())
```

**Apply it to your data.** Cooccurrence is a count, not a correlation coefficient or significance test.

<a id="compare-observed-age"></a>
## Compare observed records using shadow indicators

**Problem.** Income is absent in two records, and you want to inspect the age distribution of those records.

**Expected result.** Bind boolean missingness indicators and compare observed mean ages: 20 with income present, 35 with income absent.

Copy the complete block into a fresh Python file and run it:

```python
import pandas as pd
import nabularpy as nb


def compare_observed_age():
    data = pd.DataFrame({"age": [20, 30, 40], "income": [100, None, None]})
    bound = nb.shadow(data, bind=True)
    observed = bound.groupby("income_NA")["age"].mean()
    assert observed.loc[False] == 20
    assert observed.loc[True] == 35
    # This describes an association; it does not identify the cause of missingness.
    assert "income_NA" not in data.columns
    return observed


print(compare_observed_age())
```

**Apply it to your data.** This describes the recorded sample. It neither explains why income is missing nor supplies an imputation rule.

[API details and error policies](api.md) · [Comparison with primitives](comparison.md) · [Benchmark evidence](../benchmarks/RESULTS.md)
