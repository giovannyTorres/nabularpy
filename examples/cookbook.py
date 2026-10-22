"""Three descriptive missingness workflows using synthetic survey records."""

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


def discover_jointly_missing_fields():
    data = pd.DataFrame({"age": [30, None, None], "income": [100, None, 200]})
    report = nb.inspect(data)
    assert report.cooccurrence().loc["age", "income"] == 1
    assert report.patterns()["count"].sum() == 3
    assert report.patterns()["pattern"].tolist() == [(), ("age", "income"), ("age",)]
    return report.patterns()


def compare_observed_age():
    data = pd.DataFrame({"age": [20, 30, 40], "income": [100, None, None]})
    bound = nb.shadow(data, bind=True)
    observed = bound.groupby("income_NA")["age"].mean()
    assert observed.loc[False] == 20
    assert observed.loc[True] == 35
    # This describes an association; it does not identify the cause of missingness.
    assert "income_NA" not in data.columns
    return observed


if __name__ == "__main__":
    for recipe in (compare_form_versions, discover_jointly_missing_fields, compare_observed_age):
        print(recipe.__name__, recipe(), sep="\n")
