from collections import Counter

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

import nabularpy as nb


@given(st.lists(st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans()), max_size=80))
@settings(max_examples=90, derandomize=True, deadline=None)
def test_observed_patterns_and_counts_against_python_reference(rows):
    columns = ["a", "b", "c", "d"]
    frame = pd.DataFrame(
        [[None if missing else 1 for missing in row] for row in rows], columns=columns
    )
    expected = Counter(
        tuple(name for name, missing in zip(columns, row, strict=True) if missing) for row in rows
    )
    patterns = nb.missing_patterns(frame)
    assert dict(zip(patterns["pattern"], patterns["count"], strict=True)) == expected
    assert sum(patterns["count"]) == len(rows)
    assert len(patterns) <= len(rows)
    matrix = nb.inspect(frame).cooccurrence()
    for i, left in enumerate(columns):
        for j, right in enumerate(columns):
            assert matrix.loc[left, right] == sum(row[i] and row[j] for row in rows)


@given(st.lists(st.one_of(st.none(), st.integers()), max_size=100))
@settings(max_examples=70, derandomize=True, deadline=None)
def test_summary_shadow_and_rows_agree(values):
    frame = pd.DataFrame({"value": pd.Series(values, dtype=object)})
    summary = nb.missing_summary(frame)
    shadow = nb.shadow(frame)
    assert summary["missing"].iloc[0] == sum(value is None for value in values)
    assert summary["missing"].iloc[0] == shadow["value_NA"].sum()
    assert nb.inspect(frame).row_summary()["missing"].sum() == summary["missing"].sum()
    assert summary["missing"].iloc[0] + summary["present"].iloc[0] == len(values)
