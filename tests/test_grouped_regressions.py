"""Grouped aggregation contracts independent of per-group DataFrame slicing."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

import nabularpy as nb


@given(
    st.lists(
        st.tuples(
            st.one_of(st.none(), st.integers(min_value=0, max_value=4)),
            st.one_of(st.none(), st.sampled_from(["N", "S", "unused"])),
            st.booleans(),
            st.booleans(),
        ),
        max_size=70,
    ),
    st.booleans(),
)
@settings(max_examples=80, deadline=None, derandomize=True)
def test_multiple_groups_match_independent_row_accumulation(rows, include_group):
    frame = pd.DataFrame(
        [
            [number, region, None if a_missing else 1, None if b_missing else "value"]
            for number, region, a_missing, b_missing in rows
        ],
        columns=["number", "region", "a", "b"],
        dtype=object,
    )
    # Deliberately duplicate indexes: grouping must use row positions.
    frame.index = [i % 3 for i in range(len(frame))]
    original = frame.copy(deep=True)
    variables = frame.columns.tolist() if include_group else ["a", "b"]
    groups = OrderedDict()
    for number, region, a_missing, b_missing in rows:
        counts = groups.setdefault((number, region), [0] * (len(variables) + 1))
        flags = [number is None, region is None, a_missing, b_missing]
        for position, absent in enumerate(flags if include_group else flags[2:]):
            counts[position] += absent
        counts[-1] += 1
    result = nb.missing_by(frame, group=["number", "region"], include_group=include_group)
    assert len(result) == len(groups) * len(variables)
    for position, ((number, region), counts) in enumerate(groups.items()):
        segment = result.iloc[position * len(variables) : (position + 1) * len(variables)]
        for key, label in [("number", number), ("region", region)]:
            assert all(
                pd.isna(value) if label is None else value == label for value in segment[key]
            )
        assert segment["variable"].tolist() == variables
        assert segment["missing"].tolist() == counts[:-1]
        assert segment["n_rows"].tolist() == [counts[-1]] * len(variables)
        assert segment["present"].tolist() == [counts[-1] - count for count in counts[:-1]]
        assert segment["percentage"].tolist() == pytest.approx(
            [count * 100 / counts[-1] for count in counts[:-1]]
        )
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("include_group", [False, True])
def test_nullable_and_categorical_groups_keep_observed_order_and_first_null_label(include_group):
    large = 2**53 + 1
    frame = pd.DataFrame(
        {
            "number": pd.Series([large, None, large + 1, large, pd.NA], dtype=object),
            "region": pd.Categorical(["S", "N", "N", "S", "N"], categories=["N", "S", "unused"]),
            "a": pd.Series([pd.NA, 1, 2, pd.NA, pd.NA], dtype="Int64"),
            "b": pd.Series(["x", pd.NA, "y", "z", pd.NA], dtype="string"),
            "c": pd.Series([True, pd.NA, False, True, False], dtype="boolean"),
        }
    )
    result = nb.missing_by(frame, group=["number", "region"], include_group=include_group)
    width = 5 if include_group else 3
    assert result["number"].iloc[::width].tolist() == [large, None, large + 1]
    assert result["number"].iloc[width] is None
    assert result["region"].iloc[::width].tolist() == ["S", "N", "N"]
    expected = (
        [0, 0, 2, 0, 0, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0]
        if include_group
        else [2, 0, 0, 1, 2, 1, 0, 0, 0]
    )
    assert result["missing"].tolist() == expected
    assert result["n_rows"].iloc[::width].tolist() == [2, 2, 1]
    assert result.dtypes.tolist() == [
        object,
        object,
        object,
        np.dtype("int64"),
        np.dtype("int64"),
        np.dtype("int64"),
        np.dtype(float),
    ]


def test_tuple_group_labels_and_all_empty_output_schemas():
    frame = pd.DataFrame({"group": [("B", 2), ("A", 1), ("B", 2)], "value": [1, None, None]})
    result = nb.missing_by(frame, group="group")
    assert result["group"].tolist() == [("B", 2), ("A", 1)]
    assert result["missing"].tolist() == [1, 1]
    assert result["n_rows"].tolist() == [2, 1]
    for empty in [frame.iloc[:0], frame[["group"]]]:
        result = nb.missing_by(empty, group="group")
        assert result.empty
        assert result.columns.tolist() == [
            "group",
            "variable",
            "n_rows",
            "missing",
            "present",
            "percentage",
        ]
        assert result.dtypes.tolist() == [
            object,
            object,
            np.dtype("int64"),
            np.dtype("int64"),
            np.dtype("int64"),
            np.dtype(float),
        ]


def test_python_equal_numeric_group_values_and_missing_scalars_stay_together():
    frame = pd.DataFrame(
        {
            "group": pd.Series([True, 1, 1.0, False, 0, None, pd.NA, np.nan, pd.NaT], dtype=object),
            "value": [1, None, None, None, 2, None, 3, None, None],
        }
    )
    result = nb.missing_by(frame, group="group")
    assert result["group"].tolist() == [True, False, None]
    assert result["n_rows"].tolist() == [3, 2, 4]
    assert result["missing"].tolist() == [2, 1, 3]
    empty_multi = nb.missing_by(frame.iloc[:0], group=["group", "value"], include_group=True)
    assert empty_multi.empty
