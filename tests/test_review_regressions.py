"""Independent missingness references and data-preservation regressions."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

import nabularpy as nb


@given(st.lists(st.lists(st.booleans(), min_size=4, max_size=4), max_size=30))
@settings(max_examples=35, deadline=None, derandomize=True)
def test_observed_patterns_match_independent_row_counter(flags):
    names = ["a", "b", "c", "d"]
    frame = pd.DataFrame(
        [[None if missing else 1 for missing in row] for row in flags], columns=names, dtype=object
    )
    expected = Counter(
        tuple(name for name, missing in zip(names, row, strict=True) if missing) for row in flags
    )
    patterns = nb.missing_patterns(frame)
    actual = dict(zip(patterns["pattern"], patterns["count"], strict=True))
    assert actual == expected
    assert patterns["count"].sum() == len(frame)
    assert len(patterns) <= len(frame)


def test_cooccurrence_counts_are_integer_and_match_pairwise_reference():
    frame = pd.DataFrame(
        {"a": [None, None, 1, None], "b": [None, 2, None, None], "c": [None, None, None, None]}
    )
    result = nb.inspect(frame).cooccurrence()
    mask = frame.isna()
    for left in frame.columns:
        for right in frame.columns:
            assert result.loc[left, right] == sum(
                bool(a) and bool(b) for a, b in zip(mask[left], mask[right], strict=True)
            )
    assert result.loc["a", "a"] == 3
    assert result.loc["a", "b"] == 2
    assert all(pd.api.types.is_integer_dtype(dtype) for dtype in result.dtypes)


def test_null_groups_do_not_collide_with_literal_missing_sentinel():
    frame = pd.DataFrame(
        {"region": ["<missing>", None, "<missing>", None], "value": [None, 1, 2, None]}
    )
    result = nb.missing_by(frame, group="region")
    literal = result.loc[result["region"] == "<missing>"].iloc[0]
    null = result.loc[result["region"].isna()].iloc[0]
    assert len(result) == 2
    assert literal["n_rows"] == null["n_rows"] == 2
    assert literal["missing"] == null["missing"] == 1


def test_categorical_group_omits_unused_levels_and_includes_nulls():
    frame = pd.DataFrame(
        {
            "region": pd.Categorical(["N", None, "N"], categories=["N", "unused"]),
            "value": [None, None, 1],
        }
    )
    result = nb.missing_by(frame, group="region")
    assert len(result) == 2
    assert "unused" not in result["region"].tolist()
    null = result.loc[result["region"].isna()].iloc[0]
    assert null["n_rows"] == null["missing"] == 1
    complete = nb.missing_by(frame, group="region", include_group=True)
    group_null = complete.loc[complete["region"].isna() & (complete["variable"] == "region")].iloc[
        0
    ]
    assert group_null["missing"] == group_null["n_rows"] == 1


@pytest.mark.parametrize("add_literal_na", [False, True])
def test_integer_group_ids_are_not_rounded_by_null_group_labels(add_literal_na):
    large = 2**53 + 1
    groups = [large, large + 1, large, None] + (["NA"] if add_literal_na else [])
    values = [None, 1, None, None] + ([5] if add_literal_na else [])
    frame = pd.DataFrame({"segment": pd.Series(groups, dtype=object), "value": values})
    result = nb.missing_by(frame, group="segment")
    populated = result.loc[result["segment"].notna()]
    integer_ids = [int(value) for value in populated["segment"] if not isinstance(value, str)]
    assert integer_ids == [large, large + 1]
    assert populated["n_rows"].tolist() == [2, 1] + ([1] if add_literal_na else [])
    assert populated["missing"].tolist() == [2, 0] + ([0] if add_literal_na else [])
    if add_literal_na:
        assert populated["segment"].iloc[-1] == "NA"


def test_grouped_variable_labels_keep_original_exact_objects():
    large = 2**53 + 1
    frame = pd.DataFrame([["A", 1, None]], columns=pd.Index(["segment", large, None], dtype=object))
    result = nb.missing_by(frame, group="segment")
    assert int(result["variable"].iloc[0]) == large
    assert result["variable"].iloc[1] is None


def test_shadow_and_row_summary_preserve_duplicate_index_and_input():
    frame = pd.DataFrame({"a": [None, 0, False], "b": [1, None, 2]}, index=[9, 9, 1])
    original = frame.copy(deep=True)
    shadow = nb.shadow(frame)
    rows = nb.inspect(frame).row_summary()
    assert shadow.index.tolist() == rows.index.tolist() == [9, 9, 1]
    assert shadow["a_NA"].tolist() == [True, False, False]
    assert shadow["b_NA"].tolist() == [False, True, False]
    assert rows["missing"].tolist() == [1, 1, 0]
    assert rows["present"].tolist() == [1, 1, 2]
    assert_frame_equal(frame, original)


def test_shadow_bound_columns_cannot_overwrite_original_values():
    source = pd.DataFrame({"a": [1], "a_NA": ["original"]})
    with pytest.raises(ValueError):
        nb.shadow(source, bind=True)
    assert source["a_NA"].tolist() == ["original"]


def test_shadow_stringified_column_labels_must_remain_unique():
    source = pd.DataFrame([[1, None]], columns=pd.Index([1, "1"], dtype=object))
    with pytest.raises(ValueError):
        nb.shadow(source)


@pytest.mark.parametrize(
    "operation", [nb.missing_summary, nb.missing_patterns, nb.shadow, nb.inspect]
)
def test_duplicate_raw_columns_rejected(operation):
    with pytest.raises(ValueError):
        operation(pd.DataFrame([[1, None]], columns=["a", "a"]))


def test_zero_column_rows_have_one_observed_empty_pattern():
    frame = pd.DataFrame(index=[5, 5, 9])
    patterns = nb.missing_patterns(frame)
    assert patterns["pattern"].tolist() == [()]
    assert patterns["count"].tolist() == [3]
    assert patterns["n_missing"].tolist() == [0]
    assert patterns["percentage"].tolist() == [100.0]
    rows = nb.inspect(frame).row_summary()
    assert rows["missing"].tolist() == rows["present"].tolist() == [0, 0, 0]
    assert rows["percentage"].tolist() == [0.0, 0.0, 0.0]
    assert nb.missing_summary(frame).empty
    assert nb.inspect(frame).cooccurrence().shape == (0, 0)


def test_zero_row_frame_has_defined_zero_percentages_and_no_patterns():
    frame = pd.DataFrame({"a": pd.Series([], dtype="string"), "b": pd.Series([], dtype="Int64")})
    summary = nb.missing_summary(frame)
    assert summary["missing"].tolist() == summary["present"].tolist() == [0, 0]
    assert summary["percentage"].tolist() == [0.0, 0.0]
    assert nb.missing_patterns(frame).empty
    assert nb.inspect(frame).row_summary().empty
    assert nb.inspect(frame).cooccurrence().to_numpy().sum() == 0


def test_many_columns_only_materialize_observed_patterns():
    frame = pd.DataFrame([[None] * 130, [1] * 130, [None] * 130])
    result = nb.missing_patterns(frame)
    assert len(result) == 2
    assert result["count"].tolist() == [2, 1]
    assert result["n_missing"].tolist() == [130, 0]


def test_report_snapshot_and_returned_frames_are_independent():
    source = pd.DataFrame({"region": ["A", "B"], "value": [None, 1]})
    report = nb.inspect(source)
    source.loc[:, "value"] = 999
    summary = report.summary()
    summary.loc[:, "missing"] = 999
    assert report.summary().loc[report.summary()["variable"] == "value", "missing"].iloc[0] == 1
    assert report.missing_by("region")["missing"].tolist() == [1, 0]


@pytest.mark.parametrize("group", ["variable", "missing", "percentage", "n_rows", "present"])
def test_group_names_cannot_overwrite_summary_metadata(group):
    with pytest.raises(ValueError):
        nb.missing_by(pd.DataFrame({group: ["A"], "x": [None]}), group=group)


def test_optional_plot_runs_with_headless_backend():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    axis = nb.inspect(pd.DataFrame({"a": [None, 1], "b": [None, None]})).plot()
    assert [patch.get_height() for patch in axis.patches] == [50.0, 100.0]
    plt.close(axis.figure)


def test_normal_import_does_not_load_optional_matplotlib():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import nabularpy; assert 'matplotlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
