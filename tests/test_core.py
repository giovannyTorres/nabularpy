import builtins

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import nabularpy as nb


def test_summary_and_report_snapshot():
    raw = pd.DataFrame({"a": [1, None, None], "b": [None, 2, None]})
    report = nb.inspect(raw)
    summary = report.summary()
    assert summary["variable"].tolist() == ["a", "b"]
    assert summary["missing"].tolist() == [2, 2]
    assert summary["present"].tolist() == [1, 1]
    assert summary["percentage"].tolist() == pytest.approx([200 / 3, 200 / 3])
    assert_frame_equal(summary, nb.missing_summary(raw))
    raw.loc[:, "a"] = 0
    summary.loc[:, "missing"] = 0
    assert report.summary()["missing"].tolist() == [2, 2]
    assert report.row_summary()["missing"].tolist() == [1, 1, 2]
    assert report.cooccurrence().to_numpy().tolist() == [[2, 1], [1, 2]]


def test_patterns_order_complete_pattern_and_optional_values():
    raw = pd.DataFrame({"a": [None, 1, None, 2, 3], "b": [1, None, 2, 3, None]})
    patterns = nb.missing_patterns(raw)
    assert patterns["pattern"].tolist() == [("a",), ("b",), ()]
    assert patterns["count"].tolist() == [2, 2, 1]
    assert patterns["n_missing"].tolist() == [1, 1, 0]
    assert patterns["percentage"].tolist() == [40, 40, 20]
    assert_frame_equal(patterns, nb.inspect(raw).patterns())


def test_empty_and_all_missing_tables():
    empty = pd.DataFrame({"a": pd.Series([], dtype="Int64"), "b": pd.Series([], dtype="string")})
    assert nb.missing_summary(empty)["percentage"].tolist() == [0.0, 0.0]
    assert nb.missing_patterns(empty).empty
    assert nb.inspect(empty).row_summary().empty
    assert nb.inspect(empty).cooccurrence().to_numpy().tolist() == [[0, 0], [0, 0]]
    no_columns = pd.DataFrame(index=["same", "same", "same"])
    assert nb.missing_patterns(no_columns)["pattern"].tolist() == [()]
    assert nb.missing_patterns(no_columns)["count"].tolist() == [3]
    assert nb.missing_summary(no_columns).empty
    assert nb.inspect(no_columns).row_summary()["percentage"].tolist() == [0.0, 0.0, 0.0]
    assert nb.inspect(no_columns).cooccurrence().empty
    all_null = pd.DataFrame({"a": [pd.NA] * 3, "b": [pd.NaT] * 3})
    assert nb.missing_patterns(all_null)["pattern"].tolist() == [("a", "b")]
    assert nb.inspect(all_null).cooccurrence().to_numpy().tolist() == [[3, 3], [3, 3]]


def test_shadow_no_mutation_bind_and_label_policy():
    raw = pd.DataFrame({"value": [1, None]}, index=["duplicate", "duplicate"])
    before = raw.copy(deep=True)
    shadow = nb.shadow(raw)
    assert shadow["value_NA"].tolist() == [False, True]
    assert shadow.index.tolist() == raw.index.tolist()
    assert shadow.dtypes.tolist() == [bool]
    bound = nb.shadow(raw, suffix="_missing", bind=True)
    assert bound.columns.tolist() == ["value", "value_missing"]
    assert_frame_equal(bound[["value"]], raw)
    assert_frame_equal(raw, before)
    with pytest.raises(nb.MissingDataError, match="suffix"):
        nb.shadow(raw, suffix="")
    with pytest.raises(nb.MissingDataError, match="collide"):
        nb.shadow(pd.DataFrame([[1, None]], columns=[1, "1"]))
    with pytest.raises(nb.MissingDataError, match="collide"):
        nb.shadow(pd.DataFrame({"a": [1], "a_NA": [None]}), bind=True)


def test_grouped_multiple_columns_and_null_segments():
    raw = pd.DataFrame(
        {"region": ["N", "N", None, "<missing>"], "period": [1, 1, 2, 2], "a": [1, None, None, 3]}
    )
    result = nb.missing_by(raw, group=["region", "period"])
    assert result["n_rows"].tolist() == [2, 1, 1]
    assert result["missing"].tolist() == [1, 1, 0]
    assert result["percentage"].tolist() == [50, 100, 0]
    assert pd.isna(result["region"].iloc[1])
    assert result["region"].iloc[2] == "<missing>"
    included = nb.inspect(raw).missing_by("region", include_group=True)
    assert included["variable"].tolist() == ["region", "period", "a"] * 3
    assert nb.missing_by(raw[["region"]], group="region").empty
    assert nb.missing_by(raw.iloc[:0], group="region").empty


@pytest.mark.parametrize("group", [[], ["region", "region"], [1], ["absent"]])
def test_invalid_group_rules(group):
    with pytest.raises(nb.MissingDataError):
        nb.missing_by(pd.DataFrame({"region": ["N"], "a": [None]}), group=group)


def test_invalid_labels_inputs_and_group_values():
    for function in [nb.inspect, nb.missing_summary, nb.missing_patterns, nb.shadow]:
        with pytest.raises(TypeError):
            function([1, 2])
        with pytest.raises(nb.MissingDataError, match="flat"):
            function(pd.DataFrame([[1]], columns=pd.MultiIndex.from_tuples([("a", "b")])))
    with pytest.raises(nb.MissingDataError, match="metadata"):
        nb.missing_by(pd.DataFrame({"variable": [1], "a": [None]}), group="variable")
    with pytest.raises(nb.MissingDataError, match="hashable"):
        nb.missing_by(pd.DataFrame({"group": [[1], [2]], "a": [1, None]}), group="group")


def test_nonstring_labels_supported_without_guessing_missing_values():
    frame = pd.DataFrame([[None, "NA", "", np.inf]], columns=[1, "text", "empty", "infinity"])
    assert nb.missing_summary(frame)["missing"].tolist() == [1, 0, 0, 0]
    assert nb.missing_patterns(frame)["pattern"].tolist() == [(1,)]


def test_plot_without_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def import_without_plot(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("unavailable optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_plot)
    with pytest.raises(ImportError, match="nabularpy\\[plot\\]"):
        nb.inspect(pd.DataFrame({"a": [None]})).plot()


def test_plot_headless_with_and_without_supplied_axes(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    report = nb.inspect(pd.DataFrame({"a": [1, None], "b": [None, None]}))
    axes = report.plot()
    assert [bar.get_height() for bar in axes.patches] == [50, 100]
    axes.figure.savefig(tmp_path / "missing.png")
    assert (tmp_path / "missing.png").stat().st_size > 0
    figure, supplied = plt.subplots()
    assert report.plot(ax=supplied) is supplied
    assert nb.inspect(pd.DataFrame()).plot(ax=supplied) is supplied
    plt.close(figure)
    plt.close(axes.figure)
