import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools


def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's dtm module
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get variance_filter_tool from tools_filtering
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Invoke tool with threshold 0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert Known Answer 5 counts and term sets
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert sorted(result["kept_terms"]) == ["alpha", "beta", "delta", "gamma"]
    assert result["removed_terms"] == ["always"]

    # Assert full_report structure matching contract (first col term, second col variance, sorted descending)
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Check exact expected variances from Known Answer 5
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-3)


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert result["removed_terms"] == ["always"]

    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Check exact expected Pearson r values from Known Answer 6
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert result["removed_terms"] == ["always"]

    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Check exact expected Spearman rho values from Known Answer 6
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
