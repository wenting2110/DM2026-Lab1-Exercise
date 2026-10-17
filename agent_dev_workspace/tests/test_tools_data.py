import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
    # Initialize single session
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the sample dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get check_missing_tool from tools_data
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Invoke tool on "text" column
    result = check_missing_tool.invoke({"column_name": "text"})

    # Assert Known Answer 2 from TEST_FIXTURE.md
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["missing_pct"] == pytest.approx(12.5, abs=1e-3)
    assert result["missing_indices"] == [7]


def test_tokenize_tool_known_answer_10():
    # Independent session for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({"column_name": "text", "output_key": "unigrams"})

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert session.artifacts["unigrams"] == expected_tokens
    assert result["n_documents"] == 8
    assert result["total_tokens"] == 27
    assert result["unique_unigrams"] == 5
    assert result["avg_tokens_per_doc"] == pytest.approx(3.375, abs=1e-3)


def test_list_files_tool_known_answer_11():
    # Independent session
    session = SessionState()

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Invoke list_files_tool on "newdataset" directory
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    # Assert Known Answer 11 from TEST_FIXTURE.md
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
    assert result["n_directories"] == 0
    assert result["n_files"] == 1


def test_check_duplicates_tool_known_answer_3():
    # Test 1: Inspect duplicates without dropping (drop=False)
    session_inspect = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session_inspect)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools_inspect = make_tools(session_inspect)
    check_duplicates_tool = next(t for t in data_tools_inspect if t.name == "check_duplicates_tool")

    result_inspect = check_duplicates_tool.invoke({"subset": ["text"], "drop": False})

    assert result_inspect["duplicate_count"] == 1
    assert result_inspect["duplicate_indices"] == [5, 6]
    assert result_inspect["rows_remaining"] == 8

    # Test 2: Drop duplicates with keep=False (drop=True)
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")

    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools_drop = make_tools(session_drop)
    check_duplicates_tool_drop = next(t for t in data_tools_drop if t.name == "check_duplicates_tool")

    result_drop = check_duplicates_tool_drop.invoke({"subset": ["text"], "drop": True})

    assert result_drop["duplicate_count"] == 1
    assert result_drop["rows_remaining"] == 6
    assert len(session_drop.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Assert Known Answer 13 from TEST_FIXTURE.md
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["sampled_rows"] == 4
    assert len(session.dataframe) == 4


def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({"text_column": "text", "category_column": "category_name"})

    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    cat_stats = result["per_category_stats"]
    assert "catA" in cat_stats
    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["max"] == 29

    assert "catB" in cat_stats
    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["max"] == 30

    # Verify pending_figure plotting requirement
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    assert len(session.pending_figure.axes) > 0


def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test default peek (3 rows)
    res_default = inspect_data_tool.invoke({"n_rows": 3})
    assert res_default["total_rows"] == 8
    assert res_default["inspected_rows"] == 3
    assert "text" in res_default["columns"]
    assert "category_name" in res_default["data"][0]

    # Test column subset and text truncation safeguard
    res_trunc = inspect_data_tool.invoke({"columns": ["text"], "n_rows": 1, "max_char_len": 10})
    assert res_trunc["inspected_rows"] == 1
    assert res_trunc["data"][0]["text"] == "always alp..."
