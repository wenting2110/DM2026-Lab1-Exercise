import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # Single session reused for both loading and DTM building
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the test fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get build_dtm_tool from student's tools_dtm module
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Assert vocabulary match (Known Answer 4)
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert sorted(session.feature_names) == expected_vocab

    # Assert matrix dimensions and sparsity (Known Answer 4)
    assert session.feature_matrix.shape == (8, 5)
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Assert artifact state
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    result = term_frequency_tool.invoke({})

    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    assert result["term_frequencies"] == expected_frequencies


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Full matrix request (exceeds available 8x5)
    result_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]

    assert result_full["n_documents"] == 8
    assert result_full["n_terms"] == 5
    assert result_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # 3x2 corner slice request
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert result_slice["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
