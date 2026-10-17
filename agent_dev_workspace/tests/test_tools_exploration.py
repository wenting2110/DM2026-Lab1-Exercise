import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools


def test_cosine_similarity_tool_known_answer_16():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's tools_dtm
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get cosine_similarity_tool
    exploration_tools = make_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer 16 Pair 1: Doc 0 vs Doc 1 (both catA, near-identical text) -> 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 1})
    assert res_0_1["similarity_score"] == pytest.approx(0.9847, abs=1e-3)

    # Known Answer 16 Pair 2: Doc 0 vs Doc 3 (catA vs catB, share only 'always') -> 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 3})
    assert res_0_3["similarity_score"] == pytest.approx(0.0909, abs=1e-3)

    # Known Answer 16 Pair 3: Doc 3 vs Doc 4 (both catB, near-identical text) -> 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc1_index": 3, "doc2_index": 4})
    assert res_3_4["similarity_score"] == pytest.approx(0.9847, abs=1e-3)

    # Known Answer 16 Pair 4: Doc 0 vs Doc 7 (catA vs empty-text row) -> 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 7})
    assert res_0_7["similarity_score"] == 0.0

    # Test raw text vectorization via count_vectorizer artifact (Doc 0 text vs Doc 1 text)
    res_text = cosine_similarity_tool.invoke({
        "doc1_text": "always alpha alpha alpha beta",
        "doc2_text": "always alpha alpha beta"
    })
    assert res_text["similarity_score"] == pytest.approx(0.9847, abs=1e-3)


def test_feature_correlation_matrix_tool_known_answer_19():
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

    exploration_tools = make_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    result = feature_corr_tool.invoke({"top_k_variance": 20})

    # Assert terms sorted descending by variance
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert result["terms"] == expected_terms

    matrix = result["correlation_matrix"]
    terms = result["terms"]

    def get_corr(t1, t2):
        i1 = terms.index(t1)
        i2 = terms.index(t2)
        return matrix[i1][i2]

    # Assert all 10 unique off-diagonal pairs from Known Answer 19
    assert get_corr("alpha", "gamma") == pytest.approx(-0.5718, abs=1e-3)
    assert get_corr("alpha", "delta") == pytest.approx(-0.6882, abs=1e-3)
    assert get_corr("alpha", "beta") == pytest.approx(0.8885, abs=1e-3)
    assert get_corr("alpha", "always") == pytest.approx(0.2601, abs=1e-3)

    assert get_corr("gamma", "delta") == pytest.approx(0.8307, abs=1e-3)
    assert get_corr("gamma", "beta") == pytest.approx(-0.6435, abs=1e-3)
    assert get_corr("gamma", "always") == pytest.approx(0.3140, abs=1e-3)

    assert get_corr("delta", "beta") == pytest.approx(-0.7746, abs=1e-3)
    assert get_corr("delta", "always") == pytest.approx(0.3780, abs=1e-3)

    assert get_corr("beta", "always") == pytest.approx(0.2928, abs=1e-3)

    # Mandatory Plotting Check
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    assert len(session.pending_figure.axes) > 0
