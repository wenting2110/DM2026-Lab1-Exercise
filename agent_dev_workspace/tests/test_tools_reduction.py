import pytest
import matplotlib.figure
from sklearn.preprocessing import LabelBinarizer
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_reduce_dimensions_tool_pca_known_answer_7():
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

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Assert Known Answer 7 (PCA)
    assert result["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)
    coords_0 = session.artifacts["reduced_coordinates"][0]
    assert coords_0[0] == pytest.approx(2.3669, abs=1e-2)
    assert coords_0[1] == pytest.approx(0.8871, abs=1e-2)

    # Mandatory Plotting Check for PCA
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    assert len(session.pending_figure.axes) > 0


def test_reduce_dimensions_tool_tsne_known_answer_8():
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

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    # Assert Known Answer 8 (t-SNE)
    coords_0 = session.artifacts["reduced_coordinates"][0]
    assert coords_0[0] == pytest.approx(1330.9991, abs=1.0)
    assert coords_0[1] == pytest.approx(32.1039, abs=1.0)

    # Mandatory Plotting Check for t-SNE
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    assert len(session.pending_figure.axes) > 0


def test_reduce_dimensions_tool_umap_known_answer_9():
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

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    # Assert Known Answer 9 (UMAP)
    coords_0 = session.artifacts["reduced_coordinates"][0]
    assert coords_0[0] == pytest.approx(6.6811, abs=1.0)
    assert coords_0[1] == pytest.approx(-6.8218, abs=1.0)

    # Mandatory Plotting Check for UMAP
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    assert len(session.pending_figure.axes) > 0


def test_binarize_labels_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({"column_name": "category_name"})

    assert result["n_samples"] == 8
    assert result["n_classes"] == 2
    assert result["classes"] == ["catA", "catB"]
    assert result["class_mapping"] == {"catA": 0, "catB": 1}

    # Assert artifacts
    assert "label_binarizer" in session.artifacts
    assert isinstance(session.artifacts["label_binarizer"], LabelBinarizer)
    assert "onehot_labels" in session.artifacts
    assert len(session.artifacts["onehot_labels"]) == 8
