import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_tool_zero_terms_survived():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # On sample_fixture.csv, variance filtering keeps 0 terms for catA
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res["patterns"] == []
    assert res["note"] == "no terms survived filtering"


def test_mine_patterns_tool_fpgrowth_known_answer_17():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # catA -> alpha support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # catB -> gamma support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_tool_topk_known_answer_20():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Top-K (k=1) catA -> alpha support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # Top-K (k=1) catB -> gamma support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_tool_maxfpgrowth_known_answer_20():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # MaxFPGrowth returns 0 patterns on single-term database (Known Answer 20)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_catA["patterns"] == []

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_catB["patterns"] == []
