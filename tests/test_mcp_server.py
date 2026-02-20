"""Tests for the MCP server."""

import json

import pytest
from fastapi.testclient import TestClient

import bookbridge.config as cfg
import bookbridge.mcp_server as mcp_mod
import bookbridge.database as db_mod
from bookbridge.database import init_db, _connect, upsert_book, insert_chunk
from bookbridge.embedder import get_embedder


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_mcp.db"
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(mcp_mod, "DB_PATH", db_path)

    init_db(db_path)
    conn = _connect(db_path)
    monkeypatch.setattr(mcp_mod, "get_db", lambda: conn)

    import bookbridge.embedder as emb_mod
    monkeypatch.setattr(emb_mod, "_embedder", None)

    yield conn
    conn.close()


@pytest.fixture
def client(isolated_db):
    return TestClient(mcp_mod.mcp_app)


@pytest.fixture
def book_id(isolated_db):
    conn = isolated_db
    with conn:
        bid = upsert_book(
            conn,
            {
                "title": "Fluid Dynamics",
                "authors": ["Landau", "Lifshitz"],
                "year": 1959,
                "publisher": "Pergamon",
                "subject_areas": ["physics"],
                "tags": ["fluid"],
            },
        )
        emb = get_embedder()
        text = "The Reynolds number characterizes the flow regime in fluid dynamics."
        emb.fit([text])
        vec = emb.embed(text)
        insert_chunk(conn, bid, 0, text, 10, 10, "Reynolds Number", emb.to_bytes(vec))
    return bid


def _rpc(method, params=None, rpc_id=1):
    return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}}


# ── protocol ──────────────────────────────────────────────────────────────────


def test_initialize(client):
    r = client.post("/mcp", json=_rpc("initialize"))
    assert r.status_code == 200
    data = r.json()
    assert data["result"]["serverInfo"]["name"] == "BookBridge"
    assert "protocolVersion" in data["result"]


def test_tools_list(client):
    r = client.post("/mcp", json=_rpc("tools/list"))
    assert r.status_code == 200
    tools = r.json()["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    expected = {
        "bookbridge_search",
        "bookbridge_retrieve",
        "bookbridge_equations",
        "bookbridge_figures",
        "bookbridge_related",
        "bookbridge_reading_plan",
        "bookbridge_cite",
        "bookbridge_link_activity",
        "bookbridge_list_books",
    }
    assert expected == tool_names


def test_unknown_method(client):
    r = client.post("/mcp", json=_rpc("unknown/method"))
    assert r.status_code == 200
    data = r.json()
    assert "error" in data


def test_unknown_tool(client):
    r = client.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "nonexistent_tool", "arguments": {}}),
    )
    assert r.status_code == 200
    data = r.json()
    assert "error" in data


# ── tool calls ────────────────────────────────────────────────────────────────


def test_bookbridge_list_books(client, book_id):
    r = client.post("/mcp", json=_rpc("tools/call", {"name": "bookbridge_list_books", "arguments": {}}))
    assert r.status_code == 200
    result_text = r.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert len(result["books"]) == 1
    assert result["books"][0]["title"] == "Fluid Dynamics"


def test_bookbridge_search(client, book_id):
    r = client.post(
        "/mcp",
        json=_rpc(
            "tools/call",
            {"name": "bookbridge_search", "arguments": {"query": "Reynolds number fluid"}},
        ),
    )
    assert r.status_code == 200
    result_text = r.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert "results" in result


def test_bookbridge_cite(client, book_id):
    r = client.post(
        "/mcp",
        json=_rpc(
            "tools/call",
            {"name": "bookbridge_cite", "arguments": {"book_id": book_id, "style": "APA"}},
        ),
    )
    assert r.status_code == 200
    result_text = r.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert "citation" in result
    assert "Landau" in result["citation"]


def test_bookbridge_cite_not_found(client):
    r = client.post(
        "/mcp",
        json=_rpc(
            "tools/call",
            {"name": "bookbridge_cite", "arguments": {"book_id": "bad-id", "style": "APA"}},
        ),
    )
    assert r.status_code == 200
    result_text = r.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert "error" in result


def test_bookbridge_reading_plan(client, book_id):
    r = client.post(
        "/mcp",
        json=_rpc(
            "tools/call",
            {
                "name": "bookbridge_reading_plan",
                "arguments": {"topic": "Reynolds number turbulent flow", "goal": "implement solver"},
            },
        ),
    )
    assert r.status_code == 200
    result_text = r.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert "plan" in result
    assert result["topic"] == "Reynolds number turbulent flow"


def test_bookbridge_link_activity(client, book_id):
    r = client.post(
        "/mcp",
        json=_rpc(
            "tools/call",
            {
                "name": "bookbridge_link_activity",
                "arguments": {
                    "activity_type": "file",
                    "activity_id": "simulation.py",
                    "agent_or_process_id": "mcp-agent",
                    "references": [{"book_id": book_id, "reason": "used Reynolds formula"}],
                },
            },
        ),
    )
    assert r.status_code == 200
    result_text = r.json()["result"]["content"][0]["text"]
    result = json.loads(result_text)
    assert result["status"] == "created"


def test_mcp_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sse_endpoint(client):
    r = client.get("/sse")
    assert r.status_code == 200
