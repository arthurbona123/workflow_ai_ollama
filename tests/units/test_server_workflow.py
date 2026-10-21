import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Importa os componentes do nosso servidor
from workflow_ai_python_ollama import (
    app,
    SESSION_MEMORY,
    ContextSummary,
    SearchDecision,
    context_manager_node,
    search_router_node,
    tavily_search_node,
    prompt_builder_node,
    route_after_check,
    normalize_content
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_memory():
    """Reseta o cache em memória antes de cada teste para evitar efeito colateral."""
    SESSION_MEMORY["summary"] = ""
    SESSION_MEMORY["summarized_count"] = 0
    yield

# ==========================================
# TESTES DE UNIDADE DOS NÓS DO LANGGRAPH
# ==========================================

def test_normalize_content():
    assert normalize_content("texto simples") == "texto simples"
    assert normalize_content([{"type": "text", "text": "bloco 1"}, {"text": "bloco 2"}]) == "bloco 1\nbloco 2"
    assert normalize_content(123) == "123"

@patch("server.llm")
def test_context_manager_node_short_history(mock_llm):
    """Garante que históricos curtos (<= 6) não disparam sumarização."""
    msgs = [HumanMessage(content="Oi"), AIMessage(content="Olá")]
    state = {"messages": msgs, "summary": ""}
    
    res = context_manager_node(state)
    assert len(res["messages"]) == 2
    assert res["summary"] == ""
    mock_llm.with_structured_output.assert_not_called()

@patch("server.llm")
def test_context_manager_node_compaction(mock_llm):
    """Testa se históricos > 6 mensagens são resumidos via Pydantic e cortados para as 4 últimas."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ContextSummary(resumo_atualizado="Resumo técnico mockado.")
    mock_llm.with_structured_output.return_value = mock_structured

    msgs = [HumanMessage(content=f"Msg {i}") for i in range(8)]
    state = {"messages": msgs, "summary": ""}

    res = context_manager_node(state)
    assert len(res["messages"]) == 4  # Mantém apenas as 4 mais recentes
    assert res["summary"] == "Resumo técnico mockado."
    assert SESSION_MEMORY["summarized_count"] == 4

@patch("server.llm")
def test_search_router_node_needs_search(mock_llm):
    """Valida a decisão estruturada do roteador quando precisa de busca."""
    mock_decider = MagicMock()
    mock_decider.invoke.return_value = SearchDecision(need_search=True, query="fastapi sse streaming docs")
    mock_llm.with_structured_output.return_value = mock_decider

    state = {"messages": [HumanMessage(content="Como fazer SSE no FastAPI?")], "summary": ""}
    res = search_router_node(state)

    assert res["search_decision"].need_search is True
    assert res["search_decision"].query == "fastapi sse streaming docs"
    assert route_after_check(res) == "web_search"

@patch("server.tavily_client")
def test_tavily_search_node_success(mock_tavily):
    """Garante que o nó do Tavily respeita o contrato WebSearchOutput."""
    mock_tavily.search.return_value = {
        "results": [
            {"title": "FastAPI Docs", "content": "Use StreamingResponse para SSE."}
        ]
    }
    state = {"search_decision": SearchDecision(need_search=True, query="fastapi sse")}
    res = tavily_search_node(state)

    assert res["web_context"].query == "fastapi sse"
    assert "FastAPI Docs" in res["web_context"].resultado_busca
    mock_tavily.search.assert_called_once_with(
        query="fastapi sse",
        search_depth="basic",
        max_results=2,
        include_raw_content=False,
        include_answer=False
    )

def test_prompt_builder_node():
    """Verifica se o prompt final injeta SystemMessage da IDE, Resumo e Web Context na ordem certa."""
    state = {
        "system_prompt_ide": "Você é um assistente.",
        "summary": "O dev usa Arch Linux.",
        "web_context": MagicMock(query="q", resultado_busca="res"),
        "messages": [HumanMessage(content="Pergunta final")]
    }
    res = prompt_builder_node(state)
    final_msgs = res["final_prompt"]

    assert len(final_msgs) == 4
    assert isinstance(final_msgs[0], SystemMessage)
    assert "Arch Linux" in final_msgs[1].content
    assert "Busca Web" in final_msgs[2].content
    assert final_msgs[3].content == "Pergunta final"

# ==========================================
# TESTES DE INTEGRAÇÃO DO ENDPOINT FASTAPI
# ==========================================

@patch("server.app_graph")
@patch("server.llm")
def test_chat_completions_non_streaming(mock_llm, mock_graph):
    """Testa o endpoint /v1/chat/completions com stream=False."""
    mock_graph.invoke.return_value = {"final_prompt": [HumanMessage(content="Oi")]}
    
    async def mock_ainvoke(*args, **kwargs):
        return AIMessage(content="Resposta direta do modelo.")
    
    mock_llm.ainvoke = mock_ainvoke

    payload = {
        "model": "qwen2.5-coder-chat",
        "messages": [{"role": "user", "content": "Oi"}],
        "stream": False
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Resposta direta do modelo."

@patch("server.app_graph")
@patch("server.llm")
def test_chat_completions_streaming_sse(mock_llm, mock_graph):
    """Testa o retorno em Server-Sent Events (SSE) exigido pelo Continue.dev."""
    mock_graph.invoke.return_value = {"final_prompt": [HumanMessage(content="Oi")]}

    async def mock_astream(*args, **kwargs):
        for token in ["def ", "foo():", " pass"]:
            chunk = MagicMock()
            chunk.content = token
            yield chunk

    mock_llm.astream = mock_astream

    payload = {
        "model": "qwen2.5-coder-chat",
        "messages": [{"role": "user", "content": "Crie uma função"}],
        "stream": True
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: [DONE]" in response.text