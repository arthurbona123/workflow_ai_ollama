import os
import json
import time
import uuid
from typing import List, TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from tavily import TavilyClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# ==========================================
# 1. CONFIGURAÇÃO DOS CLIENTES (OLLAMA + TAVILY)
# ==========================================
os.environ["OPENAI_API_KEY"] = "ollama"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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") # colocar no .env depois

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    model="qwen2.5-chat:1.5b",
    temperature=0.2
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Cache leve em memória para persistir o resumo entre chamadas stateless do Continue.dev
SESSION_MEMORY = {
    "summary": "",
    "summarized_count": 0
}

# ==========================================
# 2. SCHEMAS PYDANTIC (LANGGRAPH & FASTAPI)
# ==========================================
class ContextSummary(BaseModel):
    resumo_atualizado: str = Field(
        description="Resumo técnico conciso contendo decisões de arquitetura, arquivos alterados e contexto acumulado."
    )

class SearchDecision(BaseModel):
    need_search: bool = Field(
        description="True APENAS se exigir documentação externa recente, erro específico ou dados em tempo real."
    )
    query: Optional[str] = Field(
        default=None,
        description="Query de busca curta e técnica otimizada para motores de busca."
    )

class WebSearchOutput(BaseModel):
    query: str = Field(description="Query submetida ao buscador.")
    resultado_busca: str = Field(description="Snippets consolidados (top_k).")

# Schemas de compatibilidade com a API da OpenAI (esperados pelo Continue.dev)
class OpenAIMessage(BaseModel):
    role: str
    content: Any  # Pode vir como str ou lista de blocos (multimodal/contexto do Continue)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None

# ==========================================
# 3. DEFINIÇÃO DO ESTADO E NÓS DO LANGGRAPH
# ==========================================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    system_prompt_ide: str
    summary: str
    search_decision: Optional[SearchDecision]
    web_context: Optional[WebSearchOutput]
    final_prompt: List[BaseMessage]

def context_manager_node(state: AgentState):
    messages = state.get("messages", [])
    total_msgs = len(messages)
    
    # Se o usuário clicou em "New Session" no Continue, reseta o cache de resumo
    if total_msgs <= 2:
        SESSION_MEMORY["summary"] = ""
        SESSION_MEMORY["summarized_count"] = 0

    current_summary = SESSION_MEMORY["summary"]
    summarized_count = SESSION_MEMORY["summarized_count"]

    # Se passar de 6 mensagens (User/AI), compacta as antigas e preserva as 4 mais recentes
    if total_msgs > 6:
        cutoff = total_msgs - 4
        new_to_summarize = messages[summarized_count:cutoff]
        recent_messages = messages[-4:]
        
        if new_to_summarize:
            history_text = "\n".join(
                [f"{'Dev' if isinstance(m, HumanMessage) else 'IA'}: {m.content[:800]}" for m in new_to_summarize]
            )
            summary_prompt = (
                f"Resumo anterior: {current_summary if current_summary else 'Nenhum'}\n\n"
                f"Novas interações:\n{history_text}\n\n"
                "Instrução: Atualize o resumo técnico mantendo nomes de funções, decisões e problemas atuais. Seja direto."
            )
            try:
                structured_llm = llm.with_structured_output(ContextSummary)
                res = structured_llm.invoke([HumanMessage(content=summary_prompt)])
                current_summary = res.resumo_atualizado
            except Exception:
                res = llm.invoke([HumanMessage(content=summary_prompt + "\nRetorne APENAS o resumo.")])
                current_summary = res.content.strip()
                
            SESSION_MEMORY["summary"] = current_summary
            SESSION_MEMORY["summarized_count"] = cutoff
            print(f"[LangGraph Proxy] Histórico comprimido ({len(current_summary)} chars). Ativas: {len(recent_messages)}")
            
        return {"messages": recent_messages, "summary": current_summary, "web_context": None}
        
    return {"messages": messages, "summary": current_summary, "web_context": None}

def search_router_node(state: AgentState):
    last_user_msg = state["messages"][-1].content
    summary = state.get("summary", "")
    
    # Limita a leitura aos últimos 1000 caracteres para focar na pergunta real do dev (ignorando anexos gigantes do @file)
    clean_question = last_user_msg[-1000:] if len(last_user_msg) > 1000 else last_user_msg
    
    router_prompt = (
        f"Contexto prévio: {summary}\n"
        f"Pedido do desenvolvedor: {clean_question}\n\n"
        "Precisa consultar documentação atualizada ou busca web externa para responder? Se sim, crie uma query curta."
    )
    
    try:
        decider_llm = llm.with_structured_output(SearchDecision)
        decision = decider_llm.invoke([HumanMessage(content=router_prompt)])
    except Exception:
        decision = SearchDecision(need_search=False, query=None)
        
    return {"search_decision": decision}

def tavily_search_node(state: AgentState):
    query = state["search_decision"].query
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=2,              # top_k = 2 para proteger o contexto do modelo
            include_raw_content=False,
            include_answer=False
        )
        results = response.get("results", [])
        consolidated = "\n".join([f"- [{r.get('title')}]: {r.get('content')}" for r in results]) if results else "Sem resultados."
    except Exception as e:
        consolidated = f"Erro na busca: {str(e)}"
        
    search_output = WebSearchOutput(query=query, resultado_busca=consolidated)
    print(f"[Tavily] Busca concluída para: '{query}'")
    return {"web_context": search_output}

def prompt_builder_node(state: AgentState):
    """Monta a lista enxuta de mensagens que será enviada ao Ollama (com ou sem streaming)."""
    prompt_messages = []
    
    if state.get("system_prompt_ide"):
        prompt_messages.append(SystemMessage(content=state["system_prompt_ide"]))
        
    if state.get("summary"):
        prompt_messages.append(SystemMessage(content=f"Resumo comprimido do histórico:\n{state['summary']}"))
        
    if state.get("web_context"):
        wc = state["web_context"]
        prompt_messages.append(
            SystemMessage(content=f"Busca Web (Query: {wc.query}):\n{wc.resultado_busca}")
        )
        
    prompt_messages.extend(state["messages"])
    return {"final_prompt": prompt_messages}

def route_after_check(state: AgentState):
    decision = state.get("search_decision")
    if decision and decision.need_search and decision.query:
        return "web_search"
    return "prompt_builder"

# Compilação do Grafo Pré-Inferência
workflow = StateGraph(AgentState)
workflow.add_node("memory_manager", context_manager_node)
workflow.add_node("search_router", search_router_node)
workflow.add_node("web_search", tavily_search_node)
workflow.add_node("prompt_builder", prompt_builder_node)

workflow.set_entry_point("memory_manager")
workflow.add_edge("memory_manager", "search_router")
workflow.add_conditional_edges("search_router", route_after_check, {
    "web_search": "web_search",
    "prompt_builder": "prompt_builder"
})
workflow.add_edge("web_search", "prompt_builder")
workflow.add_edge("prompt_builder", END)

app_graph = workflow.compile()

# ==========================================
# 4. ENDPOINT FASTAPI (OPENAI-COMPATIBLE)
# ==========================================
app = FastAPI(title="LangGraph Proxy for Continue.dev")

def normalize_content(content: Any) -> str:
    """Extrai texto caso o Continue envie array de objetos no campo content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([item.get("text", "") for item in content if isinstance(item, dict) and "text" in item])
    return str(content)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    lc_messages = []
    system_prompt_ide = ""
    
    # Separa a SystemMessage da IDE do histórico conversacional
    for m in request.messages:
        text = normalize_content(m.content)
        if m.role == "system":
            system_prompt_ide += text + "\n"
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=text))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=text))

    # 1. Roda o Grafo (Poda + Sumarização + Busca Tavily)
    initial_state: AgentState = {
        "messages": lc_messages,
        "system_prompt_ide": system_prompt_ide.strip(),
        "summary": "",
        "search_decision": None,
        "web_context": None,
        "final_prompt": []
    }
    
    processed_state = app_graph.invoke(initial_state)
    final_prompt = processed_state["final_prompt"]
    
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # 2. Retorno em Streaming SSE (Exigido pelo Continue.dev para UI fluida e Apply)
    if request.stream:
        async def sse_generator():
            async for chunk in llm.astream(final_prompt):
                if chunk.content:
                    payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk.content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            
            # Pacote de fechamento da stream
            end_payload = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(end_payload)}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # 3. Fallback caso stream=False
    response = await llm.ainvoke(final_prompt)
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_time,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.content},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    print("=== Iniciando Proxy LangGraph na porta 8000 ===")
    uvicorn.run(app, host="127.0.0.1", port=8000)