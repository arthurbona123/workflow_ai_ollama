import os
import re
import json
import time
import uuid
from typing import List, TypedDict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURAÇÃO DE AMBIENTE E API
# ==========================================

load_dotenv()

os.environ["OPENAI_API_KEY"] = "lm-studio"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("⚠️ TAVILY_API_KEY não encontrada! Verifique o seu arquivo .env.")

app = FastAPI(title="Continue.dev LangGraph Proxy", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CONFIGURAÇÃO DOS MODELOS (LM STUDIO)
# ==========================================

# Gemma 4 (Principal / Reasoning)
main_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="qwen/qwen3-1.7b",
    temperature=0.2,
    top_p=0.9
)

# LFM 2.5 1.2B (Helper de Memória)
helper_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="liquid/lfm2.5-1.2b",
    temperature=0.3,
    top_p=0.9,
    max_tokens=1024
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Cache de Sessão (Como o Continue envia o histórico todo, usamos isso para não resumir tudo do zero)
SESSION_MEMORY = {
    "summary": "",
    "summarized_count": 0
}

# ==========================================
# 3. SYSTEM PROMPTS DA ARQUITETURA
# ==========================================
MAIN_SYSTEM_PROMPT = (

    "Atue como Engenheiro da Computação Sênior focado em Performance, Back-end e Arch Linux.\n"
    "DIRETRIZES DE LEITURA DO PROMPT:\n"
    "1. [1. MEMÓRIA DE CURTO PRAZO] contém o contexto e diretrizes da IDE. NUNCA responda a perguntas passadas.\n"
    "2. [2. DADOS DE BUSCA NA WEB] traz informações extras e documentações para consulta técnica.\n"
    "3. [3. PROMPT ATUAL DO USUÁRIO] é a sua única tarefa ativa. Responda a ela de forma profunda e técnica.\n"
    "4. Envolva blocos de código sempre dentro de tags Markdown adequadas."
)

HELPER_SYSTEM_PROMPT = (

    "Sua função é sumarizar o histórico de logs técnicos de forma telegráfica em TERCEIRA PESSOA.\n"
    "Capture decisões, arquiteturas, erros e intenções dos códigos discutidos.\n"
    "PROIBIDO responder, PROIBIDO saudações, PROIBIDO usar primeira pessoa ('Eu')."
)

# ==========================================
# 4. SCHEMAS PYDANTIC (API COMPATÍVEL & LANGGRAPH)
# ==========================================
class SearchDecision(BaseModel):

    need_search: bool = Field(description="True se o pedido pedir pesquisa externa ou documentação recente. Caso contrário, False.")

    query: Optional[str] = Field(
        default=None,
        description="Query em inglês (máx 6 palavras). OBRIGATÓRIO incluir o termo/framework principal."
    )

class OpenAIMessage(BaseModel):

    role: str
    content: Any

class ChatCompletionRequest(BaseModel):

    model: str
    messages: List[OpenAIMessage]
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None

# ==========================================
# 5. FUNÇÕES AUXILIARES SINTÁTICAS
# ==========================================
def normalize_content(content: Any) -> str:

    if isinstance(content, str):

        return content
    
    if isinstance(content, list):

        return "\n".join([item.get("text", "") for item in content if isinstance(item, dict) and "text" in item])
    
    return str(content)

def sanitize_user_prompt(raw_text: str) -> str:

    lines = [line.rstrip() for line in raw_text.splitlines()]
    cleaned = []

    for line in lines:

        cleaned.append(" ".join(line.split()) if line and not line.startswith((" ", "\t")) else line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()

def assemble_final_prompt(memoria: str, ide_prompt: str, dados_web: str, prompt_usuario: str) -> List[BaseMessage]:

    texto_memoria = f"[DIRETRIZES DA IDE]\n{ide_prompt}\n\n[HISTÓRICO]\n{memoria}" if ide_prompt else memoria
    
    structured_text = (

        "==================================================\n"
        "[1. MEMÓRIA DE CURTO PRAZO - CONTEXTO PASSADO]\n"
        f"{texto_memoria}\n\n"
        "==================================================\n"
        "[2. DADOS DE BUSCA NA WEB - CONHECIMENTO ADICIONAL]\n"
        f"{dados_web}\n\n"
        "==================================================\n"
        "[3. PROMPT ATUAL DO USUÁRIO - TAREFA A RESPONDER]\n"
        f"{prompt_usuario}\n"
        "=================================================="
    )
    return [

        SystemMessage(content=MAIN_SYSTEM_PROMPT),
        HumanMessage(content=structured_text)
    ]

# ==========================================
# 6. DEFINIÇÃO DO ESTADO E NÓS DO LANGGRAPH
# ==========================================
class AgentState(TypedDict):

    messages: List[BaseMessage]
    system_prompt_ide: str
    short_term_memory: str
    clean_user_prompt: str
    search_decision: Optional[SearchDecision]
    web_search_data: str
    final_prompt: List[BaseMessage]

def memory_and_input_prep_node(state: AgentState):

    messages = state.get("messages", [])
    total_msgs = len(messages)
    
    # Se for uma nova sessão no Continue.dev (User msg única ou poucas)
    if total_msgs <= 2:

        SESSION_MEMORY["summary"] = ""
        SESSION_MEMORY["summarized_count"] = 0

    current_summary = SESSION_MEMORY["summary"]
    summarized_count = SESSION_MEMORY["summarized_count"]

    if total_msgs > 3:

        cutoff = total_msgs - 1
        new_to_summarize = messages[summarized_count:cutoff]
        recent_messages = messages[-1:]
        
        if new_to_summarize:

            history_text = "\n".join(

                [f"- {'Usuário' if isinstance(m, HumanMessage) else 'IA'}: {m.content}" for m in new_to_summarize]
            )

            prompt_memoria = (

                f"Fatos anteriores:\n{current_summary if current_summary else 'Nenhum'}\n\n"
                f"Novos logs de memória e contexto para integrar:\n{history_text}"
            )

            try:

                res = helper_llm.invoke([

                    SystemMessage(content=HELPER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_memoria)
                ])

                current_summary = res.content.strip()
                SESSION_MEMORY["summary"] = current_summary
                SESSION_MEMORY["summarized_count"] = cutoff
                print(f"[Memória LFM 1.2B] Atualizada ({len(current_summary)} chars)")

            except Exception as e:

                print(f"[Erro LFM 1.2B]: {e}")
                
        active_messages = recent_messages

    else:

        active_messages = messages

    prior_buffer = "\n".join([f"{'Usuário' if isinstance(m, HumanMessage) else 'IA'}: {m.content}" for m in active_messages[:-1]])
    partes_memoria = filter(None, [current_summary, prior_buffer])
    memoria_consolidada = "\n\n".join(partes_memoria) if partes_memoria else "Nenhum histórico."

    raw_user_input = active_messages[-1].content
    clean_user_input = sanitize_user_prompt(raw_user_input)

    return {

        "short_term_memory": memoria_consolidada,
        "clean_user_prompt": clean_user_input,
        "web_search_data": "Nenhuma busca web solicitada.",
    }

def gemma_router_node(state: AgentState):

    router_prompt = (

        f"Memória:\n{state['short_term_memory']}\n\n"
        f"Prompt do Usuário:\n{state['clean_user_prompt']}\n\n"
        "Precisa pesquisar algo externo (Tavily) para responder?"
    )

    try:

        structured_llm = main_llm.with_structured_output(SearchDecision)
        decision = structured_llm.invoke([

            SystemMessage(content="Responda baseado no schema exigido para buscas na web, caso seja relevante ao contexto atual."),
            HumanMessage(content=router_prompt)
        ])

    except Exception as e:

        print(f"[Erro Router Gemma]: {e}")
        decision = SearchDecision(need_search=False, query=None)
        
    if decision.need_search:

        print(f"[Gemma Router] Busca solicitada: {decision.query}")

    return {"search_decision": decision}

def tavily_search_node(state: AgentState):

    query = state["search_decision"].query

    try:

        response = tavily_client.search(query=query, search_depth="basic", max_results=2)
        results = response.get("results", [])
        clean_snippets = []

        for idx, r in enumerate(results, 1):

            content_clean = re.sub(r"\s{2,}", " ", r.get("content", "").replace("[...]", " ").strip())
            clean_snippets.append(f"Fonte {idx}:\n{content_clean}")
        web_text = "\n\n".join(clean_snippets) if clean_snippets else "Nenhum resultado relevante."

    except Exception as e:

        web_text = f"Falha na busca web: {e}"
        
    return {"web_search_data": web_text}

def prompt_builder_node(state: AgentState):

    """Constrói o prompt, mas NÃO executa a LLM (Para permitir SSE Streaming no FastAPI)."""

    final_messages = assemble_final_prompt(

        memoria=state["short_term_memory"],
        ide_prompt=state["system_prompt_ide"],
        dados_web=state["web_search_data"],
        prompt_usuario=state["clean_user_prompt"]
    )
    return {"final_prompt": final_messages}

def route_after_router(state: AgentState):

    decision = state.get("search_decision")
    if decision and decision.need_search and decision.query:

        return "web_search"
    return "prompt_builder"

workflow = StateGraph(AgentState)

workflow.add_node("memory_and_input_prep", memory_and_input_prep_node)
workflow.add_node("gemma_router", gemma_router_node)
workflow.add_node("web_search", tavily_search_node)
workflow.add_node("prompt_builder", prompt_builder_node)

workflow.set_entry_point("memory_and_input_prep")
workflow.add_edge("memory_and_input_prep", "gemma_router")

workflow.add_conditional_edges("gemma_router", route_after_router, {
    "web_search": "web_search",
    "prompt_builder": "prompt_builder"
})

workflow.add_edge("web_search", "prompt_builder")
workflow.add_edge("prompt_builder", END)
app_graph = workflow.compile()

# ==========================================
# 7. ENDPOINT SSE (VS CODE / CONTINUE.DEV)
# ==========================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    lc_messages = []
    system_prompt_ide = ""
    
    # 1. Separa o System Prompt gerado pela extensão Continue.dev das mensagens reais
    for m in request.messages:

        text = normalize_content(m.content)
        if m.role == "system":

            system_prompt_ide += text + "\n"
        elif m.role == "user":

            lc_messages.append(HumanMessage(content=text))
        elif m.role == "assistant":

            lc_messages.append(AIMessage(content=text))

    if not lc_messages:

        raise HTTPException(status_code=400, detail="Sem mensagens do usuário.")

    # 2. Executa a Preparação no LangGraph (Memória -> Router -> Busca -> Prompt Triplo)
    initial_state: AgentState = {

        "messages": lc_messages,
        "system_prompt_ide": system_prompt_ide.strip(),
        "short_term_memory": "",
        "clean_user_prompt": "",
        "search_decision": None,
        "web_search_data": "",
        "final_prompt": []
    }
    
    processed_state = app_graph.invoke(initial_state)
    final_prompt = processed_state["final_prompt"]
    
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # 3. Entrega Assíncrona em Stream SSE (Server-Sent Events) compatível com OpenAI
    if request.stream:

        async def sse_generator():

            try:

                async for chunk in main_llm.astream(final_prompt):

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
                        
                # Sinal de Encerramento (Final Reason: stop)
                end_payload = {

                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }

                yield f"data: {json.dumps(end_payload)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as stream_err:

                print(f"[Erro de Stream]: {stream_err}")
                yield "data: [DONE]\n\n"
                
        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # 4. Fallback caso a IDE peça retorno síncrono (sem stream)

    response = await main_llm.ainvoke(final_prompt)
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

    print("=== Iniciando LangGraph Proxy (Gemma4/LFM) na porta 8000 ===")
    uvicorn.run(app, host="127.0.0.1", port=8000)