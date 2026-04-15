"""3. Multi-Agent Supervisor — 허브-앤-스포크 라우팅

토폴로지: START → supervisor ⇄ [searcher | reader] → END
수퍼바이저가 어느 워커에게 위임할지 결정.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from graphs._common import init_llm, tavily_search, init_playwright_tools


# --- State ---
class SupervisorState(MessagesState):
    next: str


# --- Workers ---
# Searcher: Tavily 검색
@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    return tavily_search(query)


searcher_llm = init_llm(provider="aws").bind_tools([search_web])
searcher_tools = ToolNode([search_web])

# Reader: Playwright 웹 읽기
playwright_tools = init_playwright_tools()
reader_llm = init_llm(provider="aws").bind_tools(playwright_tools)
reader_tools = ToolNode(playwright_tools)


# --- Supervisor ---
supervisor_llm = init_llm(provider="aws")

SUPERVISOR_PROMPT = """You are a supervisor managing two workers:
- "searcher": searches the web for information using search queries
- "reader": navigates to specific URLs and reads webpage content

Given the conversation, decide which worker should act next, or if the task is complete.
Respond with ONLY one of: "searcher", "reader", or "FINISH".
Do not explain your reasoning, just output the worker name or FINISH."""


def supervisor(state: SupervisorState) -> SupervisorState:
    messages = [
        {"role": "system", "content": SUPERVISOR_PROMPT},
        *state["messages"],
    ]
    response = supervisor_llm.invoke(messages)
    content = response.content.strip().strip('"').strip("'")

    # 유효한 워커 이름 또는 FINISH 로 정규화
    if "searcher" in content.lower():
        next_worker = "searcher"
    elif "reader" in content.lower():
        next_worker = "reader"
    else:
        next_worker = "FINISH"

    return {"next": next_worker}


def searcher_node(state: SupervisorState) -> SupervisorState:
    response = searcher_llm.invoke(state["messages"])
    # 도구 호출이 있으면 실행
    if response.tool_calls:
        tool_results = searcher_tools.invoke({"messages": [response]})
        return {"messages": [response, *tool_results["messages"]]}
    return {"messages": [AIMessage(content=response.content, name="searcher")]}


def reader_node(state: SupervisorState) -> SupervisorState:
    response = reader_llm.invoke(state["messages"])
    if response.tool_calls:
        tool_results = reader_tools.invoke({"messages": [response]})
        return {"messages": [response, *tool_results["messages"]]}
    return {"messages": [AIMessage(content=response.content, name="reader")]}


# --- Routing ---
def route_supervisor(state: SupervisorState) -> Literal["searcher", "reader", "__end__"]:
    next_val = state.get("next", "FINISH")
    if next_val == "FINISH":
        return END
    return next_val


# --- Graph ---
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("searcher", searcher_node)
builder.add_node("reader", reader_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, ["searcher", "reader", END])
builder.add_edge("searcher", "supervisor")
builder.add_edge("reader", "supervisor")

graph = builder.compile()
