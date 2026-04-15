"""4. Human-in-the-loop — 인터럽트 기반 에이전트

토폴로지: ReAct + interrupt_before=["tools"]
도구 실행 전 중단점을 두어 사용자가 승인/거부/수정 가능.
"""

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from graphs._common import init_llm, tavily_search


@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다. 최신 정보나 사실 확인이 필요할 때 사용하세요."""
    return tavily_search(query)


tools = [web_search]
llm = init_llm(provider="aws").bind_tools(tools)


def agent(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# interrupt_before=["tools"] 로 도구 실행 전 사용자 승인 요구
graph = builder.compile(interrupt_before=["tools"])
