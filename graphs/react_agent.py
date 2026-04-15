"""2. ReAct Agent — 도구 호출 루프

토폴로지: START → agent ⇄ tools → END
LLM이 도구 호출 여부를 결정하는 조건부 루프.
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

graph = builder.compile()
