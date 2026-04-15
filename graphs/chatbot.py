"""1. Simple Chatbot — 선형 그래프

토폴로지: START → chat → END
가장 단순한 구조. 도구 없이 LLM만 사용.
"""

from langgraph.graph import StateGraph, START, END, MessagesState

from graphs._common import init_llm

llm = init_llm(provider="aws")


def chat(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
