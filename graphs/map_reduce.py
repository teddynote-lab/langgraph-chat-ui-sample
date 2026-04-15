"""5. Map-Reduce — 동적 병렬 실행

토폴로지: START → planner → [researcher × N] → synthesizer → END Send() API로 동적으로 병렬 노드를 생성하여 각 서브토픽을 병렬 연구 후 종합.
"""

from __future__ import annotations

import operator
import json
from typing import Annotated, List, TypedDict
from pydantic import Field
from typing_extensions import NotRequired, Required

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from graphs._common import init_llm, tavily_search


class OverallState(TypedDict):
    messages: Annotated[list, operator.add]
    research_topics: list[str]
    research_results: Annotated[list, operator.add]

    #test_files: Annotated[List[str], Field(json_schema_extra={"x-field-display": "required"})]


class ResearcherInput(TypedDict):
    topic: str


# --- LLM ---
llm = init_llm(provider="aws")


# --- Nodes ---
def planner(state: OverallState) -> OverallState:
    """사용자 질문을 분석하여 연구 서브토픽으로 세분화"""
    messages = state["messages"]
    user_query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    prompt = f"""다음 질문에 대해 조사할 서브토픽 2~4개를 JSON 배열로 반환하세요.
반드시 JSON 배열만 출력하고, 다른 텍스트는 포함하지 마세요.

질문: {user_query}

예시 출력: ["서브토픽1", "서브토픽2", "서브토픽3"]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # JSON 배열 파싱
    try:
        # 코드블록 내부 JSON 처리
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        topics = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        topics = [user_query]

    return {"research_topics": topics}


def researcher(state: ResearcherInput) -> OverallState:
    """개별 서브토픽에 대해 웹 검색 수행"""
    topic = state["topic"]
    search_result = tavily_search(topic)
    return {
        "research_results": [
            f"## {topic}\n\n{search_result}"
        ]
    }


def synthesizer(state: OverallState) -> OverallState:
    """모든 연구 결과를 종합하여 최종 답변 생성"""
    results = state.get("research_results", [])
    messages = state["messages"]
    user_query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    combined = "\n\n---\n\n".join(results)
    prompt = f"""다음 연구 결과들을 종합하여 원래 질문에 대한 포괄적인 답변을 작성하세요.

원래 질문: {user_query}

연구 결과:
{combined}

종합 답변:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


def route_to_researchers(state: OverallState) -> list[Send]:
    """planner 결과에 따라 각 서브토픽에 대해 Send로 병렬 researcher 생성"""
    topics = state.get("research_topics", [])
    return [Send("researcher", {"topic": t}) for t in topics]


# --- Graph ---
builder = StateGraph(OverallState)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", route_to_researchers, ["researcher"])
builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()
