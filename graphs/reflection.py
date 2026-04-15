"""6. Reflection — 생성-비평 루프

토폴로지: START → generator → critic → {max iterations?} → generator (루프) 또는 END
생성자가 초안 작성 → 비평자가 피드백 → 개선 반복 (최대 3회).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from graphs._common import init_llm


# --- State ---
class ReflectionState(TypedDict):
    messages: Annotated[list, operator.add]
    iteration: Annotated[int, operator.add]


MAX_ITERATIONS = 3

llm = init_llm(provider="aws")


# --- Nodes ---
def generator(state: ReflectionState) -> ReflectionState:
    """초안 작성 또는 피드백 반영하여 개선"""
    messages = state["messages"]

    # 첫 반복이면 초안 작성, 이후엔 피드백 반영
    iteration = state.get("iteration", 0)
    if iteration == 0:
        system_msg = "당신은 뛰어난 작가입니다. 사용자의 요청에 따라 글을 작성하세요."
    else:
        system_msg = "당신은 뛰어난 작가입니다. 비평가의 피드백을 반영하여 글을 개선하세요. 개선된 전체 글을 출력하세요."

    prompt_messages = [
        {"role": "system", "content": system_msg},
        *messages,
    ]
    response = llm.invoke(prompt_messages)
    return {
        "messages": [AIMessage(content=response.content, name="generator")],
        "iteration": 1,
    }


def critic(state: ReflectionState) -> ReflectionState:
    """생성된 글에 대해 비평 및 개선 제안"""
    messages = state["messages"]

    system_msg = """당신은 날카로운 비평가입니다. 방금 생성된 글에 대해 구체적이고 건설적인 피드백을 제공하세요.
- 논리적 흐름
- 표현의 명확성
- 빠진 관점이나 내용
- 개선할 점

피드백을 간결하게 제시하세요."""

    prompt_messages = [
        {"role": "system", "content": system_msg},
        *messages,
    ]
    response = llm.invoke(prompt_messages)
    return {
        "messages": [HumanMessage(content=response.content, name="critic")],
    }


def should_continue(state: ReflectionState) -> str:
    """최대 반복 횟수에 도달했는지 확인"""
    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        return "end"
    return "continue"


# --- Graph ---
builder = StateGraph(ReflectionState)
builder.add_node("generator", generator)
builder.add_node("critic", critic)

builder.add_edge(START, "generator")
builder.add_edge("generator", "critic")
builder.add_conditional_edges(
    "critic",
    should_continue,
    {
        "continue": "generator",
        "end": END,
    },
)

graph = builder.compile()
