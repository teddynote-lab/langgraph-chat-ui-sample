"""STORM Research — Main Graph Definition

Migrated from STORM-Research with LangGraph 1.0 enhancements:
- Native StateGraph API with input/output schema separation
- Send() for parallel interview fan-out
- Parallel edges for report writing (introduction, body, conclusion)
- Extended LLM provider support (upstage, aws, openai, anthropic, azure)
- Fixed state schema bugs from original

Topology:
  START → create_analysts → [conduct_interview × N] →
    ┌─ write_report ────────┐
    ├─ write_introduction ──┼─→ finalize_report → END
    └─ write_conclusion ────┘
"""

from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

from graphs.storm.state import (
    InterviewState,
    InputState,
    OutputState,
    ResearchGraphState,
    Perspectives,
    SearchQuery,
)
from graphs.storm.prompts import (
    ANALYST_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    ANSWER_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    SECTION_WRITER_INSTRUCTIONS,
    REPORT_WRITER_INSTRUCTIONS,
    INTRO_CONCLUSION_INSTRUCTIONS,
)
from graphs.storm.configuration import Configuration
from graphs.storm.tools import get_search_tools
from graphs.storm.utils import load_chat_model


# ====================== Analyst Generation ======================


async def create_analysts(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Generate analyst personas tailored to the research topic."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    messages = state.get("messages", [])
    topic = messages[-1].content if messages else state.get("topic")
    if not topic:
        raise ValueError("Topic must be provided in messages or state['topic'].")

    max_analysts = state.get("max_analysts", configuration.max_analysts)

    structured_model = model.with_structured_output(Perspectives)
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback=state.get("human_analyst_feedback", ""),
        max_analysts=max_analysts,
    )

    result = await structured_model.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts."),
    ])

    return {"analysts": result.analysts, "topic": topic}


# ====================== Interview Nodes ======================


async def generate_question(state: InterviewState, config: RunnableConfig) -> dict:
    """Analyst generates a question for the expert."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    system_message = QUESTION_INSTRUCTIONS.format(goals=state["analyst"].persona)
    question = await model.ainvoke(
        [SystemMessage(content=system_message)] + state["messages"]
    )
    return {"messages": [question]}


async def search_web(state: InterviewState, config: RunnableConfig) -> dict:
    """Search the web for relevant information."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )
    results = await search_tools.search_web(search_query.search_query)
    return {"context": [results]}


async def search_arxiv(state: InterviewState, config: RunnableConfig) -> dict:
    """Search ArXiv for academic papers."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )
    results = await search_tools.search_arxiv(search_query.search_query)
    return {"context": [results]}


async def generate_answer(state: InterviewState, config: RunnableConfig) -> dict:
    """Expert answers the analyst's question using retrieved context."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    system_message = ANSWER_INSTRUCTIONS.format(
        goals=state["analyst"].persona,
        context=state["context"],
    )
    answer = await model.ainvoke(
        [SystemMessage(content=system_message)] + state["messages"]
    )
    answer.name = "expert"
    return {"messages": [answer]}


async def save_interview(state: InterviewState) -> dict:
    """Persist interview transcript as a string."""
    return {"interview": get_buffer_string(state["messages"])}


def route_messages(
    state: InterviewState, name: str = "expert"
) -> Literal["ask_question", "save_interview"]:
    """Continue interviewing or save when done."""
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 3)

    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return "save_interview"

    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return "save_interview"

    return "ask_question"


async def write_section(state: InterviewState, config: RunnableConfig) -> dict:
    """Write a report section from a single analyst's interview."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    system_message = SECTION_WRITER_INSTRUCTIONS.format(
        focus=state["analyst"].description
    )
    section = await model.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {state['context']}"),
    ])
    return {"sections": [section.content]}


# ====================== Report Writing Nodes ======================


def initiate_all_interviews(state: ResearchGraphState) -> List[Send]:
    """Fan-out: start a parallel interview for each analyst."""
    topic = state.get("topic", "")
    return [
        Send(
            "conduct_interview",
            {
                "analyst": analyst,
                "messages": [
                    HumanMessage(
                        content=f"So you said you were writing an article on {topic}?"
                    )
                ],
                "max_num_turns": state.get("max_num_turns", 3),
            },
        )
        for analyst in state["analysts"]
    ]


async def write_report(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Synthesize all analyst sections into the report body."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    formatted_sections = "\n\n".join(state["sections"])
    system_message = REPORT_WRITER_INSTRUCTIONS.format(
        topic=state.get("topic", ""),
        context=formatted_sections,
        language=configuration.language,
    )

    report = await model.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos."),
    ])
    return {"content": report.content}


async def write_introduction(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Write the report introduction."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    formatted_sections = "\n\n".join(state["sections"])
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=state.get("topic", ""),
        formatted_str_sections=formatted_sections,
        language=configuration.language,
    )

    intro = await model.ainvoke([
        instructions,
        HumanMessage(content="Write the report introduction"),
    ])
    return {"introduction": intro.content}


async def write_conclusion(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Write the report conclusion."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    formatted_sections = "\n\n".join(state["sections"])
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=state.get("topic", ""),
        formatted_str_sections=formatted_sections,
        language=configuration.language,
    )

    conclusion = await model.ainvoke([
        instructions,
        HumanMessage(content="Write the report conclusion"),
    ])
    return {"conclusion": conclusion.content}


async def finalize_report(state: ResearchGraphState) -> dict:
    """Assemble introduction + body + conclusion into the final report."""
    content = state["content"]

    if content.startswith("## Insights"):
        content = content[len("## Insights"):].lstrip()

    sources = None
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except ValueError:
            sources = None

    final_report = (
        state["introduction"]
        + "\n\n---\n\n## Main Idea\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )

    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }


# ====================== Graph Build Functions ======================


def build_interview_graph():
    """Build the interview subgraph for a single analyst."""
    builder = StateGraph(InterviewState)

    builder.add_node("ask_question", generate_question)
    builder.add_node("search_web", search_web)
    builder.add_node("search_arxiv", search_arxiv)
    builder.add_node("answer_question", generate_answer)
    builder.add_node("save_interview", save_interview)
    builder.add_node("write_section", write_section)

    builder.add_edge(START, "ask_question")
    # Parallel search: web + arxiv
    builder.add_edge("ask_question", "search_web")
    builder.add_edge("ask_question", "search_arxiv")
    builder.add_edge("search_web", "answer_question")
    builder.add_edge("search_arxiv", "answer_question")
    # Conditional loop or save
    builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    builder.add_edge("save_interview", "write_section")
    builder.add_edge("write_section", END)

    return builder.compile().with_config(run_name="Conduct Interview")


def build_research_graph():
    """Build the main STORM research orchestration graph."""
    interview_graph = build_interview_graph()

    builder = StateGraph(
        ResearchGraphState,
        input_schema=InputState,
        output_schema=OutputState,
    )

    builder.add_node("create_analysts", create_analysts)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    builder.add_edge(START, "create_analysts")
    builder.add_conditional_edges(
        "create_analysts", initiate_all_interviews, ["conduct_interview"]
    )

    # Parallel report writing after all interviews complete
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")

    # Join: all three must finish before finalize
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
    )
    builder.add_edge("finalize_report", END)

    return builder.compile()


# ====================== Graph Instance ======================

graph = build_research_graph()
