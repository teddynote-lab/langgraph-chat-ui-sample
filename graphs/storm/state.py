"""STORM Research — State Definitions

Migrated from STORM-Research project with LangGraph 1.0 enhancements:
- Fixed TypedDict default value handling (original used dataclass `field()` in TypedDict)
- Cleaned up duplicate Annotated imports
"""

import operator
from typing import Annotated, List, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ====================== Data Models ======================


class Analyst(BaseModel):
    """Analyst persona with unique perspective and expertise."""

    affiliation: str = Field(description="Analyst's primary organization")
    name: str = Field(description="Analyst's name")
    role: str = Field(description="Analyst's role related to the topic")
    description: str = Field(
        description="Description of analyst's interests, concerns, and motivations"
    )

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\nRole: {self.role}\n"
            f"Affiliation: {self.affiliation}\nDescription: {self.description}\n"
        )


class Perspectives(BaseModel):
    """Collection of analyst personas."""

    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts including roles and affiliations"
    )


class SearchQuery(BaseModel):
    """Structured search query for information retrieval."""

    search_query: str = Field(None, description="Search query for information retrieval")


# ====================== State Definitions ======================


class InputState(TypedDict, total=False):
    """Schema for graph input."""

    messages: Annotated[Sequence[AnyMessage], add_messages]


class OutputState(TypedDict):
    """Schema for graph output."""

    final_report: str


class InterviewState(MessagesState):
    """State for a single analyst interview.

    Inherits MessagesState for automatic conversation history management.
    """

    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class ResearchGraphState(TypedDict, total=False):
    """Internal state of the entire research process.

    Uses total=False so all fields are optional, allowing partial state updates.
    """

    topic: str
    max_analysts: int
    human_analyst_feedback: Optional[str]
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
    messages: Annotated[Sequence[AnyMessage], add_messages]
