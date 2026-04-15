"""STORM Research — Configuration Management

Enhanced from original:
- Added upstage / aws provider support (matches this project's init_llm)
- Fixed language field not being passed through from_runnable_config
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig


@dataclass
class Configuration:
    """Runtime configuration exposed in LangGraph Studio."""

    model: str = field(
        default="aws/bedrock",
        metadata={
            "description": "LLM model (provider/model format)",
            "examples": [
                "upstage/solar-pro3",
                "aws/bedrock",
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini",
                "anthropic/claude-opus-4-20250514",
                "anthropic/claude-sonnet-4-20250514",
            ],
        },
    )

    max_analysts: int = field(
        default=3,
        metadata={"description": "Maximum number of analysts to generate", "range": [1, 10]},
    )

    max_interview_turns: int = field(
        default=3,
        metadata={"description": "Maximum conversation turns per interview", "range": [1, 10]},
    )

    tavily_max_results: int = field(
        default=3,
        metadata={"description": "Maximum Tavily search results", "range": [1, 10]},
    )

    arxiv_max_docs: int = field(
        default=3,
        metadata={"description": "Maximum ArXiv search documents", "range": [1, 10]},
    )

    parallel_interviews: bool = field(
        default=True,
        metadata={"description": "Whether to run interviews in parallel"},
    )

    language: Literal["Korean", "English"] = field(
        default="Korean",
        metadata={"description": "Language for the final report"},
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        defaults = cls()
        return cls(
            model=configurable.get("model", defaults.model),
            max_analysts=configurable.get("max_analysts", defaults.max_analysts),
            max_interview_turns=configurable.get(
                "max_interview_turns", defaults.max_interview_turns
            ),
            tavily_max_results=configurable.get(
                "tavily_max_results", defaults.tavily_max_results
            ),
            arxiv_max_docs=configurable.get("arxiv_max_docs", defaults.arxiv_max_docs),
            parallel_interviews=configurable.get(
                "parallel_interviews", defaults.parallel_interviews
            ),
            language=configurable.get("language", defaults.language),
        )
