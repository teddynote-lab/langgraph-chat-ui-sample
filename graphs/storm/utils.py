"""STORM Research — Utility Functions

Enhanced from original:
- Added upstage and aws (Bedrock) provider support via project's init_llm()
- Kept openai / anthropic / azure for broad compatibility
"""

import os
import sys

from langchain_core.language_models import BaseChatModel

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_chat_model(model_string: str) -> BaseChatModel:
    """Parse 'provider/model-name' and return the appropriate Chat model.

    Supported providers:
      - upstage  → ChatUpstage via init_llm
      - aws      → SanitizedChatBedrock via init_llm
      - openai   → ChatOpenAI
      - anthropic → ChatAnthropic
      - azure    → AzureChatOpenAI
    """
    try:
        provider, model_name = model_string.split("/", 1)
    except ValueError:
        raise ValueError(
            f"Model string must be 'provider/model-name'. Got: {model_string}"
        )

    if provider == "upstage":
        from llm import init_llm
        return init_llm(provider="upstage", model_name=model_name)

    if provider == "aws":
        from llm import init_llm
        return init_llm(provider="aws")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name)

    if provider == "azure":
        from langchain_openai.chat_models import AzureChatOpenAI
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not azure_endpoint or not azure_api_key:
            raise ValueError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )
        return AzureChatOpenAI(
            deployment_name=model_name,
            api_version="2024-12-01-preview",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            temperature=0.1,
        )

    raise ValueError(f"Unsupported provider: {provider}")
