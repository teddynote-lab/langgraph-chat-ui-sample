"""ChatBedrock wrapper that sanitizes messages for Bedrock API compatibility."""

import copy
from typing import Any, Iterator

from langchain_aws.chat_models import ChatBedrock
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk


class SanitizedChatBedrock(ChatBedrock):
    """ChatBedrock wrapper that sanitizes messages before sending."""

    def _sanitize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return [self._sanitize_content(item) for item in content]
        if isinstance(content, dict):
            if content.get("type") == "text" and "id" in content:
                return {k: self._sanitize_content(v) for k, v in content.items() if k != "id"}
            return {k: self._sanitize_content(v) for k, v in content.items()}
        return content

    def _sanitize_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Remove unsupported 'id' field from tool result content for Bedrock API."""
        result = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                new_msg = copy.deepcopy(msg)
                new_msg.content = self._sanitize_content(msg.content)
                result.append(new_msg)
            else:
                result.append(msg)
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        sanitized = self._sanitize_messages(messages)
        return super()._generate(sanitized, stop, run_manager, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        sanitized = self._sanitize_messages(messages)
        yield from super()._stream(sanitized, stop, run_manager, **kwargs)
