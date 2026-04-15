import os

from botocore.config import Config as BotocoreConfig
from boto3 import client as boto3_client

from langchain_upstage import ChatUpstage
from langchain.chat_models import init_chat_model

from sanitized_bedrock import SanitizedChatBedrock


def init_llm(provider: str,
             model_name: str | None = None,
             *,
             temperature=0.0,
             max_tokens=8192,
             timeout=300):
    if provider == "upstage":
        llm = ChatUpstage(name=model_name,
                          model="solar-pro3",
                          temperature=temperature,
                          reasoning_effort="medium")
    elif provider == "aws":

        boto_config = BotocoreConfig(
            read_timeout=timeout,
            connect_timeout=60,
        )

        bedrock_client = boto3_client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=boto_config,
        )

        return SanitizedChatBedrock(
            model_id=os.environ["AWS_MODEL_ID"],
            client=bedrock_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        llm = init_chat_model(model_name)
    return llm
