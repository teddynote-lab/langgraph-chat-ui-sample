"""STORM Research — Search Tools

Web search (Tavily) and academic paper search (ArXiv).
"""

from typing import Optional

from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig

from graphs.storm.configuration import Configuration


class SearchTools:
    """Configurable web + academic search."""

    def __init__(self, config: Optional[RunnableConfig] = None):
        cfg = Configuration.from_runnable_config(config)

        self.tavily_search = TavilySearchResults(max_results=cfg.tavily_max_results)
        self.arxiv_retriever = ArxivRetriever(
            load_max_docs=cfg.arxiv_max_docs,
            load_all_available_meta=True,
            get_full_documents=True,
        )

    async def search_web(self, query: str) -> str:
        try:
            results = await self.tavily_search.ainvoke(query)
            formatted = []
            for doc in results:
                formatted.append(
                    f'<Document href="{doc["url"]}"/>\n'
                    f'{doc["content"]}\n'
                    f'</Document>'
                )
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            return f"<Error>Web search error: {e}</Error>"

    async def search_arxiv(self, query: str) -> str:
        try:
            results = await self.arxiv_retriever.ainvoke(query)
            formatted = []
            for doc in results:
                meta = doc.metadata
                formatted.append(
                    f'<Document source="{meta["entry_id"]}" '
                    f'date="{meta.get("Published", "")}" '
                    f'authors="{meta.get("Authors", "")}"/>\n'
                    f'<Title>\n{meta["Title"]}\n</Title>\n\n'
                    f'<Summary>\n{meta["Summary"]}\n</Summary>\n\n'
                    f'<Content>\n{doc.page_content}\n</Content>\n'
                    f'</Document>'
                )
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            return f"<Error>ArXiv search error: {e}</Error>"


def get_search_tools(config: Optional[RunnableConfig] = None) -> SearchTools:
    return SearchTools(config)
