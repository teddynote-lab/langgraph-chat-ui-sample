import os
from datetime import datetime

from tavily import TavilyClient
from langchain_core.tools import StructuredTool


def tavily_deep_search(query: str) -> str:
    """주어진 쿼리에 대해 심층 웹 검색을 수행합니다. 여러 소스에서 상세한 정보를 수집합니다."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query,
                             search_depth="advanced",
                             max_results=5,
                             include_raw_content=False)

    now = datetime.now()
    current_time = now.strftime("%Y년 %m월 %d일 %H시 %M분")

    results = []
    for r in response.get("results", []):
        entry = f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}"
        results.append(entry)

    header = f"[심층 검색 시각: {current_time}]\n\n"
    return header + "\n---\n".join(results)


def tavily_deep_search_with_context(query: str,
                                    context: str = "") -> str:
    """이전 검색 결과를 참고하여 후속 심층 검색을 수행합니다. 반복적 리서치에 유용합니다.

    Args:
        query: 검색 쿼리
        context: 이전 검색에서 얻은 맥락 정보 (후속 검색 정제에 활용)
    """
    refined_query = f"{query} {context}".strip() if context else query
    return tavily_deep_search(refined_query)


def init_deep_search_tools():
    """딥리서치용 검색 도구 반환"""
    tools = [
        StructuredTool.from_function(
            func=tavily_deep_search,
            name="deep_web_search",
            description=
            "Perform a deep web search for a given query. Returns detailed results from multiple sources. "
            "Use this for broad initial research on any topic."),
        StructuredTool.from_function(
            func=tavily_deep_search_with_context,
            name="followup_search",
            description=
            "Perform a follow-up search using context from previous results. "
            "Use this to dig deeper into specific aspects discovered during initial research."
        ),
    ]
    return tools
