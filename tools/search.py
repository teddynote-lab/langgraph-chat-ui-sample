import os
from datetime import datetime

from tavily import TavilyClient


def tavily_search(query: str) -> str:
    """선수에 대한 객관적인 정보(이적, 경력, 소속팀 등)를 웹 검색으로 조회합니다.

    주의: 트렌드, 메타, 패치 분석 등에는 사용하지 마세요.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query, max_results=3)

    now = datetime.now()
    current_time = now.strftime("%Y년 %m월 %d일 %H시 %M분")

    results = [
        f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}\n"
        for r in response.get("results", [])
    ]

    header = f"[검색 시각: {current_time}]\n\n"
    return header + "\n---\n".join(results)
