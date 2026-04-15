"""공통 LLM 초기화, 상태 스키마, 도구 임포트"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import init_llm
from tools.search import tavily_search
from tools.playwright import init_playwright_tools

__all__ = [
    "init_llm",
    "tavily_search",
    "init_playwright_tools",
]
