# LangGraph Chat UI Samples

[LangGraph Chat UI](https://github.com/teddynote-lab/langgraph-chat-ui) 테스트를 위한 LangGraph 예시 서버입니다.

다양한 LangGraph 패턴을 하나의 서버에서 제공하여,  
Chat UI의 그래프 전환, 스트리밍, 도구 호출 시각화 등의 기능을 테스트할 수 있습니다.

## 제공 그래프

| 그래프 | 패턴 | 토폴로지 |
|---|---|---|
| `chatbot` | 선형 | `START → chat → END` |
| `react_agent` | 도구 호출 루프 | `START → agent ⇄ tools → END` |
| `hitl_agent` | Human-in-the-loop | ReAct + `interrupt_before=["tools"]` |
| `supervisor` | 멀티에이전트 | `supervisor ⇄ [searcher \| reader] → END` |
| `map_reduce` | 병렬 실행 | `planner → [researcher × N] → synthesizer` |
| `reflection` | 생성-비평 루프 | `generator ↔ critic` (최대 3회) |
| `deep_research` | Deep Agents | 오케스트레이터 + web_searcher/web_reader 서브에이전트 |
| `storm_research` | STORM | 다중 분석가 인터뷰 → 병렬 보고서 합성 |

## 시작하기

### 사전 준비

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (권장) 또는 pip

### 설치

```bash
cd sample-graphs
uv sync
```

Playwright 브라우저 설치 (`supervisor`, `deep_research` 그래프에서 사용):

```bash
uv run playwright install chromium
```

### 환경 변수

`.env` 파일에 다음 키가 필요합니다:

```
# 필수 — LLM (하나 이상)
AWS_MODEL_ID=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...

# 또는
OPENAI_API_KEY=...

# 또는
UPSTAGE_API_KEY=...

# 도구 검색
TAVILY_API_KEY=...

# 추적 (선택)
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

### 실행

```bash
# 개발 모드
uv run langgraph dev

# 또는 특정 포트
uv run langgraph dev --port 2024
```

서버가 시작되면 LangGraph Chat UI에서 `http://localhost:2024` 를 서버 URL로 등록하세요.

## LLM 프로바이더 변경

기본값은 AWS Bedrock입니다. 다른 프로바이더를 사용하려면 각 그래프 파일에서 `init_llm(provider=...)` 호출을 수정하세요:

```python
# graphs/_common.py 또는 개별 그래프 파일
llm = init_llm(provider="openai")    # OpenAI
llm = init_llm(provider="upstage")   # Upstage Solar
llm = init_llm(provider="aws")       # AWS Bedrock (기본)
```

STORM 그래프는 `langgraph.json`의 `configurable`로 런타임에 프로바이더를 변경할 수 있습니다:

```json
{
  "configurable": {
    "model": "openai/gpt-4.1"
  }
}
```
