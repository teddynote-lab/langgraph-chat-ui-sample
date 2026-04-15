import asyncio
import threading

from langchain_core.tools import StructuredTool


class PlaywrightWorker:
    """Playwright를 별도 스레드의 전용 이벤트 루프에서 실행"""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def start(self):
        """워커 스레드 시작"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60)

    def _run(self):
        """별도 스레드에서 전용 이벤트 루프 실행"""
        import asyncio as _asyncio
        self._loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(self._loop)

        self._loop.run_until_complete(self._init_browser())
        self._ready.set()

        self._loop.run_forever()

    async def _init_browser(self):
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled'],
            timeout=30000)
        self._context = await self._browser.new_context(
            user_agent=
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={
                'width': 1920,
                'height': 1080
            },
            locale='ko-KR',
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(30000)
        self._page.set_default_navigation_timeout(30000)
        await self._page.add_init_script(
            'Object.defineProperty(navigator, "webdriver", {get: () => undefined});'
        )

    def _run_coro(self, coro, timeout=60):
        """코루틴을 워커 스레드에서 실행하고 결과 반환"""
        import asyncio as _asyncio
        future = _asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def navigate(self, url: str) -> str:

        async def _nav():
            response = await self._page.goto(url,
                                             wait_until='domcontentloaded')
            await asyncio.sleep(1)
            return f"Navigated to {url}, status: {response.status if response else 'unknown'}"

        return self._run_coro(_nav())

    def extract_text(self) -> str:

        async def _extract():
            return await self._page.inner_text('body')

        return self._run_coro(_extract())

    def get_url(self) -> str:

        async def _get():
            return self._page.url

        return self._run_coro(_get())


_playwright_worker = None


def _get_worker():
    global _playwright_worker
    if _playwright_worker is None:
        _playwright_worker = PlaywrightWorker()
        _playwright_worker.start()
    return _playwright_worker


async def navigate_browser(url: str) -> str:
    """웹 브라우저로 URL에 접속합니다."""
    worker = _get_worker()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, worker.navigate, url)


async def extract_page_text() -> str:
    """현재 페이지의 텍스트를 추출합니다."""
    worker = _get_worker()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, worker.extract_text)


async def get_current_url() -> str:
    """현재 페이지의 URL을 반환합니다."""
    worker = _get_worker()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, worker.get_url)


def init_playwright_tools():
    """커스텀 Playwright 도구 반환"""
    tools = [
        StructuredTool.from_function(
            coroutine=navigate_browser,
            name="navigate_browser",
            description=
            "Navigate to a URL using web browser. Use this to visit websites like esports.op.gg."
        ),
        StructuredTool.from_function(
            coroutine=extract_page_text,
            name="extract_text",
            description="Extract all text from the current webpage."),
        StructuredTool.from_function(
            coroutine=get_current_url,
            name="current_webpage",
            description="Get the URL of the current page."),
    ]
    return tools
