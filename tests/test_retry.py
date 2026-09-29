"""LLM 客户端重试逻辑测试：退避、Retry-After、不可重试错误、reasoning_content 特判。"""

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from codegen.config import Settings
from codegen.errors import RetryExhausted
from codegen.llm import LLMClient, backoff_delay, retry_after_seconds, should_retry

URL = "https://api.deepseek.com/chat/completions"


def make_status_error(status: int, message: str = "error"):
    request = httpx.Request("POST", URL)
    response = httpx.Response(status, request=request,
                              text='{"error": {"message": "%s"}}' % message)
    # openai v3：APIStatusError(message, *, response, body)
    return APIStatusError(message, response=response, body={"error": {"message": message}})


def make_timeout_error():
    return APITimeoutError(httpx.Request("POST", URL))


def make_conn_error():
    return APIConnectionError(request=httpx.Request("POST", URL))


class FakeResponse:
    """最小可用的成功响应：message 为 dict（llm.py 兼容两种形态）。"""

    def __init__(self, content="ok", finish="stop"):
        self.choices = [
            type("C", (), {"message": {"role": "assistant", "content": content},
                           "finish_reason": finish})()
        ]


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.messages_seen = []

    def create(self, **kwargs):
        self.messages_seen.append(kwargs.get("messages"))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.chat = type("Chat", (), {"completions": FakeCompletions(outcomes)})()


def make_llm(outcomes, settings=None, sleeps=None):
    s = settings or Settings(api_key="sk-test")
    sleeps = sleeps if sleeps is not None else []
    client = FakeClient(outcomes)
    llm = LLMClient(s, client=client, sleep_fn=sleeps.append)
    return llm, client.chat.completions, sleeps


def test_success_passthrough():
    llm, completions, _ = make_llm([FakeResponse("你好")])
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "你好"
    assert resp.message["role"] == "assistant"
    assert resp.truncated is False
    assert len(completions.messages_seen) == 1


def test_retries_on_429_then_succeeds():
    llm, completions, sleeps = make_llm([make_status_error(429), FakeResponse("ok")])
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "ok"
    assert len(completions.messages_seen) == 2
    assert len(sleeps) == 1


def test_retry_exhausted_after_5xx():
    llm, completions, sleeps = make_llm([make_status_error(500)] * 6)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("应当抛出 RetryExhausted")
    except RetryExhausted as e:
        assert "重试 5 次" in str(e)
    assert len(completions.messages_seen) == 6  # 首次 + 5 次重试


def test_no_retry_on_401():
    llm, completions, sleeps = make_llm([make_status_error(401)])
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("应当抛出 RetryExhausted")
    except RetryExhausted:
        pass
    assert len(completions.messages_seen) == 1
    assert sleeps == []  # 401 不重试、不等待


def test_retry_after_header_respected():
    err = make_status_error(429)
    err.response.headers["Retry-After"] = "7"
    llm, completions, sleeps = make_llm([err, FakeResponse("ok")])
    llm.chat([{"role": "user", "content": "hi"}])
    assert sleeps == [7.0]


def test_backoff_is_exponential_and_bounded():
    delays = [backoff_delay(i) for i in range(5)]
    assert all(1.0 <= d <= 30.5 for d in delays)
    assert delays[0] < delays[3]


def test_should_retry_classification():
    assert should_retry(make_status_error(429)) is True
    assert should_retry(make_status_error(500)) is True
    assert should_retry(make_status_error(400)) is False
    assert should_retry(make_status_error(401)) is False
    assert should_retry(make_timeout_error()) is True
    assert should_retry(make_conn_error()) is True


def test_retry_after_seconds_missing_header():
    assert retry_after_seconds(make_status_error(429)) is None


def test_reasoning_content_400_strips_fields_and_retries():
    msgs = [{"role": "assistant", "content": "x", "reasoning_content": "思考内容"},
            {"role": "user", "content": "hi"}]
    err = make_status_error(400, message="invalid request: reasoning_content not allowed")
    llm, completions, _ = make_llm([err, FakeResponse("ok")])
    resp = llm.chat(msgs)
    assert resp.content == "ok"
    assert len(completions.messages_seen) == 2
    second_call = completions.messages_seen[1]
    assert all("reasoning_content" not in m for m in second_call)
