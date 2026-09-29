"""DeepSeek LLM 客户端：重试退避 + 消息兼容。

关键约束（2026-09 核实，详见 Design.md §7）：
- deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，默认 deepseek-v4-flash；
- 不发送 tool_choice 参数（V4 思考模式下连 "auto" 都会返回 400）；
- 多轮工具循环需原样回传整条 assistant 消息（保留 tool_calls / reasoning_content）；
- 思考模式默认关闭（thinking=disabled），稳定且便宜。
"""

import random
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .config import Settings
from .errors import RetryExhausted

MAX_RETRIES = 5
BASE_DELAY = 1.0   # 秒
MAX_DELAY = 30.0   # 秒
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def should_retry(exc: Exception) -> bool:
    """可重试异常：429 / 5xx / 超时 / 连接错误。纯函数，独立可测。"""
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    return isinstance(exc, (APITimeoutError, APIConnectionError))


def backoff_delay(attempt: int) -> float:
    """指数退避 + 抖动：min(1s × 2^attempt, 30s) + U(0, 0.5s)。"""
    return min(BASE_DELAY * (2 ** attempt), MAX_DELAY) + random.uniform(0, 0.5)


def retry_after_seconds(exc: Exception) -> float | None:
    """429 时优先读 Retry-After 头（秒）。"""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _mentions_reasoning(exc: Exception) -> bool:
    """错误信息（含 body）是否与 reasoning_content 相关（V4 思考模式 400 特征）。"""
    import json as _json
    text = str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        text += _json.dumps(body)
    elif isinstance(body, (str, bytes)):
        text += str(body)
    return "reasoning_content" in text


def _strip_reasoning(messages: list[dict]) -> list[dict]:
    """剥离消息中的 reasoning_content 字段（V4 思考模式 400 的兜底）。"""
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


@dataclass
class ChatResponse:
    message: dict       # 完整 assistant 消息（原样回传，保留 tool_calls 等字段）
    content: str
    finish_reason: str
    truncated: bool     # finish_reason == "length"


class LLMClient:
    """OpenAI 兼容客户端（对接 DeepSeek）。client/sleep_fn 可注入以便测试。"""

    def __init__(self, settings: Settings, client=None, sleep_fn=time.sleep):
        self.settings = settings
        self._sleep = sleep_fn
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        kwargs: dict = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            "extra_body": {"thinking": {"type": self.settings.thinking}},
        }
        if tools:
            kwargs["tools"] = tools
        # 注意：不发送 tool_choice / temperature（DeepSeek V4 兼容性约束）
        stripped = False
        attempt = 0
        while True:
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                # 特判：reasoning_content 相关 400 → 剥离该字段后重试一次
                if (not stripped and isinstance(exc, APIStatusError)
                        and exc.status_code == 400 and _mentions_reasoning(exc)):
                    kwargs["messages"] = _strip_reasoning(messages)
                    stripped = True
                    continue
                if attempt < MAX_RETRIES and should_retry(exc):
                    delay = retry_after_seconds(exc) or backoff_delay(attempt)
                    self._sleep(delay)
                    attempt += 1
                    continue
                raise RetryExhausted(f"LLM 调用失败（重试 {attempt} 次后放弃）：{exc}") from exc
        choice = resp.choices[0]
        msg = choice.message
        message = msg if isinstance(msg, dict) else msg.model_dump(exclude_none=True)
        content = message.get("content") or ""
        finish = getattr(choice, "finish_reason", "stop")
        return ChatResponse(message=message, content=content, finish_reason=finish, truncated=finish == "length")
