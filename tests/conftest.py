"""pytest 共享夹具与测试辅助。

FakeLLM 为脚本化假模型：按序弹出预设 ChatResponse 并记录每次请求，
使 Agent 循环测试完全离线、确定。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codegen.config import Settings
from codegen.llm import ChatResponse
from codegen.memory import ConversationMemory


@pytest.fixture
def settings(tmp_path):
    s = Settings(api_key="sk-test")
    s.workspace = tmp_path / "workspace"
    s.sessions_dir = tmp_path / "sessions"
    s.workspace.mkdir(parents=True)
    s.sessions_dir.mkdir(parents=True)
    s.memory_max_messages = 20
    s.run_timeout = 5
    return s


@pytest.fixture
def memory(settings):
    return ConversationMemory(settings)


class FakeLLM:
    """脚本化 LLM：按序弹出预设 ChatResponse，并记录每次请求。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return ChatResponse(
            message={"role": "assistant", "content": "（脚本耗尽）"},
            content="（脚本耗尽）", finish_reason="stop", truncated=False,
        )


def make_tool_call_message(tool_call_id: str, name: str, arguments: dict) -> dict:
    """构造一条带合法 tool_calls 结构的 assistant 消息。"""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }


def make_response(message: dict, finish: str = "stop", truncated: bool = False) -> ChatResponse:
    return ChatResponse(
        message=message,
        content=message.get("content") or "",
        finish_reason=finish,
        truncated=truncated,
    )
