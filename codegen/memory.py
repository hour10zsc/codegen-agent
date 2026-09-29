"""会话记忆：滑动窗口 + jsonl 持久化。

配对不变量：含 tool_calls 的 assistant 消息与其 tool 消息（按
tool_call_id 配对）必须成对保留或丢弃——孤立的 tool_call 回传给
API 会得到 400 错误。窗口裁剪由 find_cut_point() 保证这一点。
"""

import json
import os
from pathlib import Path

from .config import Settings


def estimated_tokens(messages: list[dict]) -> int:
    """粗略估算 token 数（DeepSeek 未公开 tokenizer，用经验公式）：
    中日韩字符按 1 token 计，其余字符 4 个 ≈ 1 token。"""

    def count(text: str) -> int:
        cjk = sum(1 for ch in text if "⺀" <= ch <= "鿿" or "＀" <= ch <= "￯")
        return cjk + (len(text) - cjk) // 4

    total = 0
    for m in messages:
        total += count(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += count(str(fn.get("name") or "")) + count(str(fn.get("arguments") or ""))
        if m.get("role") == "tool":
            total += count(str(m.get("tool_call_id") or ""))
    return total


def _valid_start(messages: list[dict], start: int) -> bool:
    """从 start 开始扫描：每条 tool 消息的 tool_call_id 都能在区间内
    找到对应的 assistant 消息（无孤儿 tool 消息）。"""
    seen: set = set()
    for m in messages[start:]:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                seen.add(tc.get("id"))
        elif role == "tool" and m.get("tool_call_id") not in seen:
            return False
    return True


def find_cut_point(messages: list[dict], max_messages: int) -> int:
    """滑动窗口切割点：保留尾部 ≤ max_messages 条，且不产生孤儿 tool 消息。

    从窗口尾部对应的起点开始，若破坏配对不变量则右移起点（连同整个
    工具往返一起丢弃）；同时保证窗口内至少有一条 user 消息。
    消息量小（< 数百条），O(n²) 可接受，正确性优先。
    """
    n = len(messages)
    if n <= max_messages:
        return 0
    for cut in range(n - max_messages, n):
        if _valid_start(messages, cut) and any(m.get("role") == "user" for m in messages[cut:]):
            return cut
    # 兜底：窗口极小导致尾部无 user 消息时，回退到最后一个 user 消息
    # 之前最近的合法起点（宁可略超窗口大小，也不破坏配对合法性）
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=n - 1)
    cut = last_user
    while cut > 0 and not _valid_start(messages, cut):
        cut -= 1
    return cut


class ConversationMemory:
    """滑动窗口会话记忆（OpenAI 消息格式），支持 jsonl 持久化。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.messages: list[dict] = []

    def commit(self, full_history: list[dict]) -> None:
        """一轮对话结束后，用完整历史（含新的一轮）替换记忆并裁剪。"""
        self.messages = list(full_history)
        self.trim()

    def trim(self) -> None:
        """双重预算裁剪：消息条数（max_messages）+ token 估算（max_tokens）。"""
        limit = self.settings.memory_max_messages
        while True:
            cut = find_cut_point(self.messages, limit)
            window = self.messages[cut:]
            if estimated_tokens(window) <= self.settings.memory_max_tokens or limit <= 4:
                self.messages = window
                return
            limit -= 2

    def clear(self) -> None:
        self.messages = []

    def save(self, path: Path) -> None:
        """保存到 jsonl（每行一条完整消息）。临时文件 + os.replace 原子写。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for m in self.messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        """从 jsonl 加载；坏行跳过并告警，不中断加载。"""
        loaded: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[警告] {path} 第 {i} 行损坏，已跳过")
        self.messages = loaded
