"""工具抽象层：统一规格、统一结果、统一分发。

新增工具只需在 tools/__init__.py 的注册表中加一条 ToolSpec，
Agent 循环零改动（开放-封闭原则，见 Design.md §4）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ToolContext:
    """工具执行上下文：工作目录与超时等运行时约束。"""
    workspace: Path
    run_timeout: int = 15


@dataclass
class ToolResult:
    """工具执行结果。ok=False 时 as_string() 输出 Error: 前缀，模型读得懂。"""
    ok: bool
    content: str
    kind: str = "output"  # output | need_input

    @classmethod
    def success(cls, content: str) -> "ToolResult":
        return cls(ok=True, content=content)

    @classmethod
    def failure(cls, content: str) -> "ToolResult":
        return cls(ok=False, content=content)

    @classmethod
    def need_input(cls, question: str) -> "ToolResult":
        return cls(ok=True, content=question, kind="need_input")

    def as_string(self) -> str:
        """给模型看的结果文本：失败带 Error: 前缀，方便模型识别并自我修正。"""
        return self.content if self.ok else f"Error: {self.content}"


@dataclass
class ToolSpec:
    """工具规格：name/description/parameters 生成 OpenAI tools schema，fn 执行。"""
    name: str
    description: str
    parameters: dict  # JSON Schema（OpenAI function 格式）
    fn: Callable[[dict, ToolContext], ToolResult]
