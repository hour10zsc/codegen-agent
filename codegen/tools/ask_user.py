"""ask_user 工具：向用户提问，Agent 循环暂停等待答复（人机协作模式）。"""

from .base import ToolContext, ToolResult


def ask_user(args: dict, ctx: ToolContext) -> ToolResult:
    question = args.get("question")
    if not question:
        return ToolResult.failure("参数缺失：ask_user 需要 question")
    return ToolResult.need_input(str(question))
