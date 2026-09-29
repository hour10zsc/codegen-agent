"""run_code 工具：执行代码片段验证正确性（生成-验证闭环的关键）。"""

from .. import sandbox
from ..errors import ToolError
from .base import ToolContext, ToolResult


def run_code(args: dict, ctx: ToolContext) -> ToolResult:
    code = args.get("code")
    if not code:
        return ToolResult.failure("参数缺失：run_code 需要 code")
    language = args.get("language", "python")
    filename = args.get("filename")
    try:
        output = sandbox.run_code(code, language, filename, ctx.workspace, ctx.run_timeout)
    except ToolError as e:
        return ToolResult.failure(str(e))
    return ToolResult.success(output)
