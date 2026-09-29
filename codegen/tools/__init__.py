"""工具注册表：新增工具只需在 register_tools() 中加一条 ToolSpec，
Agent 循环零改动（开放-封闭原则，见 Design.md §4）。

对外接口：
- TOOL_REGISTRY      名称 → ToolSpec
- tool_schemas()     生成 OpenAI tools 参数
- dispatch()         按名称分发执行
"""

from .ask_user import ask_user
from .base import ToolContext, ToolResult, ToolSpec
from .file_tools import list_files, read_file, write_file
from .run_code import run_code

TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register_tools() -> None:
    """注册全部内置工具（幂等）。"""
    specs = [
        ToolSpec(
            name="write_file",
            description="把代码/文本写入 workspace 内的文件（生成代码的第一步，支持子目录，会自动创建）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对 workspace 的文件路径，如 main.py 或 src/util.py"},
                    "content": {"type": "string", "description": "文件完整内容"},
                },
                "required": ["path", "content"],
            },
            fn=write_file,
        ),
        ToolSpec(
            name="read_file",
            description="读取 workspace 内文本文件的内容（带行号，可分段）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对 workspace 的文件路径"},
                    "offset": {"type": "integer", "description": "起始行（从 0 开始，默认 0）"},
                    "limit": {"type": "integer", "description": "最多读取行数（默认 100，上限 500）"},
                },
                "required": ["path"],
            },
            fn=read_file,
        ),
        ToolSpec(
            name="list_files",
            description="列出 workspace 内的文件树（深度最多 3 层，自动忽略缓存与隐藏目录）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对 workspace 的子目录（可选，默认根目录）"},
                },
                "required": [],
            },
            fn=list_files,
        ),
        ToolSpec(
            name="run_code",
            description="在受限沙箱中执行一段代码并返回退出码与输出，用于验证生成代码的正确性；workspace 已在模块搜索路径中，可直接 import 其中已生成的文件",
            parameters={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["python", "javascript"], "description": "语言（默认 python）"},
                    "code": {"type": "string", "description": "要执行的完整代码（含 import/调用）"},
                    "filename": {"type": "string", "description": "写入临时目录的文件名（可选，默认 main.py/main.js）"},
                },
                "required": ["code"],
            },
            fn=run_code,
        ),
        ToolSpec(
            name="ask_user",
            description="需求不明确时向用户提问（例如实现方案二选一），不要擅自猜测",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "向用户提出的问题"},
                },
                "required": ["question"],
            },
            fn=ask_user,
        ),
    ]
    for spec in specs:
        TOOL_REGISTRY.setdefault(spec.name, spec)


def tool_schemas() -> list[dict]:
    """生成 OpenAI 格式的 tools 参数。"""
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in TOOL_REGISTRY.values()
    ]


def dispatch(name: str, args: dict, ctx: ToolContext) -> ToolResult:
    """按名称分发执行工具。未知工具返回失败结果（由 Agent 循环回传模型）。"""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return ToolResult.failure(f"未知工具 {name}，可用工具：{', '.join(TOOL_REGISTRY)}")
    return spec.fn(args, ctx)


register_tools()
