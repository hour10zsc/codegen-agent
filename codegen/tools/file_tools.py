"""文件工具：write_file / read_file / list_files。

所有路径解析后必须仍在 workspace 内（防目录穿越与绝对路径），
见 _resolve_in_workspace。
"""

from pathlib import Path

from ..errors import ToolError
from .base import ToolContext, ToolResult

MAX_FILE_BYTES = 1 * 1024 * 1024
MAX_LIST_DEPTH = 3
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".run"}


def _resolve_in_workspace(ctx: ToolContext, path: str) -> Path:
    """把路径解析到 workspace 内；越界（.. / 绝对路径）一律拒绝。"""
    raw = str(path).replace("\\", "/")  # 兼容 Windows 反斜杠输入
    p = (ctx.workspace / raw).resolve()
    if not p.is_relative_to(ctx.workspace.resolve()):
        raise ToolError("路径越界：只允许在 workspace 内操作，禁止 .. 与绝对路径")
    return p


def write_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    content = args.get("content")
    if not path or content is None:
        return ToolResult.failure("参数缺失：write_file 需要 path 和 content")
    try:
        p = _resolve_in_workspace(ctx, str(path))
    except ToolError as e:
        return ToolResult.failure(str(e))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return ToolResult.failure(f"写入失败：{e}")
    rel = p.relative_to(ctx.workspace)
    lines = len(content.splitlines())
    return ToolResult.success(f"已写入 {rel}：{len(content.encode('utf-8'))} 字节，{lines} 行")


def read_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    if not path:
        return ToolResult.failure("参数缺失：read_file 需要 path")
    try:
        p = _resolve_in_workspace(ctx, str(path))
    except ToolError as e:
        return ToolResult.failure(str(e))
    if not p.exists():
        return ToolResult.failure(f"文件不存在：{path}（可用 list_files 查看现有文件）")
    if p.is_dir():
        return ToolResult.failure(f"{path} 是目录，请用 list_files 查看内容")
    if p.stat().st_size > MAX_FILE_BYTES:
        return ToolResult.failure("文件超过 1MB，拒绝整读（可用 offset/limit 分段读取）")
    try:
        data = p.read_bytes()
    except OSError as e:
        return ToolResult.failure(f"读取失败：{e}")
    if b"\x00" in data[:4096]:
        return ToolResult.failure("疑似二进制文件，拒绝读取")
    lines = data.decode("utf-8", errors="replace").splitlines()
    try:
        offset = max(0, int(args.get("offset", 0)))
        limit = min(max(1, int(args.get("limit", 100))), 500)
    except (TypeError, ValueError):
        return ToolResult.failure("offset/limit 必须是整数")
    selected = lines[offset:offset + limit]
    numbered = "\n".join(f"{i + 1:>5} | {line}" for i, line in enumerate(selected, start=offset))
    if offset + limit < len(lines):
        numbered += f"\n… 共 {len(lines)} 行，已显示第 {offset + 1}-{offset + len(selected)} 行，可用 offset 继续读取"
    return ToolResult.success(numbered or "（空文件）")


def list_files(args: dict, ctx: ToolContext) -> ToolResult:
    root = ctx.workspace
    if args.get("path"):
        try:
            root = _resolve_in_workspace(ctx, str(args["path"]))
        except ToolError as e:
            return ToolResult.failure(str(e))
    if not root.exists():
        return ToolResult.failure(f"目录不存在：{args.get('path', 'workspace')}")
    if not root.is_dir():
        return ToolResult.failure(f"{args.get('path')} 不是目录")
    out = ["." if root == ctx.workspace else str(root.relative_to(ctx.workspace))]

    def walk(d: Path, depth: int) -> None:
        if depth >= MAX_LIST_DEPTH:
            out.append("  " * (depth + 1) + "…（更深层已省略）")
            return
        entries = sorted(d.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        for e in entries:
            if e.name.startswith(".") or (e.is_dir() and e.name in SKIP_DIRS):
                continue
            if e.is_dir():
                out.append("  " * (depth + 1) + e.name + "/")
                walk(e, depth + 1)
            else:
                out.append("  " * (depth + 1) + e.name)

    walk(root, 0)
    return ToolResult.success("\n".join(out))
