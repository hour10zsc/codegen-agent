"""命令行界面：交互 REPL 与单命令模式，均复用 codegen.agent.run_agent 核心。"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .agent import AgentHooks, NeedInput, resume_agent, run_agent
from .config import Settings, require_api_key
from .errors import RetryExhausted
from .llm import LLMClient
from .memory import ConversationMemory
from .tools import TOOL_REGISTRY

HELP_TEXT = """可用命令：
  /tools            列出全部工具
  /save [名称]      保存当前会话（默认按时间戳命名）
  /load <名称>      加载会话
  /clear            清空上下文记忆
  /history [n]      查看最近 n 条历史（默认 10）
  /quit, /exit      退出
直接输入自然语言即可开始对话。"""


def parse_slash_command(line: str) -> tuple[str, list[str]] | None:
    """解析斜杠命令。行首 / 视为命令，返回 (命令名, 参数)；否则返回 None。"""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


def truncate(text: str, limit: int = 80) -> str:
    """压缩为单行并截断（终端展示用）。"""
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_cli_hooks() -> AgentHooks:
    """工具过程打印回调（演示视频的核心素材）。"""

    def on_tool_start(name: str, args: dict) -> None:
        brief = ", ".join(f"{k}={truncate(str(v), 40)}" for k, v in args.items())
        print(f"  [tool] {name}({brief})")

    def on_tool_end(name: str, result) -> None:
        print(f"         → {truncate(result.as_string())}")

    return AgentHooks(on_tool_start=on_tool_start, on_tool_end=on_tool_end)


class CLI:
    """REPL 外壳：斜杠命令 + 对话轮次，Agent 核心逻辑全在 codegen.agent。"""

    def __init__(self, settings: Settings, llm: LLMClient, memory: ConversationMemory,
                 hooks: AgentHooks, no_tools: bool = False):
        self.settings = settings
        self.llm = llm
        self.memory = memory
        self.hooks = hooks
        self.no_tools = no_tools

    def _tools_arg(self):
        return [] if self.no_tools else None

    def run_turn(self, prompt: str) -> str:
        """运行一轮对话，处理 NeedInput 挂起，返回给用户看的答案。"""
        result = run_agent(prompt, self.memory, self.llm, self.settings,
                           hooks=self.hooks, tools=self._tools_arg())
        while isinstance(result, NeedInput):
            print(f"  [?] {result.question}")
            try:
                answer = input("  你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                return "（已取消）"
            if not answer:
                return "（已取消）"
            result = resume_agent(answer, result.messages, self.memory, self.llm,
                                  self.settings, hooks=self.hooks, tools=self._tools_arg())
        return result.answer

    def handle_slash(self, cmd: str, args: list[str]) -> str | None:
        """处理斜杠命令。返回 None 表示退出 REPL。"""
        if cmd in ("quit", "exit"):
            return None
        if cmd == "help":
            return HELP_TEXT
        if cmd == "tools":
            lines = ["可用工具："]
            for name, spec in TOOL_REGISTRY.items():
                required = ", ".join(spec.parameters.get("required", []))
                lines.append(f"  {name}: {spec.description}（必需参数：{required or '无'}）")
            return "\n".join(lines)
        if cmd == "save":
            name = args[0] if args else datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.settings.sessions_dir / f"{name}.jsonl"
            self.memory.save(path)
            return f"会话已保存到 {path}（{len(self.memory.messages)} 条消息）"
        if cmd == "load":
            if not args:
                return "用法：/load <会话名>"
            path = self.settings.sessions_dir / f"{args[0]}.jsonl"
            if not path.exists():
                return f"会话不存在：{path}"
            self.memory.load(path)
            return f"已加载会话 {args[0]}（{len(self.memory.messages)} 条消息）"
        if cmd == "clear":
            n = len(self.memory.messages)
            self.memory.clear()
            return f"已清空记忆（{n} 条消息）"
        if cmd == "history":
            try:
                n = min(int(args[0]) if args else 10, 50)
            except ValueError:
                return "用法：/history [n]"
            if not self.memory.messages:
                return "记忆为空"
            lines = []
            for m in self.memory.messages[-n:]:
                role = m.get("role")
                content = m.get("content")
                if not content and m.get("tool_calls"):
                    names = [tc["function"]["name"] for tc in m["tool_calls"]]
                    content = f"（调用工具：{', '.join(names)}）"
                lines.append(f"  {role}: {truncate(str(content or ''), 50)}")
            return "\n".join(lines)
        return f"未知命令 /{cmd}，输入 /help 查看帮助"

    def repl(self) -> None:
        print("代码生成 Agent（Homework 1）——输入 /help 查看命令，/quit 退出")
        while True:
            try:
                line = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                return
            if not line:
                continue
            slash = parse_slash_command(line)
            if slash:
                out = self.handle_slash(*slash)
                if out is None:
                    print("再见！")
                    return
                print(out)
                continue
            try:
                answer = self.run_turn(line)
            except RetryExhausted as e:
                print(f"[错误] {e}")
                continue
            print(f"\nAgent > {answer}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codegen", description="代码生成 Agent（软件工程 Homework 1）")
    p.add_argument("prompt", nargs="?", help="单命令模式：直接给出一条任务描述")
    p.add_argument("--workspace", help="覆盖文件工具的工作目录")
    p.add_argument("--model", help="覆盖 .env 中的模型名")
    p.add_argument("--max-iter", type=int, help="覆盖最大迭代次数")
    p.add_argument("--session", help="启动时加载的会话名（sessions/<名称>.jsonl）")
    p.add_argument("--no-tools", action="store_true", help="禁用工具（纯聊天调试模式）")
    p.add_argument("--quiet", action="store_true", help="不打印工具调用过程")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    if args.workspace:
        settings.workspace = Path(args.workspace).resolve()
    if args.model:
        settings.model = args.model
    if args.max_iter:
        settings.max_iterations = args.max_iter
    require_api_key(settings)
    llm = LLMClient(settings)
    memory = ConversationMemory(settings)
    if args.session:
        path = settings.sessions_dir / f"{args.session}.jsonl"
        if path.exists():
            memory.load(path)
        else:
            print(f"[警告] 会话 {args.session} 不存在，从空记忆开始")
    hooks = AgentHooks() if args.quiet else make_cli_hooks()
    cli = CLI(settings, llm, memory, hooks, no_tools=args.no_tools)
    if args.prompt:
        try:
            print(f"Agent > {cli.run_turn(args.prompt)}")
        except RetryExhausted as e:
            print(f"[错误] {e}")
            sys.exit(1)
    else:
        cli.repl()


if __name__ == "__main__":
    main()
