"""配置加载：从 .env / 环境变量读取，启动时校验必需项。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _resolve_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    return (PROJECT_ROOT / Path(raw)).resolve()


@dataclass
class Settings:
    """全部可配置项，默认值即 .env.example 的默认值。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"  # deepseek-chat 已于 2026-07-24 停用
    thinking: str = "disabled"        # disabled | enabled
    max_tokens: int = 8192
    max_iterations: int = 8
    workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")
    sessions_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "sessions")
    run_timeout: int = 15
    memory_max_messages: int = 20
    memory_max_tokens: int = 64000

    @classmethod
    def load(cls) -> "Settings":
        """从项目根 .env 加载配置，并确保运行时目录存在。"""
        load_dotenv(PROJECT_ROOT / ".env")
        s = cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            thinking=os.getenv("DEEPSEEK_THINKING", "disabled").strip(),
            max_tokens=_get_int("DEEPSEEK_MAX_TOKENS", 8192),
            max_iterations=_get_int("AGENT_MAX_ITERATIONS", 8),
            workspace=_resolve_path("AGENT_WORKSPACE", "workspace"),
            sessions_dir=_resolve_path("AGENT_SESSIONS_DIR", "sessions"),
            run_timeout=_get_int("RUN_TIMEOUT", 15),
            memory_max_messages=_get_int("MEMORY_MAX_MESSAGES", 20),
            memory_max_tokens=_get_int("MEMORY_MAX_TOKENS", 64000),
        )
        s.workspace.mkdir(parents=True, exist_ok=True)
        s.sessions_dir.mkdir(parents=True, exist_ok=True)
        return s


def require_api_key(settings: Settings) -> None:
    """CLI 启动校验：缺 key 时给出可操作的提示。"""
    if not settings.api_key:
        print("未检测到 DEEPSEEK_API_KEY。")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 填入你的 API Key（platform.deepseek.com 申请）")
        print(f"  3. 重新运行（配置文件：{PROJECT_ROOT / '.env'}）")
        raise SystemExit(1)
