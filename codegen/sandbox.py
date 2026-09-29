"""受限代码执行沙箱（作业级，不是安全沙箱）。

提供的隔离：独立临时目录、执行超时、输出截断、环境变量白名单、
超时后清理整个进程树。不应执行不可信输入；生产级隔离需 Docker/虚拟机。
"""

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .errors import ToolError

LANGUAGES = {
    "python": {"default_file": "main.py", "cmd": ["{python}", "{file}"]},
    "javascript": {"default_file": "main.js", "cmd": ["node", "{file}"]},
}
MAX_OUTPUT_CHARS = 4000
MAX_TIMEOUT = 60
ENV_WHITELIST = {
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATH", "TEMP", "TMP",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "COMSPEC", "PATHEXT", "OS",
}


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（输出过长，已截断；总长 {len(text)} 字符）"


def _merge(stdout: str, stderr: str) -> str:
    parts = []
    if stdout:
        parts.append("[stdout]\n" + stdout.rstrip())
    if stderr:
        parts.append("[stderr]\n" + stderr.rstrip())
    return _truncate("\n".join(parts) or "（无输出）")


def _kill_process_tree(pid: int) -> None:
    """Windows 下 subprocess 超时只杀直接子进程，这里杀整棵进程树。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            import signal
            os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass  # 清理尽力而为，不影响主流程


def run_code(code: str, language: str, filename: str | None, workspace: Path, timeout: int) -> str:
    """在 workspace/.run/<随机>/ 中执行一段代码，返回「退出码 + 输出」文本。"""
    if language not in LANGUAGES:
        raise ToolError(f"不支持的语言 {language}，可用：{', '.join(LANGUAGES)}")
    cfg = LANGUAGES[language]
    file = filename or cfg["default_file"]
    if Path(file).name != file:
        raise ToolError("filename 只能是文件名，不能带路径")
    if language == "python":
        exe = sys.executable  # 与 Agent 同解释器，保证存在
    else:
        exe = shutil.which(cfg["cmd"][0])
        if exe is None:
            raise ToolError(f"未找到解释器 {cfg['cmd'][0]}，请确认已安装并加入 PATH")
    timeout = max(1, min(int(timeout), MAX_TIMEOUT))
    run_dir = workspace / ".run" / uuid.uuid4().hex[:8]
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (run_dir / file).write_text(code, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k in ENV_WHITELIST}
        env["PYTHONIOENCODING"] = "utf-8"
        # 执行目录是临时目录，把 workspace 加入模块搜索路径，
        # 让 write_file 生成的代码可以直接 import / require
        env["PYTHONPATH"] = str(workspace)
        env["NODE_PATH"] = str(workspace)
        cmd = [p.replace("{python}", exe).replace("{file}", file) for p in cfg["cmd"]]
        popen_kwargs = dict(
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            cwd=run_dir, env=env,
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            out, err = proc.communicate()  # 回收僵尸进程
            return f"执行超时（>{timeout}s），已强制终止进程树"
        return f"退出码 {proc.returncode}\n{_merge(out, err)}"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
