"""Streamlit Web 界面：聊天式 UI，与 CLI 共用 codegen.agent 核心。

Streamlit 注意事项：
- LLM 客户端用 st.cache_resource 缓存（openai 客户端不可 pickle，
  不能放进 session_state）；
- session_state 只存可序列化数据（记忆消息、显示消息、挂起状态）。
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # 允许从任意目录 streamlit run webapp/app.py

from codegen.agent import AgentHooks, NeedInput, resume_agent, run_agent
from codegen.config import Settings
from codegen.errors import RetryExhausted
from codegen.llm import LLMClient
from codegen.memory import ConversationMemory

st.set_page_config(page_title="代码生成 Agent", page_icon="🤖")

EXAMPLE_PROMPTS = {
    "快速排序并验证": "用 Python 写一个快速排序函数，并用 run_code 验证它排序正确。",
    "斐波那契数列": "写一个斐波那契数列生成器，并验证前 10 项输出正确。",
    "统计代码行数": "写一个程序统计 workspace 中所有 .py 文件的总行数，并运行验证。",
}


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, base_url: str, model: str, thinking: str) -> LLMClient:
    """LLM 客户端缓存：避免每次 rerun 重建连接。"""
    s = Settings(api_key=api_key, base_url=base_url, model=model, thinking=thinking)
    return LLMClient(s)


def init_state(settings: Settings) -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(settings)
    if "display" not in st.session_state:
        st.session_state.display: list[dict] = []
    if "pending" not in st.session_state:
        st.session_state.pending = None  # NeedInput 的挂起消息序列


def sidebar(settings: Settings) -> None:
    memory = st.session_state.memory
    with st.sidebar:
        st.header("会话")
        if st.button("清空当前会话", use_container_width=True):
            memory.clear()
            st.session_state.display = []
            st.session_state.pending = None
            st.rerun()
        save_name = st.text_input("会话名", value="default")
        c1, c2 = st.columns(2)
        if c1.button("保存", use_container_width=True):
            memory.save(settings.sessions_dir / f"{save_name}.jsonl")
            st.success(f"已保存到 sessions/{save_name}.jsonl")
        if c2.button("加载", use_container_width=True):
            path = settings.sessions_dir / f"{save_name}.jsonl"
            if path.exists():
                memory.load(path)
                st.success(f"已加载（{len(memory.messages)} 条消息）")
            else:
                st.warning("会话不存在")
        st.divider()
        st.caption(
            f"模型：{settings.model}\n\n"
            f"工作区：{settings.workspace}\n\n"
            f"记忆：{len(memory.messages)} 条消息"
        )


def run_turn(prompt: str, llm: LLMClient, settings: Settings):
    """执行一轮对话并渲染结果（含工具过程）。"""
    memory = st.session_state.memory
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_logs: list[dict] = []
    hooks = AgentHooks(
        on_tool_start=lambda name, args: tool_logs.append({"name": name, "args": args}),
        on_tool_end=lambda name, result: tool_logs.append({"name": name, "result": result.as_string()}),
    )
    with st.chat_message("assistant"):
        try:
            with st.spinner("思考中…"):
                if st.session_state.pending is not None:
                    pending = st.session_state.pending
                    st.session_state.pending = None
                    result = resume_agent(prompt, pending, memory, llm, settings, hooks=hooks)
                else:
                    result = run_agent(prompt, memory, llm, settings, hooks=hooks)
            if isinstance(result, NeedInput):
                text = f"❓ {result.question}"
                st.markdown(text)
                st.session_state.pending = result.messages
            else:
                text = result.answer
                st.markdown(text)
            st.session_state.display.append({"role": "assistant", "content": text})
        except RetryExhausted as e:
            st.error(str(e))
        if tool_logs:
            with st.expander(f"🔧 工具调用过程（{len(tool_logs) // 2} 次）", expanded=False):
                for log in tool_logs:
                    if "args" in log:
                        st.markdown(f"**调用 {log['name']}**")
                        st.code(json.dumps(log["args"], ensure_ascii=False, indent=2), language="json")
                    else:
                        st.caption("返回：")
                        st.code(log["result"], language="text")
    # 展示列表只保留最近 50 条，防止无限增长
    st.session_state.display = st.session_state.display[-50:]


def main() -> None:
    settings = Settings.load()
    if not settings.api_key:
        st.error(
            "未检测到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入 "
            "API Key（platform.deepseek.com 申请）后重启。"
        )
        st.stop()
    init_state(settings)
    llm = get_llm(settings.api_key, settings.base_url, settings.model, settings.thinking)
    sidebar(settings)

    st.title("🤖 代码生成 Agent")
    st.caption("用自然语言描述需求，Agent 会写入文件、运行验证并汇报结果。")
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, (label, prompt) in zip(cols, EXAMPLE_PROMPTS.items()):
        if col.button(label, use_container_width=True):
            st.session_state.pending_prompt = prompt

    for m in st.session_state.display:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("描述你想要的功能…") or st.session_state.pop("pending_prompt", None)
    if prompt:
        run_turn(prompt, llm, settings)


main()
