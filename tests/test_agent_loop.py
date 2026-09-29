"""Agent 循环测试：用脚本化 FakeLLM 离线验证完整闭环。"""

import pytest

from conftest import FakeLLM, make_response, make_tool_call_message

from codegen.agent import AgentResult, NeedInput, resume_agent, run_agent
from codegen.prompts import few_shot_messages


def test_few_shot_messages_are_well_formed():
    """格式不变量：few-shot 里每条 tool 消息都有配对 assistant，且以纯文本结尾。"""
    msgs = few_shot_messages()
    seen = set()
    for m in msgs:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                assert tc["id"]
                seen.add(tc["id"])
        elif m["role"] == "tool":
            assert m["tool_call_id"] in seen
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls")


def test_agent_loop_writes_file_and_answers(settings, memory):
    """完整闭环：write_file → run_code → 最终回答，文件真实落盘、记忆配对合法。"""
    fake = FakeLLM([
        make_response(make_tool_call_message(
            "c1", "write_file",
            {"path": "qsort.py", "content": "def qsort(a):\n    return sorted(a)\n"})),
        make_response(make_tool_call_message(
            "c2", "run_code",
            {"language": "python", "code": "from qsort import qsort\nprint(qsort([3, 1, 2]))"})),
        make_response({"role": "assistant", "content": "已生成 qsort.py 并验证通过：[1, 2, 3]"}),
    ])
    result = run_agent("用 Python 写快速排序并验证", memory, fake, settings)
    assert isinstance(result, AgentResult)
    assert "验证通过" in result.answer
    assert (settings.workspace / "qsort.py").exists()
    # 记忆 = user + 2×(assistant+tool) + final assistant = 6 条
    assert len(memory.messages) == 6
    assert memory.messages[0]["role"] == "user"
    # 配对不变量
    seen = set()
    for m in memory.messages:
        if m["role"] == "assistant":
            seen.update(tc["id"] for tc in m.get("tool_calls", []))
        elif m["role"] == "tool":
            assert m["tool_call_id"] in seen


def test_agent_loop_backfills_tool_error(settings, memory):
    """工具失败 → 错误文本回传模型（模型在第二轮请求中能看到 Error: 前缀）。"""
    fake = FakeLLM([
        make_response(make_tool_call_message("c1", "write_file", {"path": "x.py"})),  # 缺 content
        make_response({"role": "assistant", "content": "抱歉，我重新写一次"}),
    ])
    result = run_agent("写个文件", memory, fake, settings)
    assert isinstance(result, AgentResult)
    # 按 tool_call_id 定位本轮真实回传的 tool 消息（跳过 few-shot 示例）
    tool_msg = next(m for m in fake.calls[1]["messages"]
                    if m["role"] == "tool" and m["tool_call_id"] == "c1")
    assert tool_msg["content"].startswith("Error: 参数缺失")


def test_agent_loop_handles_invalid_json_args(settings, memory):
    """模型输出非法 JSON 参数 → 回传错误，循环不崩溃。"""
    bad = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "write_file", "arguments": "这不是JSON"}}],
    }
    fake = FakeLLM([make_response(bad), make_response({"role": "assistant", "content": "好的"})])
    run_agent("写", memory, fake, settings)
    tool_msg = next(m for m in fake.calls[1]["messages"]
                    if m["role"] == "tool" and m["tool_call_id"] == "c1")
    assert "不是合法 JSON" in tool_msg["content"]


def test_agent_loop_unknown_tool(settings, memory):
    fake = FakeLLM([
        make_response(make_tool_call_message("c1", "hack_the_planet", {"x": 1})),
        make_response({"role": "assistant", "content": "明白"}),
    ])
    run_agent("hi", memory, fake, settings)
    tool_msg = next(m for m in fake.calls[1]["messages"]
                    if m["role"] == "tool" and m["tool_call_id"] == "c1")
    assert "未知工具" in tool_msg["content"]


def test_agent_loop_max_iterations(settings, memory):
    """模型一直调用工具 → 迭代上限兜底终止。"""
    responses = [make_response(make_tool_call_message(f"c{i}", "list_files", {}))
                 for i in range(settings.max_iterations)]
    fake = FakeLLM(responses)
    result = run_agent("hi", memory, fake, settings)
    assert isinstance(result, AgentResult)
    assert "最大迭代次数" in result.answer


def test_agent_loop_ask_user_and_resume(settings, memory):
    """ask_user 挂起 → 用户答复后原消息序列续接，不丢上下文。"""
    fake = FakeLLM([
        make_response(make_tool_call_message(
            "c1", "ask_user", {"question": "要递归还是迭代实现？"})),
        make_response(make_tool_call_message(
            "c2", "write_file",
            {"path": "fib.py", "content": "def fib(n):\n    return n\n"})),
        make_response({"role": "assistant", "content": "已按迭代方式生成 fib.py"}),
    ])
    result = run_agent("写斐波那契", memory, fake, settings)
    assert isinstance(result, NeedInput)
    assert "递归" in result.question
    result2 = resume_agent("迭代", result.messages, memory, fake, settings)
    assert isinstance(result2, AgentResult)
    assert (settings.workspace / "fib.py").exists()
    # 记忆包含完整往返（两次 user：初始提问 + 答复）
    roles = [m["role"] for m in memory.messages]
    assert roles.count("user") == 2


def test_agent_loop_no_tools(settings, memory):
    fake = FakeLLM([make_response({"role": "assistant", "content": "你好！"})])
    result = run_agent("hi", memory, fake, settings, tools=[])
    assert isinstance(result, AgentResult)
    assert fake.calls[0]["tools"] == []


def test_truncated_answer_gets_note(settings, memory):
    fake = FakeLLM([make_response({"role": "assistant", "content": "部分内容"},
                                  finish="length", truncated=True)])
    result = run_agent("hi", memory, fake, settings)
    assert "截断" in result.answer


def test_memory_context_included_in_next_turn(settings, memory):
    """第二轮请求应包含第一轮的记忆（上下文记忆生效）。"""
    fake = FakeLLM([
        make_response({"role": "assistant", "content": "第一轮回答"}),
        make_response({"role": "assistant", "content": "第二轮回答"}),
    ])
    run_agent("第一问", memory, fake, settings)
    run_agent("第二问", memory, fake, settings)
    second_call_msgs = fake.calls[1]["messages"]
    contents = [m.get("content") for m in second_call_msgs]
    assert "第一问" in contents and "第一轮回答" in contents
