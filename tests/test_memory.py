"""记忆模块测试：滑动窗口配对不变量、jsonl 往返、坏行容错、token 估算。"""

from codegen.memory import ConversationMemory, estimated_tokens, find_cut_point


def _msg(role, content=None, tool_call_id=None, tool_calls=None):
    m = {"role": role}
    if content is not None:
        m["content"] = content
    if tool_call_id is not None:
        m["tool_call_id"] = tool_call_id
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


def assert_paired(messages):
    """配对不变量：每条 tool 消息的 tool_call_id 都在之前的 assistant 消息中出现。"""
    seen = set()
    for m in messages:
        if m["role"] == "assistant":
            seen.update(tc["id"] for tc in m.get("tool_calls", []))
        elif m["role"] == "tool":
            assert m["tool_call_id"] in seen, f"孤儿 tool 消息：{m}"


def test_find_cut_point_within_window():
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert find_cut_point(msgs, 2) == 0


def test_find_cut_point_never_splits_tool_pair():
    # 超过窗口时，第一个「assistant(tool_calls) → tool」往返必须被整体丢弃
    msgs = [
        _msg("user", "u1"),
        _msg("assistant", tool_calls=[{"id": "c1"}]),
        _msg("tool", "t1", tool_call_id="c1"),
        _msg("assistant", "final1"),
        _msg("user", "u2"),
        _msg("assistant", "final2"),
    ]
    cut = find_cut_point(msgs, 3)
    window = msgs[cut:]
    assert len(window) <= 3
    assert_paired(window)
    assert cut >= 3  # 第一个工具往返被整体丢弃，而不是从中切开


def test_find_cut_point_keeps_last_user():
    msgs = [_msg("user", "u1"), _msg("assistant", "a1"),
            _msg("user", "u2"), _msg("assistant", "a2")]
    cut = find_cut_point(msgs, 2)
    assert msgs[cut]["role"] == "user"
    assert msgs[cut]["content"] == "u2"


def test_estimated_tokens_counts_cjk():
    assert estimated_tokens([{"role": "user", "content": "你好"}]) == 2
    assert estimated_tokens([{"role": "user", "content": "abcd"}]) == 1


def test_save_load_roundtrip(settings, memory, tmp_path):
    memory.messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
    ]
    path = tmp_path / "s.jsonl"
    memory.save(path)
    other = ConversationMemory(settings)
    other.load(path)
    assert other.messages == memory.messages


def test_load_skips_bad_lines(settings, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"role": "user", "content": "ok"}\n这不是JSON\n{"role": "assistant", "content": "a"}\n',
        encoding="utf-8",
    )
    m = ConversationMemory(settings)
    m.load(path)
    assert len(m.messages) == 2


def test_commit_trims_window(settings, memory):
    for i in range(30):
        memory.commit([{"role": "user", "content": f"u{i}"},
                       {"role": "assistant", "content": f"a{i}"}])
    assert len(memory.messages) <= settings.memory_max_messages
    # 窗口内最后一条 user 消息仍在
    assert any(m["role"] == "user" for m in memory.messages)


def test_trim_keeps_tool_pairs_intact(settings, memory):
    for i in range(10):
        memory.commit([
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "r"},
            {"role": "assistant", "content": f"final{i}"},
        ])
    assert_paired(memory.messages)
