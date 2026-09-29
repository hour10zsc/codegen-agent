"""CLI 斜杠命令解析测试（纯函数）。"""

from codegen.cli import parse_slash_command, truncate


def test_parse_slash_command_basic():
    assert parse_slash_command("/save demo") == ("save", ["demo"])
    assert parse_slash_command("/tools") == ("tools", [])
    assert parse_slash_command("/history 5") == ("history", ["5"])


def test_parse_slash_command_case_insensitive():
    assert parse_slash_command("/QUIT") == ("quit", [])


def test_parse_slash_command_normal_input():
    assert parse_slash_command("用 Python 写一个快速排序") is None
    assert parse_slash_command("") is None
    assert parse_slash_command("/") is None


def test_truncate_short_text_unchanged():
    assert truncate("短文本") == "短文本"


def test_truncate_long_text():
    assert len(truncate("x" * 200)) <= 80
    assert truncate("x" * 200).endswith("…")


def test_truncate_newlines_collapsed():
    assert "\n" not in truncate("a\nb\nc")
