"""工具层测试：路径越界拒绝、读写、列表深度、沙箱超时与输出截断。"""

import time

import pytest

from codegen.tools import dispatch
from codegen.tools.base import ToolContext


@pytest.fixture
def ctx(settings):
    return ToolContext(workspace=settings.workspace, run_timeout=3)


def test_write_and_read_roundtrip(ctx):
    r = dispatch("write_file", {"path": "hello.py", "content": "print('hi')\n"}, ctx)
    assert r.ok and "hello.py" in r.content
    assert (ctx.workspace / "hello.py").exists()
    r2 = dispatch("read_file", {"path": "hello.py"}, ctx)
    assert r2.ok and "print('hi')" in r2.content


def test_write_file_rejects_traversal(ctx):
    r = dispatch("write_file", {"path": "../evil.py", "content": "x"}, ctx)
    assert not r.ok
    assert "越界" in r.content
    assert not (ctx.workspace.parent / "evil.py").exists()


def test_write_file_rejects_absolute_path(ctx, tmp_path):
    abs_path = tmp_path / "abs.py"
    r = dispatch("write_file", {"path": str(abs_path), "content": "x"}, ctx)
    assert not r.ok
    assert not abs_path.exists()


def test_write_file_missing_params(ctx):
    r = dispatch("write_file", {"path": "x.py"}, ctx)
    assert not r.ok and "参数缺失" in r.content


def test_read_file_missing(ctx):
    r = dispatch("read_file", {"path": "nope.py"}, ctx)
    assert not r.ok and "不存在" in r.content


def test_read_file_offset_limit(ctx):
    (ctx.workspace / "lines.txt").write_text(
        "\n".join(f"line{i}" for i in range(10)), encoding="utf-8")
    r = dispatch("read_file", {"path": "lines.txt", "offset": 8, "limit": 2}, ctx)
    assert r.ok
    assert "line8" in r.content and "line9" in r.content
    assert "line7" not in r.content


def test_list_files_depth_limit(ctx):
    (ctx.workspace / "a" / "b" / "c" / "d").mkdir(parents=True)
    (ctx.workspace / "a" / "deep.py").write_text("", encoding="utf-8")
    r = dispatch("list_files", {}, ctx)
    assert r.ok
    assert "a/" in r.content and "deep.py" in r.content
    assert "d/" not in r.content  # 深度超过 3 层被截断


def test_run_code_python(ctx):
    r = dispatch("run_code", {"language": "python", "code": "print(1 + 1)"}, ctx)
    assert r.ok and "退出码 0" in r.content and "2" in r.content


def test_run_code_can_import_workspace_files(ctx):
    """沙箱临时目录应能 import workspace 里生成的文件（PYTHONPATH）。"""
    dispatch("write_file", {"path": "lib.py", "content": "def twice(x):\n    return x * 2\n"}, ctx)
    r = dispatch("run_code", {"language": "python", "code": "from lib import twice\nprint(twice(21))"}, ctx)
    assert r.ok and "退出码 0" in r.content and "42" in r.content


def test_run_code_timeout(ctx):
    start = time.time()
    r = dispatch("run_code", {"language": "python", "code": "import time\ntime.sleep(30)"}, ctx)
    elapsed = time.time() - start
    assert r.ok and "超时" in r.content
    assert elapsed < 10  # run_timeout=3s，远小于 30s 睡眠


def test_run_code_truncates_output(ctx):
    r = dispatch("run_code", {"language": "python", "code": "print('x' * 6000)"}, ctx)
    assert r.ok and "截断" in r.content


def test_run_code_unsupported_language(ctx):
    r = dispatch("run_code", {"language": "ruby", "code": "puts 1"}, ctx)
    assert not r.ok and "不支持" in r.content


def test_unknown_tool(ctx):
    r = dispatch("hack_the_planet", {}, ctx)
    assert not r.ok and "未知工具" in r.content


def test_ask_user_needs_input(ctx):
    r = dispatch("ask_user", {"question": "递归还是迭代？"}, ctx)
    assert r.ok and r.kind == "need_input"
