# 代码生成 Agent（codegen-agent）

> 软件工程课程 Homework 1：根据自然语言描述生成可运行代码的 Agent。
> 手写 Agent 循环（输入 → 推理 → 工具调用 → 输出），支持文件读写、代码执行验证、
> 上下文记忆、错误重试，提供命令行与 Web 双界面。

## 特性

- **完整 Agent 循环**：LLM 推理 → 工具调用 → 结果回填 → 继续推理，直至产出最终答案
- **5 个工具**：`write_file` / `read_file` / `list_files` / `run_code` / `ask_user`
- **生成-验证闭环**：代码先落盘，再在受限沙箱中执行验证，报错自动修复（最多 3 次）
- **上下文记忆**：滑动窗口 + 会话 jsonl 持久化，重启后继续对话
- **错误处理与重试**：指数退避、Retry-After、401 快速失败、工具错误回传自修复
- **双界面共用核心**：CLI（`python -m codegen`）与 Streamlit Web（`streamlit run webapp/app.py`）
- **46 个单元测试**：全部离线可跑，覆盖循环、记忆、重试、工具、路径安全

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+，Windows/Linux/macOS）
pip install -r requirements.txt

# 2. 配置 API Key
#    复制 .env.example 为 .env，填入你的 DeepSeek API Key
#    （在 platform.deepseek.com 申请）

# 3. 命令行：单命令模式
python -m codegen "用 Python 写一个快速排序函数，并用 run_code 验证"

# 4. 命令行：交互模式
python -m codegen

# 5. Web 界面
streamlit run webapp/app.py
```

## 演示视频

[demo.mp4](demo.mp4)（2 分钟完整演示：代码生成与验证、错误自修复、上下文记忆、Web 界面）

## 使用指南

### 命令行

| 模式 | 命令 | 说明 |
|------|------|------|
| 交互 REPL | `python -m codegen` | 逐轮对话，打印工具调用过程 |
| 单命令 | `python -m codegen "任务描述"` | 执行一轮后退出 |

常用参数：`--workspace DIR`（工作目录）、`--model NAME`（覆盖模型）、
`--session NAME`（启动加载会话）、`--no-tools`（纯聊天）、`--quiet`（隐藏工具过程）。

交互模式下的斜杠命令：

| 命令 | 行为 |
|------|------|
| `/tools` | 列出全部工具 |
| `/save [名称]` | 保存会话（默认按时间戳命名） |
| `/load <名称>` | 加载会话 |
| `/clear` | 清空上下文记忆 |
| `/history [n]` | 查看最近 n 条历史 |
| `/quit` `/exit` | 退出 |

### Web 界面

- 主区聊天式交互；侧栏可保存/加载/清空会话
- 顶部三个示例按钮一键演示（快速排序 / 斐波那契 / 统计代码行数）
- 每次回复可展开「🔧 工具调用过程」，查看每步工具的参数与返回

### 示例对话

```
你 > 用 Python 写一个快速排序，并验证
  [tool] write_file(path=qsort.py, content=def qsort(a): …)
         → 已写入 qsort.py：245 字节，11 行
  [tool] run_code(language=python, code=from qsort import qsort …)
         → 退出码 0
Agent > 已生成 qsort.py 并通过验证：排序结果正确。
```

## 配置项（.env）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 模型名。`deepseek-chat` 已于 2026-07 停用；可选 `deepseek-v4-pro` |
| `DEEPSEEK_THINKING` | `disabled` | 思考模式开关（`enabled` 自动兼容 reasoning_content） |
| `DEEPSEEK_MAX_TOKENS` | `8192` | 单次回复最大 token |
| `AGENT_MAX_ITERATIONS` | `8` | 单轮最大迭代次数 |
| `AGENT_WORKSPACE` | `workspace` | 文件工具工作目录 |
| `AGENT_SESSIONS_DIR` | `sessions` | 会话保存目录 |
| `RUN_TIMEOUT` | `15` | 代码执行超时（秒，上限 60） |
| `MEMORY_MAX_MESSAGES` | `20` | 记忆滑动窗口（消息条数） |
| `MEMORY_MAX_TOKENS` | `64000` | 记忆 token 预算（估算） |

## 项目结构

```
codegen-agent/
├── codegen/                 # 核心包：零 Streamlit 依赖，CLI/Web 共用
│   ├── agent.py             # Agent 核心循环
│   ├── llm.py               # DeepSeek 客户端（重试/退避/兼容）
│   ├── memory.py            # 滑动窗口记忆 + jsonl 持久化
│   ├── prompts.py           # 系统提示词 + few-shot 示例
│   ├── config.py            # .env 配置加载
│   ├── sandbox.py           # 受限代码执行沙箱
│   ├── cli.py               # 命令行界面
│   └── tools/               # 工具注册表与 5 个工具实现
├── webapp/app.py            # Streamlit Web 界面
├── tests/                   # pytest 测试（46 个，离线可跑）
├── Design.md                # 架构设计文档
└── README.md
```

## 测试

```bash
pytest tests/ -q        # 全部离线，不消耗 API 额度
```

覆盖：Agent 循环（FakeLLM 脚本化）、滑动窗口配对不变量、重试退避序列、
路径穿越拒绝、沙箱超时/截断、斜杠命令解析、few-shot 格式校验。

## 架构概览

详见 **[Design.md](Design.md)**，要点：

- 手写 ReAct 式工具调用循环（不用框架，架构透明、便于讲解）
- 工具注册表 + 统一 ToolResult：新增工具循环零改动
- AgentHooks 回调协议：CLI 打印与 Web 渲染共用同一份工具过程
- 「错误即信息」：工具失败不崩溃，错误文本回传模型自我修正

## 安全与免责声明

- `run_code` 是**作业级受限执行，不是安全沙箱**：仅提供独立目录、超时、输出截断、
  环境变量白名单与进程树清理；不应执行不可信输入，生产级隔离需 Docker/虚拟机。
- API Key 只存本地 `.env`（已被 .gitignore 忽略），切勿提交到仓库。
- 调用 DeepSeek API 会产生少量费用（`deepseek-v4-flash` 约 $0.14/百万输入 token）。
