# Design.md —— 代码生成 Agent 设计文档

## 1. 设计目标与需求映射

本作业要求「搭建一个简单的代码助手 Agent，掌握 LLM 调用、Prompt 设计、工具集成等
基础能力」，选择方向为**代码生成**（根据自然语言描述生成代码片段）。

下表逐条对照作业要求与本设计决策：

| 作业要求 | 本设计决策 | 落点 |
|----------|-----------|------|
| 基本 Agent 循环：输入 → 推理 → 工具调用 → 输出 | 手写 ReAct 式工具调用循环 | `codegen/agent.py`（§3） |
| 支持至少一种工具 | 5 个工具：文件读写、目录列举、代码执行、用户提问 | `codegen/tools/`（§4） |
| 命令行或简单 Web 界面 | **双界面**：CLI + Streamlit，共用同一核心 | `codegen/cli.py`、`webapp/app.py`（§8） |
| 支持上下文记忆 | 滑动窗口 + 配对不变量 + jsonl 持久化 | `codegen/memory.py`（§6） |
| 错误处理与重试机制 | 三层：LLM 调用指数退避重试；工具错误回传自修复；沙箱超时清理 | `codegen/llm.py`、`agent.py`、`sandbox.py`（§7） |
| LLM 原生 API（推荐路线之一） | openai SDK 对接 DeepSeek（OpenAI 兼容） | `codegen/llm.py` |
| Prompt 设计 / Few-shot | 中文系统提示词 + 结构化工具调用 few-shot 闭环 | `codegen/prompts.py`（§5） |

技术选型：**Python 3.13 + DeepSeek V4 系列模型 + openai SDK**，不引入 Agent 框架，
Agent 循环完全自研——架构透明、便于讲解，最能体现对 Agent 设计模式的掌握。

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        表现层（双界面）                        │
│   codegen/cli.py（REPL/单命令）      webapp/app.py（Streamlit）│
│        │  AgentHooks 回调（工具过程可视化）         │          │
├────────┴─────────────────────────────────────────┴──────────┤
│                       Agent 核心层                            │
│   codegen/agent.py：run_agent / resume_agent / _run_loop     │
│        组装消息 → 调用 LLM → 分发工具 → 回填结果 → 记忆提交    │
├───────┬──────────────────────────────┬───────────────────────┤
│ 记忆层 │          LLM 接入层           │       工具层          │
│memory │ codegen/llm.py（重试/兼容）    │ codegen/tools/        │
│滑动窗口│ DeepSeek（OpenAI 兼容 API）    │ 注册表 + 5 工具 + 沙箱 │
└───────┴──────────────────────────────┴───────────────────────┘
```

分层铁律：**Agent 层不 import 任何界面层**；CLI 与 Web 都只调用
`run_agent()/resume_agent()`。核心包零 Streamlit 依赖，`webapp/` 是唯一
import streamlit 的模块——这是「双界面共用核心」的结构证明。

新增工具的扩展路径：在 `tools/__init__.py` 注册一条 `ToolSpec` 即可，
Agent 循环零改动（开放-封闭原则）。

## 3. Agent 循环设计

### 3.1 消息流

```
messages = [system, few-shot 示例, ...记忆历史, 本轮 user 输入]
```

循环内追加两类消息：assistant 消息（**完整对象原样保存**）与 tool 消息（按
`tool_call_id` 配对）。完整对象原样回传是 DeepSeek V4 的硬性要求——思考模式
下手工重建 `{role, content}` 会丢掉 `reasoning_content` 字段导致 400。

### 3.2 循环流程

```
┌─────────┐   ┌──────────────┐   ┌──────────────────┐
│ 用户输入 │ → │ 组装消息      │ → │ LLM 调用(带重试)  │←─────────────┐
└─────────┘   └──────────────┘   └────────┬─────────┘              │
                                          │ 有 tool_calls?          │
                        ┌─────────────────┴────────────┐           │
                        ▼ 是                          ▼ 否         │
                ┌───────────────┐              ┌─────────────┐      │
                │ 逐个执行工具    │              │ 返回最终文本  │      │
                │ (参数解析+沙箱) │              └─────────────┘      │
                └───────┬───────┘                                    │
                        ▼                                            │
                ┌───────────────┐      ┌──────────────┐              │
                │ 结果回填为      │─────→│ 迭代次数<上限? │──────────────┘
                │ tool 角色消息   │      └──────────────┘ 是
                └───────────────┘             │否 → 返回超限提示
```

### 3.3 伪代码（与 agent.py 一一对应）

```
run_agent(user_input, memory, llm, settings, hooks, tools):
    schemas = tool_schemas() if tools is None else tools   # tools=[] 纯聊天
    messages = [system] + few_shot() + memory.messages + [user(user_input)]

    for i in 1..max_iterations:                 # 默认 8，防死循环
        resp = llm.chat(messages, schemas)      # 内部带重试（§7）
        messages.append(resp.message)           # 完整对象原样保存

        if resp.message 无 tool_calls:          # 终态
            memory.commit(messages[前缀之后:])   # 写入滑动窗口记忆
            return 最终文本（截断则附加提示）

        for tc in resp.message.tool_calls:      # 顺序执行（写→跑有依赖）
            args, err = safe_parse_args(tc.function.arguments)
            if err:  result = ToolResult.failure(err)          # JSON 坏了不崩溃
            else:    result = execute_tool(tc.function.name, args)
                     # 未知工具 / ToolError 也转成 failure 结果
            if result.kind == need_input:       # ask_user 触发人机协作
                return NeedInput(result.question, messages)    # 挂起
            messages.append(tool(tool_call_id=tc.id, content=result.as_string()))
            # 失败也回传（带 Error: 前缀），让模型读错误自我修正

    return "已达到最大迭代次数"                    # 双保险之二
```

### 3.4 关键设计决策

| 决策点 | 方案 | 理由 |
|--------|------|------|
| 循环终止 | 无 tool_calls + 迭代上限 | 双保险防死循环（模型可能反复调用工具） |
| 多工具调用 | 同响应内顺序逐个执行 | 工具间常有依赖（write_file → run_code），顺序语义简单可预测；代码留出并行扩展点 |
| 工具失败 | 不抛异常终止，错误文本回传模型 | 「错误即信息」——模型读过错误后自我修正，这是错误处理评分点的核心体现 |
| 参数解析 | JSON 解析失败回传错误文本 | DeepSeek 偶发输出非严格 JSON，循环永不因模型输出格式问题崩溃 |
| 消息回传 | 完整 assistant 对象原样 append | 保留 tool_calls / reasoning_content（DeepSeek V4 约束） |
| tool_choice | 从不发送 | V4 思考模式下连 `"auto"` 都返回 400（官方已知问题） |
| 思考模式 | 默认 `thinking=disabled` | 稳定、快、便宜；enabled 已做兼容（§7.3） |
| 流式输出 | v1 不做 | 流式 tool_calls 增量解析风险高、评分不要求；工具过程用回调可视化 |

### 3.5 人机协作（ask_user）

模型调用 `ask_user` 时循环**暂停**并返回 `NeedInput(question, messages)`，把完整
消息序列交给界面层；用户答复后 `resume_agent()` 原样续接 messages 继续循环，
零上下文丢失。CLI 用 `input()` 提问，Web 把问题渲染为消息、以下一次提交作为答复。

## 4. 工具系统

统一抽象（`tools/base.py`）：

- `ToolSpec(name, description, parameters, fn)` —— 注册即生成 OpenAI tools schema
- `ToolResult(ok, content, kind)` —— `as_string()` 失败时输出 `Error: ` 前缀，
  让模型能识别错误并自我修正；`kind="need_input"` 表示需要用户输入
- `ToolContext(workspace, run_timeout)` —— 运行时约束（工作目录、超时）

### 4.1 工具清单

| 工具 | 功能 | 关键错误语义（回传模型） |
|------|------|--------------------------|
| `write_file(path, content)` | 代码落盘，自动建子目录 | 路径越界拒绝；IO 错误 |
| `read_file(path, offset, limit)` | 带行号读取，可分段 | 不存在/目录/二进制/超 1MB 拒绝 |
| `list_files(path?)` | 文件树，深度 ≤3 | 目录不存在 |
| `run_code(language, code, filename?)` | 沙箱执行验证 | 语言不支持；解释器缺失；超时终止 |
| `ask_user(question)` | 需求不明确时提问 | 参数缺失 |

### 4.2 路径安全（write_file / read_file / list_files）

所有 path 参数经 `_resolve_in_workspace()`：拼接 workspace 根目录后 `resolve()`，
再用 `is_relative_to()` 校验——`../` 逃逸与绝对路径一律拒绝，返回错误给模型。

### 4.3 代码执行沙箱（sandbox.py）

| 隔离手段 | 实现 |
|----------|------|
| 目录隔离 | `workspace/.run/<uuid>/` 独立临时目录，执行后清理 |
| 超时控制 | `Popen.communicate(timeout)`，上限 60s |
| 进程树清理 | Windows 超时后 `taskkill /F /T /PID`（subprocess 超时只杀直接子进程） |
| 输出限制 | stdout/stderr 合并后截断 4000 字符 |
| 环境白名单 | 只透传 SYSTEMROOT/PATH/TEMP 等，**不含 DEEPSEEK_API_KEY** |
| 模块可见性 | workspace 加入 PYTHONPATH/NODE_PATH，write_file 生成的文件可直接 import |
| 无 shell | cmd 传 list 不用 shell=True（中文路径/空格安全） |
| 解释器预检 | python 用 sys.executable，node 用 shutil.which |

**安全边界声明**：这是作业级受限执行，**不是安全沙箱**——同一用户权限、同主机
文件系统（读限制靠白名单环境变量与约定，不构成强隔离）。不应执行不可信输入；
生产级隔离需 Docker/虚拟机/独立用户。知道边界在哪里，本身就是设计的一部分。

## 5. Prompt 设计

### 5.1 系统提示词要点（prompts.py 全文见代码）

1. **角色定位**：代码生成助手，工作环境 Python 3.13 / Node 24，文件根目录 workspace
2. **工具使用规范**：生成代码必须先 `write_file` 落盘再 `run_code` 验证；报错先
   `read_file` 定位修复，同一问题最多重试 3 次；不编造工具；参数必须合法 JSON
3. **代码输出规范**：聊天里不贴大段代码（文件里已有），回复简短；仅当用户明确
   要求「直接贴代码」才用 markdown 代码块
4. **默认约定**：默认 Python、中文回复、相对路径

要点 3 是本 Agent 与「普通聊天模型」的关键区别：普通模型会把 100 行代码贴进
聊天，我们把它约束为「落盘 + 验证 + 简报」，这正是 Agent 使用工具的体现。

### 5.2 Few-shot：结构化调用示范

在 system 后插入 **1 个完整闭环示例**（user 请求 → assistant 带合法 tool_calls 的
write_file → tool 结果 → assistant 的 run_code → tool 结果 → assistant 简短回复）。
关键设计：示例中的 assistant 消息**手工构造了合法的 tool_calls 结构**
（含 tool_call_id、function name/arguments JSON），教模型输出「结构化工具调用」
而非口述调用——这是 DeepSeek 系模型已知的弱点。只放 1 个示例：few-shot 对
DeepSeek 主要起格式示范作用，多示例消耗 token 且边际收益低。

格式不变量（有可执行测试守护）：凡含 tool_calls 的 assistant 消息，其后必有
tool_call_id 配对的 tool 消息——few-shot 与运行时消息都满足（tests/test_agent_loop.py）。

### 5.3 参数解析兜底

模型输出 `arguments` 非 JSON 时，回传「参数不是合法 JSON：…，请重新生成工具
调用」，让模型自己修正——而不是让程序崩溃。

## 6. 记忆系统

### 6.1 滑动窗口（memory.py）

- 消息条数上限 `MEMORY_MAX_MESSAGES`（默认 20）+ token 估算上限（默认 64K，
  中文 1 字 ≈ 1 token、其余 4 字符 ≈ 1 token），超限逐轮缩窗
- **配对不变量**：含 tool_calls 的 assistant 消息与其 tool 消息必须成对保留/丢弃。
  从头部裁剪时若切在「assistant(tool_calls) → tool」中间，会产生孤儿 tool_call，
  下次请求 API 直接 400。`find_cut_point()` 从窗口尾部起找合法切割点（O(n²)，
  消息量小，正确性优先）
- 永不丢 system（不进记忆）与最近一条 user（保证模型有可回答的问题）

### 6.2 持久化

`sessions/<名称>.jsonl`，每行一条完整消息 JSON（含 tool_calls / tool_call_id，
可无损恢复工具往返）；临时文件 + `os.replace` 原子写；加载跳过坏行并告警。

### 6.3 摘要扩展点（v1 未实现）

v1 只做滑动窗口。理由：自动摘要需要额外 LLM 调用且触发时机难定，滑动窗口对
作业场景（20 条 ≈ 一轮多轮工具对话）已足够且行为可预期。已预留 `/summary`
命令位（CLI）与 Design 扩展点：把历史压缩为一段摘要放回 system。

## 7. 错误处理与重试

三层防线：

### 7.1 LLM 调用重试（llm.py）

| 错误类型 | 策略 | 理由 |
|----------|------|------|
| 429 | 读 `Retry-After` 头等待；否则指数退避 | 服务端明确给出等待时间 |
| 5xx / 超时 / 连接错误 | 指数退避重试，最多 5 次 | 瞬时故障可自愈 |
| 401 / 403 | 不重试，立即失败 | Key 无效，重试无意义 |
| 400 且含 reasoning_content | 剥离该字段重试一次（特判） | V4 思考模式已知 400，剥离后合法 |

退避公式：`delay = min(1s × 2^attempt, 30s) + U(0, 0.5s)`（封顶 + 抖动，避免
惊群）。耗尽抛 `RetryExhausted`，由界面层转成用户可读错误。`should_retry()`
与 `backoff_delay()` 为纯函数，独立单测覆盖（tests/test_retry.py）。

### 7.2 工具错误回传自修复（agent.py）

工具执行失败**不终止循环**，把 `Error: <原因>` 文本作为 tool 消息回传。模型
读取错误后修正参数/路径/代码再试——这是 Agent 区别于普通程序的关键能力
（自我修复闭环），system prompt 约束同一问题最多重试 3 次。

### 7.3 沙箱错误（sandbox.py）

超时杀进程树、输出截断、解释器缺失预检（友好报错）、运行目录用完即清理。

## 8. 双界面设计

CLI 与 Web 共用 `run_agent()/resume_agent()`，界面差异通过 **AgentHooks 回调
协议**注入：

```python
AgentHooks(on_tool_start(name, args), on_tool_end(name, result))
```

- CLI：打印 `[tool] write_file(path=qsort.py, …)` 与结果截断（演示素材）
- Web：收集工具日志，渲染为可展开的「🔧 工具调用过程」（参数 JSON + 返回）

Streamlit 特有问题的规避：
- LLM 客户端用 `st.cache_resource` 缓存（openai 客户端不可 pickle，不能放
  session_state，否则每次 rerun 重建连接）
- session_state 只存可序列化数据（记忆消息、显示消息、ask_user 挂起序列）
- ask_user：问题渲染为 assistant 消息，下一次用户提交作为答复接续循环

## 9. 已知限制与未来工作

1. **非流式输出**：v1 整段返回；流式 + 增量 tool_calls 解析是明确的下一步
2. **沙箱非强隔离**：生产级需 Docker/虚拟机（§4.3 声明）
3. **无自动记忆摘要**：长会话靠滑动窗口；未来按 token 阈值触发 LLM 摘要
4. **代码库理解靠文件工具**：大仓库场景可加 RAG/嵌入检索
5. **工具并行**：多 tool_calls 目前顺序执行，可扩展为无依赖并行

## 附：关键文件索引

| 文件 | 职责 |
|------|------|
| `codegen/agent.py` | Agent 循环（§3） |
| `codegen/llm.py` | DeepSeek 客户端与重试（§7.1） |
| `codegen/memory.py` | 滑动窗口记忆（§6） |
| `codegen/prompts.py` | 系统提示词与 few-shot（§5） |
| `codegen/tools/` | 工具注册表与实现（§4） |
| `codegen/sandbox.py` | 受限执行沙箱（§4.3） |
| `codegen/cli.py` / `webapp/app.py` | 双界面（§8） |
| `tests/` | 46 个离线单元测试 |
