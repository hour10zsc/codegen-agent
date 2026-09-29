"""自定义异常：区分失败来源，便于重试逻辑与界面层分别处理。"""


class CodegenError(Exception):
    """项目基础异常。"""


class LLMError(CodegenError):
    """LLM 调用失败（网络/服务端错误等）。"""


class RetryExhausted(LLMError):
    """重试次数耗尽仍失败（LLM 客户端抛出）。"""


class ToolError(CodegenError):
    """工具执行失败（参数非法、路径越界、执行超时等）。"""


class NeedInput(CodegenError):
    """Agent 需要用户补充输入（ask_user 工具触发）。

    question: 向用户提出的问题
    messages: 挂起时的完整消息序列（含 system/few-shot/历史），
              恢复时由界面层原样续接，不丢失循环状态。
    """

    def __init__(self, question: str, messages: list[dict]):
        super().__init__(question)
        self.question = question
        self.messages = messages
