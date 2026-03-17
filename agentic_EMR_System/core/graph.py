from typing import TypedDict, List, Dict, Any


class EMRState(TypedDict):
    """
    定义贯穿整个多智能体系统的全局状态图 (State Graph)
    """
    # === 1. 基础对话与输入信息 ===
    current_input: str  # 患者当前轮次的输入文本
    dialog_history: List[Dict]  # 多轮对话历史 (格式如 [{"role": "user", "content": "..."}, ...])

    # === 2. Agent 1 (信息抽取与交互) 的输出 ===
    extracted_entities: List[str]  # 粗提取出的原声实体 (如"嗓子冒烟")
    standardized_terms: List[str]  # 经检索重排后的标准医学术语 (如"咽干")
    missing_info: List[str]  # 系统识别出缺失的关键病情要素（用于触发主动追问）

    # === 3. Agent 2 (病历生成) 的输出 ===
    draft_record: str  # 根据模板转换而成的结构化病历草稿

    # === 4. Agent 3 (逻辑校验) 的输出与反馈 ===
    is_valid: bool  # 逻辑校验是否通过
    error_feedback: str  # 校验失败时生成的错误报告，用于注入Agent 1的短期记忆进行回滚修正

    # === 5. 流程控制 ===
    next_action: str  # 记录下一步的操作走向 (例如: "generate", "ask_user", "end")