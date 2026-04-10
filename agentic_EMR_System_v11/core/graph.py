from typing import TypedDict, List, Dict, Annotated
import operator
from langchain_core.messages import BaseMessage


class EMRState(TypedDict):
    """
    定义贯穿整个多智能体系统的全局状态图 (State Graph)
    """

    # === 1. 对话历史 ===
    messages: Annotated[List[BaseMessage], operator.add]

    # === 2. 问诊流程控制 ===
    is_finished: bool

    # === 3. Agent 1 (信息抽取) 的输出 ===
    entities: List[Dict]
    missing_slots: List[str]

    # === 4. Agent 2 (病历生成) 的输出 ===
    draft_record: Dict

    # === 5. Agent 3 (质控校验) 的输出与反馈 ===
    is_valid: bool
    feedback: str
    rollback_question: str
    repair_instruction: str

    # === 6. 自动闭环修正控制字段 ===
    auto_revision_possible: bool
    need_user_input: bool
    revision_count: int
    max_revision_count: int

    pending_question_target: Dict


