from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage


class EMRState(TypedDict):
    """
    定义贯穿整个多智能体系统的全局状态图 (State Graph)
    严格对齐开题报告中的多智能体交互逻辑
    """
    # === 1. 对话历史 (核心记忆) ===
    # 使用 Annotated 和 operator.add，确保每次 append 消息时是追加而不是覆盖
    messages: Annotated[List[BaseMessage], operator.add]

    is_finished: bool  # 新增：严格标识问诊流程是否彻底结束

    # === 2. Agent 1 (信息抽取) 的输出 ===
    entities: List[Dict]  # 当前已提取并映射的标准症状库（消化科专属）
    missing_slots: List[str]  # 尚未收集全的必填项（如：发作时间、诱因）

    # === 3. Agent 2 (病历生成) 的输出 ===
    draft_record: Dict  # 结构化的病历草稿 {"chief_complaint": "...", "hpi": "..."}

    # === 4. Agent 3 (质控校验) 的输出与反馈 ===
    is_valid: bool  # 病历逻辑是否通过质控
    feedback: str  # 质控主任给出的内部修改意见（用于注入 Agent1 记忆进行回滚修正）
    rollback_question: str  # 质控主任建议向患者追问的话术（比如确认腹泻和便秘的时间先后）
    error_report: Dict[str, Any]  # 结构化质控错误（冲突类型、冲突字段、建议追问槽位）
    revision_round: int  # 质控回滚轮次计数
    last_valid_snapshot_id: str  # 最近一次通过质控的快照标识


    long_term_memory: str     # 存放从向量库中检索出的、与当前对话相关的长期记忆
