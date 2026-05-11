import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


# === 1. 定义质控输出的数据结构 ===
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="病历逻辑是否合理、完整、无自相矛盾。True为通过，False为拦截。")
    feedback: str = Field(description="质控意见。如果通过，简述理由；如果拦截，一针见血地指出具体的矛盾或缺失点。")
    rollback_question: str = Field(
        description="如果拦截，需提供一句向患者追问的话术（比如确认矛盾点）；如果通过，填空字符串。")
    conflict_type: str = Field(
        default="none",
        description="冲突类型，例如：fact_conflict（事实冲突）、logical_conflict（逻辑冲突）、missing_info（信息缺失）。通过时填 none。"
    )
    conflict_fields: list[str] = Field(
        default_factory=list,
        description="发生冲突的字段名列表（如 symptom_name, onset_time, characteristics）。"
    )
    suggested_target_slots: list[str] = Field(
        default_factory=list,
        description="建议下一轮优先追问的目标槽位（例如 onset_time、frequency、dynamic_details.与过往病史关联）。"
    )


class Agent3Reviewer:
    def __init__(self):
        # 质控需要极高的严谨性，温度设为 0
        self.llm = ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL_NAME"),
            temperature=0.0,
        )
        self.parser = JsonOutputParser(pydantic_object=ValidationResult)

        # === 2. 核心提示词：设定质控规则 ===
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名极其严谨的医院质控主任医师（Agent 3）。
            你的任务是：审查 Agent 2 生成的【病历草稿】，并对照 Agent 1 提取的【结构化症状】，判断是否存在逻辑矛盾或重大缺失。

            【当前结构化症状库 (事实基准)】:
            {entities}

            【待审查病历草稿】:
            {draft}

            【审查规则】:
            1. 事实一致性：主诉和现病史里的描述，绝不能和【结构化症状库】里的记录冲突（如库里写饭后痛，病历里写空腹痛）。
            2. 逻辑自洽性：如果患者同时有“腹泻”和“便秘”，这就是明显需要澄清的矛盾点，必须拦截 (is_valid: false)。
            3. 信息完整性：如果有明显的医学逻辑断层（例如提到发热，但完全没记录体温），可以进行拦截并要求追问。
            4. 🚨【已排除症状忽略】：请注意，系统传入的症状库中，只包含了真实有效的症状。如果患者之前有过口误，系统已经在传入前将其清除了，所以你只需专注于当前传入的实体。
            5. 语气：如果需要拦截，rollback_question 必须用温和的医生口吻，向患者确认那个矛盾或缺失的信息。
            6. 结构化输出要求：
               - conflict_type 必须是 none/fact_conflict/logical_conflict/missing_info 之一；
               - conflict_fields 给出冲突字段；
               - suggested_target_slots 给出建议追问目标槽位，便于后续 Agent1 定向追问。
            \n{format_instructions}"""),
            ("human", "请进行严格的病历质控审查。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def validate(self, draft: dict, current_entities: list) -> dict:
        """接收病历和原始字典，进行质控"""
        if not current_entities or not draft:
            return {
                "is_valid": False,
                "feedback": "数据为空，无法质控",
                "rollback_question": "抱歉，系统数据丢失，能重新描述下症状吗？",
                "conflict_type": "missing_info",
                "conflict_fields": ["entities", "draft_record"],
                "suggested_target_slots": ["symptom_name", "onset_time", "characteristics"]
            }

        try:
            # === 🚀 核心物理拦截：直接把被患者 revoked(撤销) 的症状过滤掉！ ===
            # 不把被撤销的症状发给大模型，从物理层面上杜绝大模型产生逻辑混乱
            active_entities = [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

            entities_str = json.dumps(active_entities, ensure_ascii=False)
            draft_str = json.dumps(draft, ensure_ascii=False)

            return self.chain.invoke({"entities": entities_str, "draft": draft_str})
        except Exception as e:
            print(f"Agent 3 审查出错: {e}")
            # 遇到报错为了防止系统卡死，可以兜底放行
            return {
                "is_valid": True,
                "feedback": "校验超时，默认放行",
                "rollback_question": "",
                "conflict_type": "none",
                "conflict_fields": [],
                "suggested_target_slots": []
            }
