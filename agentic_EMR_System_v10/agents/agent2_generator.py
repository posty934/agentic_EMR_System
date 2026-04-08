import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from core.llm_factory import create_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


class MedicalRecordDraft(BaseModel):
    chief_complaint: str = Field(
        description="主诉：高度精炼的患者核心症状及持续时间。例如：'腹泻伴恶心、呕吐2天'。"
    )
    history_of_present_illness: str = Field(
        description="现病史：将患者的所有症状细节，按照时间线顺序串联成专业、通顺的医学叙述文本。"
    )


class Agent2Generator:
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     api_key=os.getenv("LLM_API_KEY"),
        #     base_url=os.getenv("LLM_BASE_URL"),
        #     model=os.getenv("LLM_MODEL_NAME"),
        #     temperature=0.2,
        # )
        self.llm = create_llm(temperature=0.2)

        self.parser = JsonOutputParser(pydantic_object=MedicalRecordDraft)

        # === 初次生成草稿 ===
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的临床住院医师。
你的任务是根据智能问诊系统收集到的【患者结构化症状数据】，撰写一份专业的门诊病历草稿（包含主诉和现病史）。

【患者结构化症状数据】:
{patient_data}

【撰写要求】:
1. 主诉 (Chief Complaint)：要求极其简练，提取最主要的 1-3 个标准术语及最长发病时间，尽量控制在20字以内。
2. 现病史 (History of Present Illness)：
   - 必须且只能基于传入的【结构化数据】撰写，严禁脑补、编造数据中不存在的症状。
   - 将患者的“原话”替换为“标准医学术语 (standard_term)”。
   - 描述要客观、准确，符合医疗文书的书写规范。
   - 本系统是预问诊系统，不要生成诊断结论、检查结论、治疗建议或处方内容。
3. 严格遵循以下输出格式：
{format_instructions}"""),
            ("human", "请开始撰写病历。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

        # === 自动闭环修正用：根据质控反馈重写草稿 ===
        self.repair_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的临床住院医师，现在需要根据质控反馈，对病历草稿做“内部自动修正”。

【当前有效结构化症状数据】:
{patient_data}

【上一版病历草稿】:
{draft_data}

【质控反馈 / 修正指令】:
{repair_instruction}

【修正规则】:
1. 只能基于【当前有效结构化症状数据】修改，严禁新增不存在的信息。
2. 删除与“被撤销症状”有关的内容，删除与实体不一致、与时间不一致、与症状特征不一致的内容。
3. 若原草稿里有正确内容，应尽量保留。
4. 若某些信息在结构化数据中缺失，不能自行补充或猜测。
5. 不要生成诊断、检查结果、治疗方案或处方。
6. 输出仍然只包含：主诉 + 现病史。
7. 严格按以下 JSON 格式输出：
{format_instructions}"""),
            ("human", "请直接给出修正后的病历草稿。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.repair_chain = self.repair_prompt | self.llm | self.parser

    def _get_active_entities(self, current_entities: list) -> list:
        if not current_entities:
            return []
        return [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

    def generate_record(self, current_entities: list) -> dict:
        if not current_entities:
            return {
                "chief_complaint": "无明确主诉",
                "history_of_present_illness": "患者未提供有效病情信息。"
            }

        try:
            active_entities = self._get_active_entities(current_entities)

            if not active_entities:
                return {
                    "chief_complaint": "无不适",
                    "history_of_present_illness": "患者自述无任何不适症状，此前陈述均已否认。"
                }

            data_str = json.dumps(active_entities, ensure_ascii=False, indent=2)
            result = self.chain.invoke({"patient_data": data_str})
            return result

        except Exception as e:
            print(f"病历生成出错: {e}")
            return {
                "chief_complaint": "生成失败",
                "history_of_present_illness": "系统生成病历时发生错误，请重试。"
            }

    def revise_record(self, current_entities: list, draft_record: dict, repair_instruction: str) -> dict:
        """
        根据 Agent3 的质控反馈，利用现有实体自动重写草稿。
        """
        if not current_entities:
            return {
                "chief_complaint": "无明确主诉",
                "history_of_present_illness": "患者未提供有效病情信息。"
            }

        try:
            active_entities = self._get_active_entities(current_entities)

            if not active_entities:
                return {
                    "chief_complaint": "无不适",
                    "history_of_present_illness": "患者自述无任何不适症状，此前陈述均已否认。"
                }

            if not draft_record:
                return self.generate_record(current_entities)

            data_str = json.dumps(active_entities, ensure_ascii=False, indent=2)
            draft_str = json.dumps(draft_record, ensure_ascii=False, indent=2)
            instruction = repair_instruction.strip() if repair_instruction else "请严格基于现有实体重写病历草稿，删除不一致内容，保留正确内容。"

            result = self.repair_chain.invoke({
                "patient_data": data_str,
                "draft_data": draft_str,
                "repair_instruction": instruction
            })
            return result

        except Exception as e:
            print(f"病历自动修正出错: {e}")
            return draft_record if draft_record else {
                "chief_complaint": "生成失败",
                "history_of_present_illness": "系统修正病历时发生错误，请重试。"
            }