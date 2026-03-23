import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


# === 1. 定义标准病历输出的结构 ===
class MedicalRecordDraft(BaseModel):
    chief_complaint: str = Field(
        description="主诉：高度精炼的患者核心症状及持续时间。例如：'腹泻伴恶心、呕吐2天'。"
    )
    history_of_present_illness: str = Field(
        description="现病史：将患者的所有症状细节，按照时间线顺序串联成专业、通顺的医学叙述文本。"
    )


class Agent2Generator:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL_NAME"),
            temperature=0.2,
        )

        self.parser = JsonOutputParser(pydantic_object=MedicalRecordDraft)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的临床住院医师。
你的任务是根据智能问诊系统收集到的【患者结构化症状数据】，撰写一份专业的门诊病历草稿（包含主诉和现病史）。

【患者结构化症状数据】:
{patient_data}

【撰写要求】:
1. 主诉 (Chief Complaint)：要求极其简练，提取最主要的 1-3 个标准术语及最长发病时间，尽量控制在20字以内。
2. 现病史 (History of Present Illness)：
   - 🚨 必须且只能基于传入的【结构化数据】撰写！严禁脑补、编造数据中不存在的症状！
   - 将患者的“原话”替换为“标准医学术语 (standard_term)”。
   - 描述要客观、准确，符合医疗文书的书写规范。
3. 严格遵循以下输出格式：\n{format_instructions}"""),
            ("human", "请开始撰写病历。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def generate_record(self, current_entities: list) -> dict:
        """接收结构化数据，生成病历字典"""
        if not current_entities:
            return {"chief_complaint": "无明确主诉", "history_of_present_illness": "患者未提供有效病情信息。"}

        try:
            # === 🚀 核心修复：Agent 2 也要物理隔离被撤销的症状！ ===
            active_entities = [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

            # 如果全都被患者否认了
            if not active_entities:
                return {"chief_complaint": "无不适",
                        "history_of_present_illness": "患者自述无任何不适症状，此前陈述均已否认。"}

            # 只把真实的 active_entities 发给大模型去写病历
            data_str = json.dumps(active_entities, ensure_ascii=False, indent=2)
            result = self.chain.invoke({"patient_data": data_str})
            return result
        except Exception as e:
            print(f"病历生成出错: {e}")
            return {"chief_complaint": "生成失败", "history_of_present_illness": "系统生成病历时发生错误，请重试。"}