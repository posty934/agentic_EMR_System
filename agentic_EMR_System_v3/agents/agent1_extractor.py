import os
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from knowledge.retriever import MedicalRetriever

load_dotenv()


class SymptomDetail(BaseModel):
    symptom_name: str = Field(description="核心症状名称，保留患者原话")
    standard_term: Optional[str] = Field(default=None, description="标准医学术语映射结果")
    status: str = Field(default="active", description="状态：'active'为确认存在的症状，'revoked'为排除")
    onset_time: Optional[str] = Field(default=None)
    characteristics: Optional[str] = Field(default=None)
    inducement: Optional[str] = Field(default=None)
    frequency: Optional[str] = Field(default=None)
    alleviating_factors: Optional[str] = Field(default=None)
    dynamic_details: Optional[Dict[str, str]] = Field(default_factory=dict,
                                                      description="Planner动态追问收集到的其他细节")


class CoarseExtractionResult(BaseModel):
    symptoms: List[SymptomDetail] = Field(description="提取出的症状及其关联属性")


class Agent1Extractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL_NAME"),
            temperature=0.0,
        )

        self.retriever = MedicalRetriever()

        # ==========================================
        # 模块 A: Extractor (信息抽取员)
        # ==========================================
        self.extract_parser = JsonOutputParser(pydantic_object=CoarseExtractionResult)
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名专业的医疗信息结构化专家。
            任务：从患者最新表达中提取病情信息，并更新到对应的症状记录中。

            【核心提取规则 - 必须严格遵守】：
            1. 保持患者的原话表述。
            2. 🚨【主动补全机制】：如果患者一次性提供了一大段病情，你必须将所有这些细节全部提取出来！无法放入基础字段的，存入 `dynamic_details` 字典。
            3. 🚨【既往史对比专属提取】：如果 AI 在上一个问题中询问了“这次和上次相比（是否加重/有新症状等）”，无论患者回答什么（哪怕是“差不多”、“没啥区别”），你必须将其提取并存入 `dynamic_details` 字典中，Key 严格设置为 "与过往病史关联"。
            4. 【上下文锚定】：结合“近期对话上下文”判断患者回答的是什么字段，并精准填入。
            5. 如果患者推翻了某症状，将 status 设为 "revoked"。

            【当前已记录的症状库】:
            {current_entities}

            【近期对话上下文 (Short-term Memory)】:
            {chat_history_str}

            【患者既往病史档案 (Long-term Memory)】:
            {long_term_memory_str}

            {format_instructions}"""),
            ("human", "患者最新回复：{patient_input}")
        ]).partial(format_instructions=self.extract_parser.get_format_instructions())
        self.extract_chain = self.extract_prompt | self.llm | self.extract_parser

        # ==========================================
        # 模块 B: Planner (动态医疗规划者) - 🚨 终极死循环熔断机制
        # ==========================================
        with open("knowledge/clinical_guidelines.json", "r", encoding="utf-8") as f:
            self.clinical_guidelines = json.load(f)

        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名极其严谨且具备临床诊断思维的消化内科主治医师（Planner Agent）。

            【当前患者的症状与已收集信息】:
            {active_entities}

            【官方临床问诊指南 (针对当前症状)】:
            {retrieved_guidelines}

            【近期对话上下文 (Short-term Memory)】:
            {chat_history_str}

            【患者既往病史档案 (Long-term Memory)】:
            {long_term_memory_str}

            【规划与推理规则】:
            1. 🚨【现病史优先原则】：必须优先对照《指南》排查当前的症状要素。未收集完前，绝对不许提及既往史！
            2. 🚨【宽容与同义满足（治愈强迫症）】：如果指南要素是复合词（如“极度口渴/尿少”），患者只要明确否定了其中一个核心词（如“无口渴”），整条要素即视为排查完毕！绝对禁止再去抠字眼问剩下的词（比如追问尿少）！
            3. 🚨【阴性排查认同】：只要患者主动提及“没有XX”，视为该要素已排查完成。
            4. 🚨【既往史单次询问机制（核心）】：当且仅当《指南》要求的所有要素都已经排查完毕后，你必须执行以下检查：
               - 查阅【患者既往病史档案】。如果存在与当前症状相关的既往史。
               - 并且，【当前患者的症状】的 dynamic_details 中 **还未包含** "与过往病史关联" 这个字段。
               - 你只能输出一条【宽泛的追问指令】：“向患者说明他之前也有过类似记录，粗略询问这次和上次比是否加重？有无新发伴随症状？”。绝对不要追问细枝末节！
            5. 🚨【绝对结束红线】：如果《指南》已排查完毕，且（无相关既往史，或者 dynamic_details 中已经存在 "与过往病史关联" 字段），你必须且只能输出大写单词：COMPLETED。无论患者说什么，绝不许再生成任何问题！

            【执行步骤】：
            第一步：严格按上述红线规则评估。
            第二步：
            - 若触及【绝对结束红线】，只输出：COMPLETED。
            - 否则，只输出【一个】最紧急的追问指令。
            """),
            ("human", "请进行思考并给出下一步指令：")
        ])
        self.planner_chain = self.planner_prompt | self.llm | StrOutputParser()

        # ==========================================
        # 模块 C: Speaker (前台问诊员)
        # ==========================================
        self.reply_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名温和、专业的医院预问诊AI助手。
            你的任务是将后台主治医师下达的【追问指令】，转化为对患者的自然语言提问。

            【主治医师下达的追问指令】:
            {planner_task}

            【规则】：
            1. 使用关切的口吻，话术要简短。
            2. 不要向患者暴露“后台医师”、“指南”、“指令”等字眼。
            """),
            ("human", "患者刚刚说：{patient_input}\n请开始提问：")
        ])
        self.reply_chain = self.reply_prompt | self.llm | StrOutputParser()

    @staticmethod
    def _build_targeted_followup(error_report: dict) -> str:
        """根据 Reviewer 结构化错误，生成定向追问话术。"""
        slot_question_map = {
            "symptom_name": "为了确认病情，能再明确一下您最主要的不适症状是什么吗？",
            "onset_time": "请再确认一下，这个症状是从什么时候开始出现的？",
            "characteristics": "这个不适具体是什么感觉（比如胀痛、绞痛、隐痛）？",
            "inducement": "在发作前有没有明显诱因，比如饮食、受凉或劳累？",
            "frequency": "这个症状出现的频率大概是怎样的？",
            "alleviating_factors": "有没有什么方式能让症状缓解一些？",
            "dynamic_details.与过往病史关联": "结合您既往类似情况，这次和上次相比有加重吗？有没有新出现的伴随症状？"
        }

        suggested_slots = error_report.get("suggested_target_slots") or []
        for slot in suggested_slots:
            if slot in slot_question_map:
                return slot_question_map[slot]

        conflict_fields = error_report.get("conflict_fields") or []
        for field in conflict_fields:
            if field in slot_question_map:
                return slot_question_map[field]

        return "我想再核实一个关键细节：这次最困扰您的症状是什么？大概从什么时候开始？"

    def extract(self, patient_input: str, current_entities: List[dict] = None, last_ai_message: str = "",
                chat_history_str: str = "", long_term_memory_str: str = "") -> List[dict]:
        try:
            entities_str = json.dumps(current_entities, ensure_ascii=False) if current_entities else "暂无记录"
            result_dict = self.extract_chain.invoke({
                "patient_input": patient_input,
                "current_entities": entities_str,
                "chat_history_str": chat_history_str,
                "long_term_memory_str": long_term_memory_str
            })
            extracted_symptoms = result_dict.get("symptoms", [])

            for ent in extracted_symptoms:
                if ent.get("status") == "active" and ent.get("symptom_name"):
                    query = ent["symptom_name"]
                    if ent.get("characteristics"):
                        query += " " + ent["characteristics"]
                    standard_term = self.retriever.get_standard_term(query)
                    ent["standard_term"] = standard_term
            return extracted_symptoms
        except Exception as e:
            print(f"提取出错: {e}")
            return []

    def generate_reply(self, patient_input: str, current_entities: List[dict], last_ai_message: str = "",
                       chat_history_str: str = "", long_term_memory_str: str = "",
                       error_report: dict = None) -> str:
        try:
            active_entities = [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

            if error_report and error_report.get("conflict_type") not in (None, "", "none"):
                return self._build_targeted_followup(error_report)

            if not active_entities:
                return "您好，系统似乎没有捕捉到您的不适症状，能请您再具体描述一下哪里不舒服吗？"

            guidelines_for_current_symptoms = {}
            for ent in active_entities:
                term = ent.get("standard_term") or ent.get("symptom_name")
                if term in self.clinical_guidelines:
                    guidelines_for_current_symptoms[term] = self.clinical_guidelines[term]

            if not guidelines_for_current_symptoms:
                guidelines_for_current_symptoms = {"通用问诊法则": "请依次核实：发病时间、具体部位、性质、频率、诱因"}

            entities_str = json.dumps(active_entities, ensure_ascii=False, indent=2)
            guidelines_str = json.dumps(guidelines_for_current_symptoms, ensure_ascii=False, indent=2)

            planner_task_raw = self.planner_chain.invoke({
                "active_entities": entities_str,
                "retrieved_guidelines": guidelines_str,
                "chat_history_str": chat_history_str,
                "long_term_memory_str": long_term_memory_str
            })

            print(f"\n[Planner 思考与决策过程] ->\n{planner_task_raw}\n")

            planner_task_clean = re.sub(r'<thinking>.*?</thinking>', '', planner_task_raw, flags=re.DOTALL).strip()

            if not planner_task_clean and "COMPLETED" in planner_task_raw:
                planner_task_clean = "COMPLETED"

            if "COMPLETED" in planner_task_clean:
                clean_input = patient_input.strip().replace("，", "").replace("。", "").replace("！", "")
                exact_finish_words = ["没有", "没", "无", "好了", "差不多", "就这些", "没别的", "没有了", "没了",
                                      "不需要了"]

                is_ending = False
                if clean_input in exact_finish_words:
                    is_ending = True
                elif "其他" in last_ai_message and any(kw in patient_input for kw in ["没有", "没", "无", "没别的"]):
                    is_ending = True

                if is_ending:
                    return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"
                else:
                    return "除了这些，您还有其他不舒服的地方吗？"

            reply = self.reply_chain.invoke({
                "patient_input": patient_input,
                "planner_task": planner_task_clean
            })
            return reply

        except Exception as e:
            print(f"生成回复出错: {e}")
            return "不好意思，系统开了个小差，您能再说一遍吗？"
