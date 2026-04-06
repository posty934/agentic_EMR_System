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


# class SymptomDetail(BaseModel):
#     symptom_name: str = Field(description="核心症状名称，保留患者原话")
#     standard_term: Optional[str] = Field(default=None, description="标准医学术语映射结果")
#     status: str = Field(default="active", description="状态：'active'为确认存在的症状，'revoked'为排除")
#     onset_time: Optional[str] = Field(default=None)
#     characteristics: Optional[str] = Field(default=None)
#     inducement: Optional[str] = Field(default=None)
#     frequency: Optional[str] = Field(default=None)
#     alleviating_factors: Optional[str] = Field(default=None)
#     dynamic_details: Optional[Dict[str, str]] = Field(
#         default_factory=dict,
#         description="Planner动态追问收集到的其他细节"
#     )
class SymptomDetail(BaseModel):
    symptom_name: str = Field(description="核心症状名称，保留患者原话")
    standard_term: Optional[str] = Field(default=None, description="标准医学术语映射结果")
    status: str = Field(default="active", description="状态：active 为确认存在的症状，revoked 为患者否认或撤销")

    # 既有通用字段
    onset_time: Optional[str] = Field(default=None, description="起病或发现时间")
    characteristics: Optional[str] = Field(default=None, description="症状性质、性状、颜色、程度等核心特征")
    inducement: Optional[str] = Field(default=None, description="诱因、触发因素、相关背景")
    frequency: Optional[str] = Field(default=None, description="频率、次数、病程规律、持续情况")
    alleviating_factors: Optional[str] = Field(default=None, description="加重或缓解因素、与进食/体位/排便等关系")

    # 新增高频结构字段
    location: Optional[str] = Field(default=None, description="部位、分布范围、侧别")
    duration_pattern: Optional[str] = Field(default=None, description="持续时间、阵发/持续、昼夜规律、进展快慢")
    severity: Optional[str] = Field(default=None, description="严重程度、量、范围、影响程度")
    associated_symptoms: Optional[str] = Field(default=None, description="伴随症状")
    negative_symptoms: Optional[str] = Field(default=None, description="已明确否认的相关症状或红旗征")
    relation_to_food: Optional[str] = Field(default=None, description="与进食、食物类型、餐后关系")
    relation_to_bowel: Optional[str] = Field(default=None, description="与排便、排气、便意、便后变化的关系")
    relation_to_position: Optional[str] = Field(default=None, description="与体位、平卧、弯腰、坐起等关系")
    progression: Optional[str] = Field(default=None, description="是否进行性加重、逐渐增多、是否波动")

    dynamic_details: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="无法稳定归入固定字段的其他问诊细节，key 必须使用具体中文槽位名"
    )



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
            (("system", """你是一名专业的医疗信息结构化专家。

任务：
从患者最新表达中提取症状信息，并更新到对应的症状记录中。

【核心提取规则 - 必须严格遵守】
1. symptom_name 必须保留患者原话，不要改写成医学术语。
2. 如果患者一次性提供了多条信息，必须尽可能完整提取。
3. 如果患者否认某个已存在症状，必须将该症状的 status 设为 "revoked"。
4. 如果患者是在补充之前某个症状的细节，优先把信息写入对应症状，而不是新建无关症状。
5. 对于无法稳定归入固定字段的信息，写入 dynamic_details，key 使用具体中文槽位名，不要写“其他信息”“补充说明”这种泛化 key。
6. 如果上一轮 AI 问的是与既往病史比较的问题，则只允许把本轮回答写入“与既往病史关联”这一项，不要随意扩散到所有症状。
7. 不要臆造症状，不要脑补未说过的信息。

【字段填写要求】
1. onset_time：起病时间、发现时间、持续多久、多久前开始
2. location：部位、分布范围、左右侧、上腹/下腹/胸骨后/肛门等
3. characteristics：性质、颜色、性状、程度、具体表现
4. inducement：诱因、触发因素、相关背景
5. frequency：频率、次数、发作规律
6. alleviating_factors：加重或缓解因素
7. duration_pattern：阵发/持续、昼夜规律、进展快慢
8. severity：严重程度、出血量、体重下降幅度、影响程度
9. associated_symptoms：伴随症状
10. negative_symptoms：患者明确否认的相关症状或红旗项
11. relation_to_food：与进食、食物种类、餐后关系
12. relation_to_bowel：与排便、排气、便意、便后变化的关系
13. relation_to_position：与平卧、弯腰、坐起、站立等体位关系
14. progression：是否逐渐加重、是否增多、是否波动

【重要约束】
1. 能写固定字段就优先写固定字段，不要全部塞进 dynamic_details。
2. 同一个症状下可以同时填写多个字段。
3. negative_symptoms 只记录患者明确否认的内容。
4. dynamic_details 中的 key 要尽量贴近问诊槽位原意，例如：
   - “是否伴黑便、呕血”
   - “是否伴体重下降”
   - “包块是否活动、是否可回纳”
   不要写成“红旗征”“补充信息”。

【当前已记录的症状库】
{current_entities}

【近期对话上下文 (Short-term Memory)】
{chat_history_str}

【患者既往病史档案 (Long-term Memory)】
{long_term_memory_str}

{format_instructions}""")
),
            ("human", "患者最新回复：{patient_input}")
        ]).partial(format_instructions=self.extract_parser.get_format_instructions())
        self.extract_chain = self.extract_prompt | self.llm | self.extract_parser

        # ==========================================
        # 模块 B: Planner (动态医疗规划者)
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
2. 🚨【宽容与同义满足】：如果指南要素是复合词（如“抗酸药或抬高床头后能否缓解”），患者只要明确否定了（如“无”或“没有”），整条要素即视为排查完毕！绝对禁止再去抠字眼问剩下的词！
3. 🚨【阴性排查认同】：只要患者主动提及“没有XX”，视为该要素已排查完成。
4. 🚨【既往史单次询问机制（核心）】：
   当且仅当《指南》要求的所有要素都已经排查完毕后，你才可以考虑既往史；
   并且，只有“与当前症状直接相关”的既往史，才允许触发对比追问。
   如果该相关症状的 dynamic_details 中还未包含 "与过往病史关联" 字段，
   你只能输出一条【宽泛的追问指令】：“向患者说明他之前也有过类似记录，粗略询问这次和上次比是否加重？有无新发伴随症状？”。
5. 🚨【绝对结束红线】：如果《指南》已排查完毕，且（无相关既往史，或者相关症状的 dynamic_details 中已经存在 "与过往病史关联" 字段），你必须且只能输出大写单词：COMPLETED。无论患者说什么，绝不许再生成任何问题！

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

    # =========================================================
    # 工具函数
    # =========================================================
    def _get_active_entities(self, current_entities: List[dict]) -> List[dict]:
        return [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

    def _is_history_comparison_question(self, text: str) -> bool:
        if not text:
            return False
        keywords = [
            "和之前相比", "和上次相比", "这次和上次比",
            "之前也有过", "之前也有类似的情况", "上次比",
            "有没有加重", "有无新发伴随症状", "有没有新的不舒服",
            "有没有出现其他新的不舒服", "症状有没有加重"
        ]
        return any(k in text for k in keywords)

    def _contains_history_compare_instruction(self, text: str) -> bool:
        if not text:
            return False
        keywords = [
            "之前也有过类似记录",
            "和上次比是否加重",
            "有无新发伴随症状",
            "之前也有过类似的情况",
            "有没有加重",
            "有没有出现其他新的不舒服",
            "和之前相比"
        ]
        return any(k in text for k in keywords)

    def _get_history_related_targets(self, current_entities: List[dict], long_term_memory_str: str) -> List[dict]:
        """
        只找“当前 active 症状”里，真正出现在既往病史文本中的症状。
        没出现，就视为“无相关既往史”。
        """
        if not current_entities or not long_term_memory_str:
            return []

        active_entities = self._get_active_entities(current_entities)
        memory_text = str(long_term_memory_str)

        no_related_flags = [
            "暂无与当前症状直接相关的既往史",
            "暂无与当前症状高度相关的既往史",
            "无既往病史记录",
            "首次就诊"
        ]
        if any(flag in memory_text for flag in no_related_flags):
            return []

        targets = []
        for ent in active_entities:
            standard_term = (ent.get("standard_term") or "").strip()
            symptom_name = (ent.get("symptom_name") or "").strip()

            candidates = []
            if standard_term and standard_term != "未知术语":
                candidates.append(standard_term)
            if symptom_name:
                candidates.append(symptom_name)

            if any(cand and cand in memory_text for cand in candidates):
                targets.append(ent)

        return targets

    def _has_history_relation(self, ent: dict) -> bool:
        dynamic_details = ent.get("dynamic_details", {})
        if not isinstance(dynamic_details, dict):
            return False
        value = str(dynamic_details.get("与过往病史关联", "")).strip()
        return value != ""

    def _all_history_targets_completed(self, current_entities: List[dict], long_term_memory_str: str) -> bool:
        targets = self._get_history_related_targets(current_entities, long_term_memory_str)
        if not targets:
            return True
        return all(self._has_history_relation(ent) for ent in targets)

    def _build_history_relation_updates(
        self,
        current_entities: List[dict],
        long_term_memory_str: str,
        patient_input: str
    ) -> List[dict]:
        """
        对“既往史对比回答”做定向写入：
        只更新真正与既往史相关的那个/那些症状。
        """
        targets = self._get_history_related_targets(current_entities, long_term_memory_str)
        if not targets:
            return []

        cleaned_answer = patient_input.strip()
        updates = []

        for ent in targets:
            updates.append({
                "symptom_name": ent.get("symptom_name"),
                "standard_term": ent.get("standard_term"),
                "status": ent.get("status", "active"),
                "dynamic_details": {
                    "与过往病史关联": cleaned_answer
                }
            })

        return updates

    # =========================================================
    # 模块 A：信息抽取
    # =========================================================
    def extract(
        self,
        patient_input: str,
        current_entities: List[dict] = None,
        last_ai_message: str = "",
        chat_history_str: str = "",
        long_term_memory_str: str = ""
    ) -> List[dict]:
        try:
            current_entities = current_entities or []

            # 如果上一轮是“既往史对比追问”，则不再让 LLM 自由写入
            # 直接做代码级定向更新
            if self._is_history_comparison_question(last_ai_message):
                targeted_updates = self._build_history_relation_updates(
                    current_entities=current_entities,
                    long_term_memory_str=long_term_memory_str,
                    patient_input=patient_input
                )
                if targeted_updates:
                    return targeted_updates
                return []

            entities_str = json.dumps(current_entities, ensure_ascii=False) if current_entities else "暂无记录"
            result_dict = self.extract_chain.invoke({
                "patient_input": patient_input,
                "current_entities": entities_str,
                "chat_history_str": chat_history_str,
                "long_term_memory_str": long_term_memory_str
            })
            extracted_symptoms = result_dict.get("symptoms", [])

            # 给缺失字段补默认值，但不覆盖 LLM 已明确给出的 revoked
            # for ent in extracted_symptoms:
            #     ent.setdefault("status", "active")
            #     ent.setdefault("dynamic_details", {})
            for ent in extracted_symptoms:
                ent.setdefault("status", "active")
                ent.setdefault("dynamic_details", {})
                ent.setdefault("location", None)
                ent.setdefault("duration_pattern", None)
                ent.setdefault("severity", None)
                ent.setdefault("associated_symptoms", None)
                ent.setdefault("negative_symptoms", None)
                ent.setdefault("relation_to_food", None)
                ent.setdefault("relation_to_bowel", None)
                ent.setdefault("relation_to_position", None)
                ent.setdefault("progression", None)

            print("\n" + "#" * 80)
            print(f"[Extractor] 患者输入: {patient_input}")
            print(f"[Extractor] LLM抽取结果(映射前): {json.dumps(extracted_symptoms, ensure_ascii=False)}")

            for idx, ent in enumerate(extracted_symptoms, start=1):
                status = ent.get("status", "active")
                symptom_name = (ent.get("symptom_name") or "").strip()

                if status == "active" and symptom_name:
                    # query = symptom_name
                    # if ent.get("characteristics"):
                    #     query += " " + ent["characteristics"]
                    query_parts = [symptom_name]

                    for key in [
                        "characteristics",
                        "location",
                        "associated_symptoms",
                        "relation_to_food",
                        "relation_to_bowel",
                        "relation_to_position",
                    ]:
                        value = ent.get(key)
                        if value:
                            query_parts.append(str(value))

                    query = " ".join(query_parts)

                    print(
                        f"[Extractor] 症状{idx} 开始映射 | "
                        f"symptom_name={symptom_name} | status={status} | query={query}"
                    )

                    standard_term = self.retriever.get_standard_term(query)
                    ent["standard_term"] = standard_term

                    print(
                        f"[Extractor] 症状{idx} 映射完成 | "
                        f"symptom_name={symptom_name} | standard_term={standard_term}"
                    )
                else:
                    print(
                        f"[Extractor] 症状{idx} 跳过映射 | "
                        f"symptom_name={symptom_name} | status={status}"
                    )

            print(f"[Extractor] 最终结果(映射后): {json.dumps(extracted_symptoms, ensure_ascii=False)}")
            print("#" * 80 + "\n")

            return extracted_symptoms


        except Exception as e:
            print(f"提取出错: {e}")
            return []

    # =========================================================
    # 模块 B + C：规划与生成回复
    # =========================================================
    def generate_reply(
        self,
        patient_input: str,
        current_entities: List[dict],
        last_ai_message: str = "",
        chat_history_str: str = "",
        long_term_memory_str: str = ""
    ) -> str:
        try:
            active_entities = self._get_active_entities(current_entities)

            if not active_entities:
                return "您好，系统似乎没有捕捉到您的不适症状，能请您再具体描述一下哪里不舒服吗？"

            # 如果上一轮问的是既往史对比问题：
            # - 没有真正相关既往史 -> 直接结束
            # - 或者相关症状已经写入 与过往病史关联 -> 直接结束
            if self._is_history_comparison_question(last_ai_message):
                history_targets = self._get_history_related_targets(active_entities, long_term_memory_str)
                if not history_targets:
                    return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"
                if self._all_history_targets_completed(active_entities, long_term_memory_str):
                    return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"

            guidelines_for_current_symptoms = {}
            for ent in active_entities:
                term = ent.get("standard_term") or ent.get("symptom_name")
                if term in self.clinical_guidelines:
                    guidelines_for_current_symptoms[term] = self.clinical_guidelines[term]

            if not guidelines_for_current_symptoms:
                guidelines_for_current_symptoms = {
                    "通用问诊法则": "请依次核实：发病时间、具体部位、性质、频率、诱因"
                }

            entities_str = json.dumps(active_entities, ensure_ascii=False, indent=2)
            guidelines_str = json.dumps(guidelines_for_current_symptoms, ensure_ascii=False, indent=2)

            planner_task_raw = self.planner_chain.invoke({
                "active_entities": entities_str,
                "retrieved_guidelines": guidelines_str,
                "chat_history_str": chat_history_str,
                "long_term_memory_str": long_term_memory_str
            })

            print(f"\n[Planner 思考与决策过程] ->\n{planner_task_raw}\n")

            planner_task_clean = re.sub(r"<thinking>.*?</thinking>", "", planner_task_raw, flags=re.DOTALL).strip()

            if not planner_task_clean and "COMPLETED" in planner_task_raw:
                planner_task_clean = "COMPLETED"


            # 如果 Planner 还在要求做既往史比较，但代码层判断：
            # 1) 没有真正相关的既往史目标；或
            # 2) 相关目标已经写完 与过往病史关联
            # 则强制收口为 COMPLETED
            if self._contains_history_compare_instruction(planner_task_clean):
                history_targets = self._get_history_related_targets(active_entities, long_term_memory_str)
                if (not history_targets) or self._all_history_targets_completed(active_entities, long_term_memory_str):
                    planner_task_clean = "COMPLETED"

            # 只要 Planner 输出 COMPLETED，就直接结束
            if "COMPLETED" in planner_task_clean:
                return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"

            reply = self.reply_chain.invoke({
                "patient_input": patient_input,
                "planner_task": planner_task_clean
            })
            return reply

        except Exception as e:
            print(f"生成回复出错: {e}")
            return "不好意思，系统开了个小差，您能再说一遍吗？"