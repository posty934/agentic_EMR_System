# import os
# import json
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
#
# load_dotenv()
#
#
# # === 1. 扩充立体的医学实体结构 ===
# class SymptomDetail(BaseModel):
#     symptom_name: str = Field(description="核心症状名称，保留患者原话")
#     standard_term: Optional[str] = Field(default=None, description="标准医学术语映射结果")
#     # === 🚀 新增状态字段 ===
#     status: str = Field(default="active",
#                         description="状态：'active'为确认存在的症状，'revoked'为患者明确否认、说错或排除的症状")
#
#     onset_time: Optional[str] = Field(default=None)
#     characteristics: Optional[str] = Field(default=None)
#     inducement: Optional[str] = Field(default=None)
#     frequency: Optional[str] = Field(default=None)
#     alleviating_factors: Optional[str] = Field(default=None)
#
#
# class CoarseExtractionResult(BaseModel):
#     symptoms: List[SymptomDetail] = Field(description="提取出的症状及其关联属性")
#
#
# # === 2. 核心逻辑：定义不同症状的“必填追问清单” ===
# # 字典的 Key 是标准医学术语，Value 是该术语必须集齐的属性字段
# SYMPTOM_SLOTS_MAP = {
#     "腹泻": ["onset_time", "characteristics", "frequency"],
#     "腹痛": ["onset_time", "characteristics", "inducement", "alleviating_factors"],
#     "胃痛": ["onset_time", "characteristics", "inducement", "alleviating_factors"],
#     "烧心": ["onset_time", "inducement"], # 烧心本身就是性质，重点问诱因
#     "反酸": ["onset_time", "inducement"],
#     "胃部不适": ["onset_time", "characteristics", "inducement"], # 模糊词必须追问性质
#     "default": ["onset_time", "characteristics", "inducement"]
# }
#
#
# class Agent1Extractor:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             api_key=os.getenv("LLM_API_KEY"),
#             base_url=os.getenv("LLM_BASE_URL"),
#             model=os.getenv("LLM_MODEL_NAME"),
#             temperature=0.1,
#         )
#
#         # --- 听写链 (抽取) 升级 ---
#         self.extract_parser = JsonOutputParser(pydantic_object=CoarseExtractionResult)
#         self.extract_prompt = ChatPromptTemplate.from_messages([
#             ("system", """你是一名专业的医疗问诊信息记录员。
#                     任务：从患者最新表达中提取病情信息，并更新到已有记录中。
#                     规则：
#                     1. 保持患者的原话表述。
#                     2. 🚨【上下文锚定】：结合AI的上一个问题，将提取的细节准确填入对应症状中。
#                     3. 如果患者回答“没有”、“不知道”，设为“无”或“不详”，不要填 null。
#                     4. 🚨【纠错与排除机制】：如果患者在最新回复中，明确澄清、否认或推翻了之前的某个症状（例如：“我没有便秘”、“我之前说错了，不拉肚子”、“那个不是的”），请务必提取出那个被否认的症状名，并将其 status 字段严格设为 "revoked"！如果确认存在，默认为 "active"。
#
#                     【系统当前已记录的症状库】:
#                     {current_entities}
#                     \n{format_instructions}"""),
#             ("human", "AI上一个问题：{last_ai_message}\n患者最新回复：{patient_input}")
#         ]).partial(format_instructions=self.extract_parser.get_format_instructions())
#         self.extract_chain = self.extract_prompt | self.llm | self.extract_parser
#
#         # --- 说话链 (动态追问) 升级 ---
#         self.reply_parser = StrOutputParser()
#         self.reply_prompt = ChatPromptTemplate.from_messages([
#             ("system", """你是一名专业的医院预问诊AI助手。
#             你的任务是根据后台系统下发的【追问任务清单】，用专业的医生口吻向患者提问。
#
#             【后台下发的追问任务清单】:
#             {task_list}
#
#             【绝对执行规则】：
#             1. 每次只能挑清单中的【一个】属性提问。
#             2. ⛔ 防死循环警报：如果患者的最新回复是“对的”、“是的”，或者明显已经回答过时间/诱因等问题，**绝对禁止**再次重复提问相同的问题！你应该假设该信息已被记录，直接越过它，去问清单里的【下一个】属性。
#             3. 如果患者回复的内容能同时满足性质和诱因（如：吃完火锅后火辣辣的），请直接跳过这两个属性的追问。
#             4. 如果【任务清单】显示为“所有症状已满足必填项”，且患者**还没有**明确表示结束，你只能提问：“除了这些，您还有其他不舒服的地方吗？
#             5. 如果用户明确表示结束，那么你只能回复"病情信息已收集完毕".””
#             """),
#             ("human", "患者刚刚说：{patient_input}\n请开始提问：")
#         ])
#         self.reply_chain = self.reply_prompt | self.llm | self.reply_parser
#
#     def extract(self, patient_input: str, current_entities: List[dict] = None, last_ai_message: str = "") -> List[dict]:
#         """执行粗提取操作，带上下文锚定"""
#         try:
#             entities_str = json.dumps(current_entities, ensure_ascii=False) if current_entities else "暂无记录"
#             result_dict = self.extract_chain.invoke({
#                 "patient_input": patient_input,
#                 "current_entities": entities_str,
#                 "last_ai_message": last_ai_message # 把锚点传给大模型
#             })
#             return result_dict.get("symptoms", [])
#         except Exception as e:
#             print(f"提取出错: {e}")
#             return []
#
#     def _generate_task_list(self, current_entities: List[dict]) -> str:
#         if not current_entities:
#             return "目前没有提取到任何症状，请引导患者描述不适。"
#
#         tasks = []
#         # === 🚀 核心逻辑：只对状态为 active 的症状进行追问 ===
#         active_entities = [ent for ent in current_entities if ent.get("status", "active") != "revoked"]
#
#         if not active_entities:
#             return "所有症状已被排除，请询问患者目前到底哪里不舒服。"
#
#         for ent in active_entities:
#             sym_name = ent.get("symptom_name", "未知")
#             std_term = ent.get("standard_term", "")
#             required_slots = SYMPTOM_SLOTS_MAP.get(std_term, SYMPTOM_SLOTS_MAP["default"])
#
#             missing_slots = []
#             for slot in required_slots:
#                 if not ent.get(slot):
#                     missing_slots.append(slot)
#
#             if missing_slots:
#                 slot_trans = {
#                     "onset_time": "发病时间", "characteristics": "性质/程度",
#                     "inducement": "诱因", "frequency": "发作频率", "alleviating_factors": "缓解/加重因素"
#                 }
#                 missing_cn = [slot_trans.get(s, s) for s in missing_slots]
#                 tasks.append(f"- 针对症状【{sym_name}】，尚缺: {', '.join(missing_cn)}")
#
#         if not tasks:
#             return "所有症状已满足必填项，无需继续深挖该症状。"
#         return "\n".join(tasks)
#
#     def generate_reply(self, patient_input: str, current_entities: List[dict]) -> str:
#         try:
#             # 1. 先用 Python 逻辑算出任务清单
#             task_list = self._generate_task_list(current_entities)
#
#             # === 🚀 核心大招：代码级状态机强拦截 (Rule-based Guardrail) ===
#             # 如果 Python 规则字典判定：所有必填项都问完了
#             if "所有症状已满足" in task_list:
#                 # 定义结束关键词的判断
#                 finish_keywords = ["没有", "没", "无", "好了", "差不多", "就这些", "没别的"]
#
#                 # 只要患者的话里包含了结束词，或者回复特别短
#                 if any(kw in patient_input for kw in finish_keywords) or len(patient_input) <= 2:
#                     # 【物理拦截】直接返回触发词，不给大模型自由发挥的机会！
#                     return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"
#                 else:
#                     # 【物理拦截】强制它只能问这一句话
#                     return "除了这些，您还有其他不舒服的地方吗？"
#
#             # 2. 只有当任务没满时，才放权给大模型去生成自然追问
#             reply = self.reply_chain.invoke({
#                 "patient_input": patient_input,
#                 "task_list": task_list
#             })
#             return reply
#
#         except Exception as e:
#             print(f"生成回复出错: {e}")
#             return "不好意思，系统开了个小差，您能再说一遍吗？"

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# 引入我们自己写的 RAG 检索器
from knowledge.retriever import MedicalRetriever

load_dotenv()


class SymptomDetail(BaseModel):
    symptom_name: str = Field(description="核心症状名称，保留患者原话")
    standard_term: Optional[str] = Field(default=None, description="标准医学术语映射结果")
    status: str = Field(default="active", description="状态：'active'为确认存在的症状，'revoked'为排除")

    # 既然引入了 Planner，我们不再死板限制只有这几个字段，而是用一个字典存动态细节
    # 但为了兼容你之前的 UI 和逻辑，我们保留常见字段，增加一个 dynamic_details 灵活应对
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
            temperature=0.1,
        )

        self.retriever = MedicalRetriever()

        # ==========================================
        # 模块 A: Extractor (信息抽取员) - 🚨 修复对患者主动提供的信息视而不见
        # ==========================================
        self.extract_parser = JsonOutputParser(pydantic_object=CoarseExtractionResult)
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名专业的医疗信息结构化专家。
            任务：从患者最新表达中提取病情信息，并更新到对应的症状记录中。

            【核心提取规则 - 必须严格遵守】：
            1. 保持患者的原话表述。
            2. 🚨【主动补全机制】：如果患者一次性提供了一大段病情（例如“大便水样状，没有拉完想拉的感觉，可能是吃火锅辣的，没有发烧”），你【必须】将所有这些细节全部提取出来！
               - 无法放入基础字段的，必须存入 `dynamic_details` 字典。例如患者主动说“没有发烧”，存为 {"发烧": "无"}；说“没有拉完还想拉”，存为 {"里急后重感": "无"}。绝不能丢弃患者主动提供的阴性或阳性症状！
            3. 【上下文锚定】：如果患者的回答非常简短，结合“AI上一个问题”来判断他回答的是什么字段，并精准填入。
            4. 【基础字段精准对齐】：
               - 时间/多久了 -> onset_time
               - 频率（如"一阵一阵"、"一天三次"） -> frequency 
               - 性质（如"水样便"、"绞痛"） -> characteristics
               - 诱因（如"吃火锅"、"受凉"） -> inducement
            5. 如果患者推翻了某症状，将 status 设为 "revoked"。

            【当前已记录的症状库】:\n{current_entities}\n{format_instructions}"""),
            ("human", "AI上一个问题：{last_ai_message}\n患者最新回复：{patient_input}")
        ]).partial(format_instructions=self.extract_parser.get_format_instructions())
        self.extract_chain = self.extract_prompt | self.llm | self.extract_parser

        # ==========================================
        # 模块 B: Planner (动态医疗规划者) - 🚨 修复死板追问的强迫症
        # ==========================================
        with open("knowledge/clinical_guidelines.json", "r", encoding="utf-8") as f:
            self.clinical_guidelines = json.load(f)

        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名极其严谨但懂得变通的消化内科主治医师（Planner Agent）。
            你手里拿着《国家执业医师病史采集指南》。

            【当前患者的症状与已收集信息】:
            {active_entities}

            【官方临床问诊指南 (针对当前症状)】:
            {retrieved_guidelines}

            【规划规则】:
            1. 对照《指南》，检查【当前患者信息】是否收集全了。🚨务必检查 `dynamic_details`！
            2. 🚨【宽容匹配原则】：如果患者已经描述了某个要素的核心特征（例如大便性状已记录为'水样便'），则直接视为该要素收集完毕！绝对不要再去死板地追问“有没有黏液、有没有血丝”！
            3. 🚨【阴性排查原则】：如果患者主动说明了“没有XX”（如没有发烧、没有口渴，见 `dynamic_details`），也视为该要素排查完毕。
            4. 🚨【结束红线】：如果指南要求的主要维度都已大致覆盖，必须、立刻、只输出单词：COMPLETED。绝对不要输出其他任何解释文字！
            5. 如果确实缺少核心要素，每次只挑【一个】最紧急的要素下达追问指令。

            直接输出对前台助手下达的追问指令。
            """),
            ("human", "请查阅指南后，给出下一步追问指令：")
        ])
        self.planner_chain = self.planner_prompt | self.llm | StrOutputParser()

        # ==========================================
        # 模块 C: Speaker (前台问诊员)
        # ==========================================
        self.reply_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名温和、专业的医院预问诊AI助手。
            你的任务是将后台主治医师（Planner）下达的【追问指令】，转化为对患者的自然语言提问。

            【主治医师下达的追问指令】:
            {planner_task}

            【规则】：
            1. 使用关切的口吻，话术要简短，像真人在微信聊天一样。
            2. 不要向患者暴露“后台医师”、“指南”、“指令”等字眼。
            """),
            ("human", "患者刚刚说：{patient_input}\n请开始提问：")
        ])
        self.reply_chain = self.reply_prompt | self.llm | StrOutputParser()

    def extract(self, patient_input: str, current_entities: List[dict] = None, last_ai_message: str = "") -> List[dict]:
        """执行粗提取 -> RAG术语映射 -> 实体融合"""
        try:
            entities_str = json.dumps(current_entities, ensure_ascii=False) if current_entities else "暂无记录"

            # 1. 大模型粗提取
            result_dict = self.extract_chain.invoke({
                "patient_input": patient_input,
                "current_entities": entities_str,
                "last_ai_message": last_ai_message
            })
            extracted_symptoms = result_dict.get("symptoms", [])

            # 2. 🚀 RAG 术语动态映射
            for ent in extracted_symptoms:
                if ent.get("status") == "active" and ent.get("symptom_name"):
                    # 构建更丰富的 Query 去检索，以提高命中率
                    query = ent["symptom_name"]
                    if ent.get("characteristics"):
                        query += " " + ent["characteristics"]

                    # 调用 retriever 获取标准术语
                    standard_term = self.retriever.get_standard_term(query)
                    ent["standard_term"] = standard_term

            return extracted_symptoms
        except Exception as e:
            print(f"提取出错: {e}")
            return []

    def generate_reply(self, patient_input: str, current_entities: List[dict], last_ai_message: str = "") -> str:
        try:
            active_entities = [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

            if not active_entities:
                return "您好，系统似乎没有捕捉到您的不适症状，能请您再具体描述一下哪里不舒服吗？"

            # ==========================================
            # 🚀 知识检索阶段 (RAG for Planner)
            # ==========================================
            guidelines_for_current_symptoms = {}
            for ent in active_entities:
                term = ent.get("standard_term") or ent.get("symptom_name")
                if term in self.clinical_guidelines:
                    guidelines_for_current_symptoms[term] = self.clinical_guidelines[term]

            if not guidelines_for_current_symptoms:
                guidelines_for_current_symptoms = {"通用问诊法则": "请依次核实：发病时间、具体部位、性质、频率、诱因"}

            # ==========================================
            # 🚀 规划阶段 (带着知识去思考)
            # ==========================================
            entities_str = json.dumps(active_entities, ensure_ascii=False, indent=2)
            guidelines_str = json.dumps(guidelines_for_current_symptoms, ensure_ascii=False, indent=2)

            planner_task = self.planner_chain.invoke({
                "active_entities": entities_str,
                "retrieved_guidelines": guidelines_str
            })

            print(f"[Planner 查阅指南后决策] -> {planner_task}")

            if "COMPLETED" in planner_task:
                # 🚨 修复误杀：更严谨的结束判断逻辑
                clean_input = patient_input.strip().replace("，", "").replace("。", "").replace("！", "")
                exact_finish_words = ["没有", "没", "无", "好了", "差不多", "就这些", "没别的", "没有了", "没了",
                                      "不需要了"]

                is_ending = False

                # 情况1：用户仅仅回复了极其简短的结束词（如单纯回复“没有了”）
                if clean_input in exact_finish_words:
                    is_ending = True
                # 情况2：AI 上一个问题确实是在问“其他”，且用户回答包含了否定词
                elif "其他" in last_ai_message and any(kw in patient_input for kw in ["没有", "没", "无", "没别的"]):
                    is_ending = True

                if is_ending:
                    return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"
                else:
                    return "除了这些，您还有其他不舒服的地方吗？"

            # 3. 将指令转化为人话发给用户
            reply = self.reply_chain.invoke({
                "patient_input": patient_input,
                "planner_task": planner_task
            })
            return reply

        except Exception as e:
            print(f"生成回复出错: {e}")
            return "不好意思，系统开了个小差，您能再说一遍吗？"