import os
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from validators.kg_validator import KGValidator

from knowledge.retriever import MedicalRetriever

load_dotenv()


class SymptomDetail(BaseModel):
    symptom_name: str = Field(description="核心症状名称，保留患者原话")
    standard_term: Optional[str] = Field(default=None, description="标准医学术语映射结果")
    status: str = Field(default="active", description="状态：active 为确认存在的症状，revoked 为患者否认或撤销")

    onset_time: Optional[str] = Field(default=None, description="起病或发现时间")
    characteristics: Optional[str] = Field(default=None, description="症状性质、性状、颜色、程度等核心特征")
    inducement: Optional[str] = Field(default=None, description="诱因、触发因素、相关背景")
    frequency: Optional[str] = Field(default=None, description="频率、次数、病程规律、持续情况")
    alleviating_factors: Optional[str] = Field(default=None, description="加重或缓解因素、与进食/体位/排便等关系")

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
        self.kg_validator = KGValidator()


        self.debug = os.getenv("AGENT_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

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
4. 如果患者是在补充之前某个症状的细节，必须优先把信息写入对应症状，而不是新建无关症状。
   尤其当 current_entities 中已经存在某个症状时，本轮如果只是继续回答这个症状的细节，
   symptom_name 必须复用已有记录中的 symptom_name，不要改写成新的近义说法。
   例如：系统已存在“便秘”，本轮患者说“大便干硬、四五天一次、排便很费力”，
   仍应把 symptom_name 写成“便秘”，而不是新建“大便干硬”或其他新症状。
5. 对于无法稳定归入固定字段的信息，写入 dynamic_details，key 使用具体中文槽位名，不要写“其他信息”“补充说明”这种泛化 key。
6. 如果上一轮 AI 问的是与既往病史比较的问题，则只允许把本轮回答写入“与既往病史关联”这一项，不要随意扩散到所有症状。
7. 不要臆造症状，不要脑补未说过的信息。
8. 如果患者本轮只是对上一轮 AI 问题作简短直接回答（如“有”“没有”“会”“不会”“会好一些”“更明显”“饭后更明显”），
   必须优先结合【上一轮 AI 刚刚问的问题】理解语义，并写入对应字段；不能因为患者没有复述完整问题而漏提取。
9. 如果 current_entities 中某个症状带有 "is_locked": true，说明该症状的预定义问诊要点已经收集完毕。
   除非患者本轮是在明确撤销这个症状，否则禁止继续把新信息写回这个已锁定症状。

10. 如果上一轮 AI 问的是复合枚举问题，例如：
   - “油腻饮食、饮酒、咖啡或夜宵后是否更明显”
   - “抗酸药或抬高床头后能否缓解”
   那么患者只要：
   - 明确肯定其中任一项；或
   - 对整题作总体肯定/否定；
   就视为该复合要素已经回答过。
   本轮只提取患者明确说出的内容，不要把未提及分支脑补成阴性，也不要因为没有逐项展开而认为该复合要素仍未回答。

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

【上一轮 AI 刚刚问的问题】
{last_ai_message}

【近期对话上下文 (Short-term Memory)】
{chat_history_str}

【患者既往病史档案 (Long-term Memory)】
{long_term_memory_str}

【跨轮已完成问答轨迹 (Anti-Repeat Memory)】
{qa_trace_str}

【跨轮补充规则】
11. 如果【跨轮已完成问答轨迹】表明某个问点已经被患者明确回答，本轮又是在继续补充这个问点，必须把信息写回对应已有症状，不要新建同义症状。
12. 对于“有 / 没有 / 会 / 不会 / 饭后明显 / 能缓解一些”这类短答，必须优先结合 last_ai_message 和【跨轮已完成问答轨迹】理解其指向；不要因为回答简短就漏提取。
13. 如果 current_entities 中已有同一医学概念的症状，本轮即使换了说法，也优先复用已有 symptom_name；不要把同一症状拆成两个实体。


{format_instructions}""")),
            ("human", "患者最新回复：{patient_input}")
        ]).partial(format_instructions=self.extract_parser.get_format_instructions())
        self.extract_chain = self.extract_prompt | self.llm | self.extract_parser

        # ==========================================
        # 模块 B + C 合并：Dialogue Planner-Speaker
        # ==========================================
        with open("knowledge/clinical_guidelines.json", "r", encoding="utf-8") as f:
            self.clinical_guidelines = json.load(f)

        self.dialogue_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名极其严谨且具备临床诊断思维的消化内科主治医师，同时也是一名温和、专业的医院预问诊AI助手。

你的任务不是输出后台指令，而是：
- 先在内部严格完成临床规划；
- 再直接输出一句能够对患者说的话。
如果满足结束条件，则只输出：COMPLETED

【当前患者的症状与已收集信息】:
{active_entities}

【官方临床问诊指南 (针对当前症状)】:
{retrieved_guidelines}

【上一轮 AI 刚刚问的问题】:
{last_ai_message}

【近期对话上下文 (Short-term Memory)】:
{chat_history_str}

【患者既往病史档案 (Long-term Memory)】:
{long_term_memory_str}

【跨轮已完成问答轨迹 (Anti-Repeat Memory)】
{qa_trace_str}

【规划与推理规则】:
1. 🚨【现病史优先原则】：必须优先对照《指南》排查当前的症状要素。未收集完前，绝对不许提及既往史！
2. 🚨【宽容与同义满足】：如果指南要素是复合词（如“抗酸药或抬高床头后能否缓解”），患者只要明确否定了（如“无”或“没有”），整条要素即视为排查完毕！绝对禁止再去抠字眼问剩下的词！
2.1 🚨【直接回答承接规则】：
如果患者本轮只是对上一轮 AI 问题作简短直接回答，如“会”“不会”“有”“没有”“更明显”“会好一些”，
必须优先判定为“患者已经回答了上一轮问题”，禁止原样或换句话重复再问一遍同一个要素。

2.2 🚨【复合枚举题完成规则】：
对于“A/B/C/D 后是否更明显”“X 或 Y 能否缓解”这类复合问题，
患者只要：
- 明确提到其中任一项为阳性；或
- 对整题作总体肯定；或
- 对整题作总体否定；
就视为这个复合要素已经完成。
不要因为患者没有逐项覆盖所有分支，就继续追问同级分支。

2.3 🚨【禁止同义重复追问】：
在输出下一句前，必须先检查：
上一轮 AI 的问题是否已经被患者本轮回答。
如果已经回答，下一句必须切换到别的未完成要素；
绝对禁止把同一个问题换一种说法重新问一次。

2.4 🚨【跨轮问答轨迹优先】：
判断某个问点是否已经问过/答过时，优先参考【跨轮已完成问答轨迹】。
只要轨迹中该问点已有明确患者回答，就视为已覆盖，不得再次用同义句重复追问。

2.5 🚨【仅两种情况允许重问】：
只有在以下两种情况，才允许对同一点再次追问：
- 患者前次回答与当前信息明显冲突；
- 患者前次回答明显答非所问，仍不足以判断。
除这两种情况外，禁止重复追问已经答过的点。


3. 🚨【阴性排查认同】：只要患者主动提及“没有XX”，视为该要素已排查完成。
4. 🚨【既往史单次询问机制（核心）】：
   当且仅当《指南》要求的所有要素都已经排查完毕后，你才可以考虑既往史；
   并且，只有“与当前症状直接相关”的既往史，才允许触发对比追问。
   如果该相关症状的 dynamic_details 中还未包含 "与过往病史关联" 字段，
   你只能向患者做一次宽泛追问：说明之前也有过类似记录，粗略询问这次和上次比是否加重？有无新发伴随症状？
5. 🚨【新增症状确认机制（核心）】：
   当《指南》要求的所有要素都已经排查完毕，且相关既往史比较也已完成时，
   这只代表“当前已知症状的信息完整”，不代表整个问诊结束。
   你必须先判断近期对话上下文中，AI 是否已经明确询问过：
   “除了这些，您还有其他不舒服的地方吗？”、“还有没有其他症状？” 这类“新增症状确认问题”。

   - 如果还没有问过，你不能输出 COMPLETED。
     你只能输出一句对患者的自然追问：询问是否还有新增症状或其他不适。
   - 只有当上一轮 AI 已经明确询问过“还有没有其他不舒服”，
     且患者这一次明确回答“没有了 / 就这些 / 没别的 / 暂时没有”等结束语时，
     你才可以输出 COMPLETED。
6. 🚨【绝对结束红线】：
   只有同时满足以下条件时，才允许输出 COMPLETED：
   - 当前所有已知症状的指南要素已排查完毕；
   - 无相关既往史，或相关既往史比较已完成；
   - 已经完成“是否还有新增症状”的显式确认，且患者明确否认还有新症状。

【输出规则】：
1. 若满足【绝对结束红线】，只输出：COMPLETED
2. 否则，只输出一句可以直接对患者说的话；必须短、自然、温和、专业。
3. 一次只问一个最紧急的问题。
4. 不要输出你的思考过程，不要输出“指令：”“分析：”“下一步：”之类前缀。
5. 不要使用项目符号，不要解释你的规划依据。
6. 不要向患者暴露“后台医师”“指南”“指令”“规划”“结构化信息”“既往史检索”等字眼。
7. 除了 COMPLETED 之外，输出内容必须是患者可见话术，而不是后台任务描述。"""),
            ("human", "患者刚刚说：{patient_input}\n请直接给出下一句对患者说的话；如果已经满足结束条件，只输出 COMPLETED。")
        ])
        self.dialogue_chain = self.dialogue_prompt | self.llm | StrOutputParser()
        self.single_slot_question_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名温和、专业的医院预问诊AI助手。
        你的唯一任务：把“指定的单个问诊要点”改写成一句自然、简短、患者可直接回答的追问。

        【当前症状】
        {target_symptom}

        【本轮唯一允许追问的要点】
        {target_slot}

        【本轮允许涉及的关键词】
        {allowed_keywords}

        【上一轮 AI 刚刚问的问题】
        {last_ai_message}

        【近期对话上下文】
        {chat_history_str}

        【严格规则】
        1. 只能围绕【本轮唯一允许追问的要点】发问。
        2. 可以有一句很短的承接，例如“好的，我再确认一下”；但真正的问题内容必须与该要点一一对应。
        3. 不得新增该要点之外的任何新症状、新红旗、新例子、新并列项。
        4. 如果要点里是“A/B/C”这种并列内容，你只能使用这些原有内容或它们的直接口语化表达，禁止额外扩展成新的 D/E/F。
        5. 例如：要点是“是否伴有极度口渴/尿少(脱水风险)”时，只能问口渴、尿少，绝不能扩展为头晕、心慌、眼前发黑、乏力、精神差等。
        6. 只输出一句患者可见的话，不要解释。"""),
            ("human", "请把这个单个问诊要点改写成一句自然追问：{target_slot}")
        ])
        self.single_slot_question_chain = self.single_slot_question_prompt | self.llm | StrOutputParser()

    # =========================================================
    # 工具函数
    # =========================================================
    def _get_agent1_visible_entities(
        self,
        current_entities: List[dict],
        pending_question_target: Optional[Dict[str, str]] = None
    ) -> List[dict]:
        visible = []

        pending_symptom = ""
        if pending_question_target:
            pending_symptom = (
                pending_question_target.get("symptom_name")
                or pending_question_target.get("standard_term")
                or ""
            ).strip()

        for ent in current_entities or []:
            if ent.get("status", "active") == "revoked":
                continue

            symptom_name = (ent.get("symptom_name") or "").strip()
            standard_term = (ent.get("standard_term") or "").strip()

            # 未锁定症状：始终给 Agent1 看
            if not ent.get("is_locked", False):
                visible.append(ent)
                continue

            # 已锁定症状：默认不给 Agent1 看
            # 但如果它正好是 pending target 对应症状，允许保留
            if pending_symptom and pending_symptom in {symptom_name, standard_term}:
                visible.append(ent)

        return visible

    def _ensure_entity_meta(self, ent: dict) -> dict:
        ent.setdefault("slot_answers", {})
        ent.setdefault("is_locked", False)
        ent.setdefault("dynamic_details", {})
        return ent

    def _is_negative_confirmation_question(self, text: str) -> bool:
        q = str(text or "").strip()
        q_norm = re.sub(r"\s+", "", q)

        if not q_norm:
            return False

        # 这些本质是“有无/有没有/是否有”开放问句，不是“否定确认句”
        if any(x in q_norm for x in ["有没有", "有无", "是否有", "有没有那种", "有没有这种", "有沒有"]):
            return False

        # 真正的否定确认句：前面已经明确给出了“没有/无/未见”，后面在让患者确认
        if any(x in q_norm for x in ["完全没有", "并没有", "都没有", "未见", "无明显"]):
            if any(x in q_norm for x in ["对吗", "对不对", "是吗", "对吧", "吧"]):
                return True

        if "没有" in q_norm and any(x in q_norm for x in ["对吗", "对不对", "是吗", "对吧", "吧"]):
            return True

        return False

    def _normalize_answer_by_question(self, patient_input: str, last_ai_message: str) -> str:
        answer = str(patient_input or "").strip()

        # 只有当上一问是真正“否定确认句”时，回答“对/是”才需要折算成“没有”
        if self._is_negative_confirmation_question(last_ai_message) and self._is_affirmative_answer(answer):
            return "没有"

        return answer

    def _extract_existing_answer_for_slot(self, ent: dict, slot: str) -> str:
        slot_answers = ent.get("slot_answers", {})
        if isinstance(slot_answers, dict):
            value = slot_answers.get(slot)
            if value is not None and str(value).strip() != "":
                return str(value).strip()

        field = self.kg_validator._infer_field_from_slot(slot)
        strict_match_fields = {
            "associated_symptoms",
            "negative_symptoms",
            "inducement",
            "alleviating_factors",
            "relation_to_food",
            "relation_to_bowel",
            "relation_to_position",
        }

        if field:
            value = ent.get(field)
            if value is not None and str(value).strip() != "":
                if field in strict_match_fields:
                    if self.kg_validator._field_value_matches_slot(value, slot):
                        return str(value).strip()
                else:
                    return str(value).strip()

        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            for dk, dv in dynamic_details.items():
                if dv is None or str(dv).strip() == "":
                    continue
                dk_norm = self.kg_validator._normalize_text(dk)
                slot_norm = self.kg_validator._normalize_text(slot)
                if dk_norm == slot_norm or dk_norm in slot_norm or slot_norm in dk_norm:
                    return str(dv).strip()

        return ""

    def _entity_all_guideline_slots_completed(self, ent: dict) -> bool:
        if not ent or ent.get("status", "active") == "revoked":
            return False

        standard_term = (ent.get("standard_term") or ent.get("symptom_name") or "").strip()
        if not standard_term:
            return False

        slots = self._get_guideline_slots_for_symptom(standard_term)
        if not slots:
            return False

        slot_answers = ent.get("slot_answers", {})
        if not isinstance(slot_answers, dict):
            return False

        return all(str(slot_answers.get(slot, "")).strip() != "" for slot in slots)

    def _cleanup_slot_shadow_details(self, ent: dict) -> dict:
        dynamic_details = ent.get("dynamic_details", {})
        if not isinstance(dynamic_details, dict) or not dynamic_details:
            return ent

        standard_term = (ent.get("standard_term") or ent.get("symptom_name") or "").strip()
        slots = self._get_guideline_slots_for_symptom(standard_term)

        if not slots:
            return ent

        cleaned = {}
        for dk, dv in dynamic_details.items():
            dk_norm = self.kg_validator._normalize_text(dk)
            is_slot_shadow = False

            for slot in slots:
                slot_norm = self.kg_validator._normalize_text(slot)
                if not dk_norm or not slot_norm:
                    continue
                if dk_norm == slot_norm or dk_norm in slot_norm or slot_norm in dk_norm:
                    is_slot_shadow = True
                    break

            if not is_slot_shadow:
                cleaned[dk] = dv

        ent["dynamic_details"] = cleaned
        return ent


    def refresh_entities_slot_state(self, current_entities: List[dict]) -> List[dict]:
        refreshed = []

        for ent in current_entities or []:
            copied = dict(ent or {})
            self._ensure_entity_meta(copied)

            if copied.get("status", "active") == "revoked":
                copied["is_locked"] = False
                refreshed.append(copied)
                continue

            standard_term = (copied.get("standard_term") or copied.get("symptom_name") or "").strip()
            slots = self._get_guideline_slots_for_symptom(standard_term)

            for slot in slots:
                if str(copied["slot_answers"].get(slot, "")).strip():
                    continue
                answer_text = self._extract_existing_answer_for_slot(copied, slot)
                if answer_text:
                    copied["slot_answers"][slot] = answer_text

            copied["is_locked"] = self._entity_all_guideline_slots_completed(copied)
            refreshed.append(copied)

        return refreshed

    def _build_forced_slot_update(
            self,
            pending_question_target: Dict[str, str],
            patient_input: str,
            last_ai_message: str
    ) -> dict:
        slot = pending_question_target["slot"]
        answer = self._normalize_answer_by_question(patient_input, last_ai_message)

        return {
            "symptom_name": pending_question_target["symptom_name"],
            "standard_term": pending_question_target["standard_term"],
            "status": "active",
            "onset_time": None,
            "characteristics": None,
            "inducement": None,
            "frequency": None,
            "alleviating_factors": None,
            "location": None,
            "duration_pattern": None,
            "severity": None,
            "associated_symptoms": None,
            "negative_symptoms": None,
            "relation_to_food": None,
            "relation_to_bowel": None,
            "relation_to_position": None,
            "progression": None,
            "dynamic_details": {},
            "slot_answers": {
                slot: answer
            },
            "is_locked": False,
        }

    def plan_next_turn(
        self,
        patient_input: str,
        current_entities: List[dict],
        last_ai_message: str = "",
        chat_history_str: str = "",
        long_term_memory_str: str = "",
        qa_trace_str: str = ""
    ) -> Dict[str, object]:
        current_entities = self.refresh_entities_slot_state(current_entities)
        active_entities = self._get_active_entities(current_entities)
        visible_entities = self._get_agent1_visible_entities(current_entities, None)

        if not active_entities:
            return {
                "reply": "您好，系统似乎没有捕捉到您的不适症状，能请您再具体描述一下哪里不舒服吗？",
                "pending_question_target": {},
                "entities": current_entities,
            }

        next_target = self._pick_next_guideline_target(visible_entities)

        if next_target:
            question = self._generate_single_slot_question(
                target_symptom=next_target["symptom"],
                target_slot=next_target["slot"],
                allowed_keywords=next_target["allowed_keywords"],
                last_ai_message=last_ai_message,
                chat_history_str=chat_history_str
            )
            return {
                "reply": question,
                "pending_question_target": {
                    "type": "guideline_slot",
                    "symptom_name": next_target["symptom_name"],
                    "standard_term": next_target["standard_term"],
                    "slot": next_target["slot"],
                },
                "entities": current_entities,
            }

        history_targets = self._get_history_related_targets(active_entities, long_term_memory_str)
        if history_targets and not self._all_history_targets_completed(active_entities, long_term_memory_str):
            return {
                "reply": "我看到您之前也有过类似不适，我再确认一下，这次和之前比有没有更重一些，或者有没有新出现的不舒服？",
                "pending_question_target": {},
                "entities": current_entities,
            }

        if self._is_asking_about_additional_symptoms(last_ai_message):
            if self._patient_explicitly_declines_more_symptoms(patient_input):
                return {
                    "reply": "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。",
                    "pending_question_target": {},
                    "entities": current_entities,
                }
            return {
                "reply": "好的，您再具体说一下还有什么不舒服，好吗？",
                "pending_question_target": {"type": "additional_symptoms"},
                "entities": current_entities,
            }

        return {
            "reply": "除了这些，您还有其他不舒服的地方吗？",
            "pending_question_target": {"type": "additional_symptoms"},
            "entities": current_entities,
        }

    def _get_guideline_slots_for_symptom(self, symptom_term: str) -> List[str]:
        ordered_slots = []
        guideline = self.clinical_guidelines.get(symptom_term, {})

        if isinstance(guideline, dict):
            for section in ["必问核心要素", "必问鉴别要素", "高危红旗征(必须排查)"]:
                values = guideline.get(section, [])
                if isinstance(values, list):
                    ordered_slots.extend([str(v).strip() for v in values if str(v).strip()])

        if not ordered_slots and self.kg_validator.graph.has_symptom(symptom_term):
            ordered_slots.extend(self.kg_validator.graph.get_required_slots(symptom_term))
            ordered_slots.extend(self.kg_validator.graph.get_redflag_slots(symptom_term))

        deduped = []
        seen = set()
        for slot in ordered_slots:
            if slot and slot not in seen:
                deduped.append(slot)
                seen.add(slot)

        return deduped

    def _get_unlocked_active_entities(self, current_entities: List[dict]) -> List[dict]:
        return [
            ent for ent in self._get_active_entities(current_entities)
            if not ent.get("is_locked", False)
        ]

    def _pick_next_guideline_target(self, current_entities: List[dict]) -> Optional[Dict[str, str]]:
        for ent in self._get_unlocked_active_entities(current_entities):
            standard_term = (ent.get("standard_term") or ent.get("symptom_name") or "").strip()
            display_name = (ent.get("symptom_name") or standard_term).strip()

            if not standard_term:
                continue

            for slot in self._get_guideline_slots_for_symptom(standard_term):
                if self.kg_validator._slot_is_filled(ent, slot):
                    continue

                keywords = []
                seen = set()
                for kw in self.kg_validator._derive_keywords_from_slot(slot):
                    kw = str(kw).strip()
                    if kw and kw not in seen:
                        keywords.append(kw)
                        seen.add(kw)

                return {
                    "symptom": display_name,
                    "symptom_name": display_name,
                    "standard_term": standard_term,
                    "slot": slot,
                    "allowed_keywords": "、".join(keywords[:12]) if keywords else slot
                }

        return None

    def _generate_single_slot_question(
        self,
        target_symptom: str,
        target_slot: str,
        allowed_keywords: str,
        last_ai_message: str = "",
        chat_history_str: str = ""
    ) -> str:
        raw = self.single_slot_question_chain.invoke({
            "target_symptom": target_symptom,
            "target_slot": target_slot,
            "allowed_keywords": allowed_keywords,
            "last_ai_message": last_ai_message,
            "chat_history_str": chat_history_str,
        })

        clean = re.sub(r"<thinking>.*?</thinking>", "", str(raw), flags=re.DOTALL).strip()
        clean = re.sub(r"\s+", " ", clean)

        if clean and "COMPLETED" not in clean.upper():
            return clean

        return f"我再确认一下，关于您这次{target_symptom}的情况，{target_slot}？"

    def _is_brief_direct_answer(self, text: str) -> bool:
        normalized = self.kg_validator._normalize_text(text)
        if not normalized:
            return False

        direct_words = {
            "有", "没有", "无", "没", "对", "是", "不是", "不对",
            "会", "不会", "有的", "没有的", "有点", "一点点",
            "还好", "明显", "不明显", "是的", "不是的"
        }
        if normalized in direct_words:
            return True

        raw = str(text or "").strip()
        if len(normalized) <= 4 and not any(x in raw for x in ["，", ",", "。", "；", ";", "但是", "不过", "因为", "所以"]):
            return True

        return False

    def _is_negative_answer(self, text: str) -> bool:
        normalized = self.kg_validator._normalize_text(text)
        return normalized in {
            "没有", "无", "没", "不是", "不对", "没有的", "未见", "并没有"
        }

    def _is_affirmative_answer(self, text: str) -> bool:
        normalized = self.kg_validator._normalize_text(text)
        return normalized in {
            "有", "对", "是", "会", "有的", "是的", "对的"
        }

    def _compact_slot_text(self, slot: str) -> str:
        text = str(slot or "").strip()
        text = re.sub(r"[（(].*?[）)]", "", text)
        for prefix in [
            "是否伴有", "是否伴", "是否有", "有无", "是否", "最近有没有",
            "大便有没有", "您的大便有没有", "您有没有", "那您有没有"
        ]:
            if text.startswith(prefix):
                text = text[len(prefix):]
        text = text.replace("/", "、").replace("或", "、")
        text = text.replace("情况", "").replace("症状", "")
        text = text.strip("：:，。！？?呢吗 ")
        return text or str(slot or "").strip()

    def _entity_all_guideline_slots_completed(self, ent: dict) -> bool:
        if not ent or ent.get("status", "active") == "revoked":
            return False

        standard_term = (ent.get("standard_term") or ent.get("symptom_name") or "").strip()
        if not standard_term:
            return False

        slots = self._get_guideline_slots_for_symptom(standard_term)
        if not slots:
            return False

        return all(self.kg_validator._slot_is_filled(ent, slot) for slot in slots)

    def refresh_entities_lock_state(self, current_entities: List[dict]) -> List[dict]:
        refreshed = []
        for ent in current_entities or []:
            copied = dict(ent or {})
            if copied.get("status", "active") != "revoked":
                copied["is_locked"] = bool(copied.get("is_locked", False) or self._entity_all_guideline_slots_completed(copied))
            refreshed.append(copied)
        return refreshed

    def _infer_question_target(self, last_ai_message: str, current_entities: List[dict]) -> Optional[Dict[str, str]]:
        if not last_ai_message:
            return None
        if self._is_asking_about_additional_symptoms(last_ai_message):
            return None
        if self._is_history_comparison_question(last_ai_message):
            return None

        q_norm = self.kg_validator._normalize_text(last_ai_message)
        best = None
        best_score = 0

        for ent in self._get_active_entities(current_entities):
            standard_term = (ent.get("standard_term") or ent.get("symptom_name") or "").strip()
            symptom_name = (ent.get("symptom_name") or standard_term).strip()
            if not standard_term:
                continue

            for slot in self._get_guideline_slots_for_symptom(standard_term):
                score = 0
                candidates = [slot] + self.kg_validator._derive_keywords_from_slot(slot)
                seen = set()

                for cand in candidates:
                    cand = str(cand).strip()
                    cand_norm = self.kg_validator._normalize_text(cand)
                    if not cand_norm or cand_norm in seen:
                        continue
                    seen.add(cand_norm)

                    if cand_norm in q_norm:
                        score += 3 if cand_norm == self.kg_validator._normalize_text(slot) else 1

                if score > best_score:
                    best_score = score
                    best = {
                        "symptom_name": symptom_name,
                        "standard_term": standard_term,
                        "slot": slot
                    }

        return best if best_score >= 2 else None

    def _sanitize_dynamic_details(
        self,
        dynamic_details: dict,
        candidate_slots: List[str],
        force_slot: Optional[str] = None
    ) -> dict:
        cleaned = {}
        if not isinstance(dynamic_details, dict):
            return cleaned

        for dk, dv in dynamic_details.items():
            if dv is None or str(dv).strip() == "":
                continue

            if force_slot:
                cleaned[force_slot] = str(dv).strip()
                continue

            dk_norm = self.kg_validator._normalize_text(dk)
            matched_slot = None

            for slot in candidate_slots or []:
                slot_norm = self.kg_validator._normalize_text(slot)
                if not slot_norm:
                    continue
                if dk_norm == slot_norm or dk_norm in slot_norm or slot_norm in dk_norm:
                    matched_slot = slot
                    break

            if matched_slot:
                cleaned[matched_slot] = str(dv).strip()

        return cleaned

    def _build_fallback_slot_update(
            self,
            question_target: Dict[str, str],
            patient_input: str,
            last_ai_message: str
    ) -> dict:
        slot = question_target["slot"]
        answer = self._normalize_answer_by_question(patient_input, last_ai_message)

        return {
            "symptom_name": question_target["symptom_name"],
            "standard_term": question_target["standard_term"],
            "status": "active",
            "onset_time": None,
            "characteristics": None,
            "inducement": None,
            "frequency": None,
            "alleviating_factors": None,
            "location": None,
            "duration_pattern": None,
            "severity": None,
            "associated_symptoms": None,
            "negative_symptoms": None,
            "relation_to_food": None,
            "relation_to_bowel": None,
            "relation_to_position": None,
            "progression": None,
            "dynamic_details": {},
            "slot_answers": {
                slot: answer
            },
            "is_locked": False,
        }

    def _sanitize_extracted_entities(
        self,
        extracted_symptoms: List[dict],
        current_entities: List[dict],
        question_target: Optional[Dict[str, str]],
        patient_input: str,
        last_ai_message: str
    ) -> List[dict]:
        cleaned = []
        brief_direct = self._is_brief_direct_answer(patient_input)

        for ent in extracted_symptoms or []:
            ent = dict(ent or {})
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
            ent.setdefault("slot_answers", {})
            ent.setdefault("is_locked", False)

            if question_target and brief_direct:
                ent["symptom_name"] = question_target["symptom_name"]
                ent["standard_term"] = question_target["standard_term"]

            standard_term = (ent.get("standard_term") or "").strip()
            candidate_slots = self._get_guideline_slots_for_symptom(standard_term) if standard_term else []
            force_slot = question_target["slot"] if (question_target and brief_direct) else None
            ent["dynamic_details"] = self._sanitize_dynamic_details(
                ent.get("dynamic_details", {}),
                candidate_slots=candidate_slots,
                force_slot=force_slot
            )

            cleaned.append(ent)

        if not cleaned and question_target:
            cleaned.append(
                self._build_fallback_slot_update(
                    question_target=question_target,
                    patient_input=patient_input,
                    last_ai_message=last_ai_message
                )
            )

        return cleaned


    def _get_active_entities(self, current_entities: List[dict]) -> List[dict]:
        return [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

    def _find_existing_entity_by_symptom_name(
        self,
        symptom_name: str,
        current_entities: List[dict]
    ) -> Optional[dict]:
        symptom_name = (symptom_name or "").strip()
        if not symptom_name:
            return None

        for ent in current_entities or []:
            if ent.get("status", "active") == "revoked":
                continue
            if (ent.get("symptom_name") or "").strip() == symptom_name:
                return ent
        return None

    def _find_existing_entity_for_revocation(
        self,
        new_ent: dict,
        current_entities: List[dict]
    ) -> Optional[dict]:
        symptom_name = (new_ent.get("symptom_name") or "").strip()
        standard_term = (new_ent.get("standard_term") or "").strip()

        active_entities = [
            ent for ent in (current_entities or [])
            if ent.get("status", "active") != "revoked"
        ]

        if not active_entities:
            return None

        # 1. 先按 symptom_name 精确命中
        for ent in active_entities:
            if symptom_name and (ent.get("symptom_name") or "").strip() == symptom_name:
                return ent

        # 2. 再按 standard_term 精确命中
        for ent in active_entities:
            if standard_term and (ent.get("standard_term") or "").strip() == standard_term:
                return ent

        # 3. 如果 new_ent 只有 symptom_name，没有 standard_term，
        #    但 symptom_name 恰好等于某个已有实体的 standard_term，也应命中
        for ent in active_entities:
            ent_standard = (ent.get("standard_term") or "").strip()
            if symptom_name and ent_standard and symptom_name == ent_standard:
                return ent

        return None


    def _build_mapping_query(self, ent: dict) -> str:
        symptom_name = (ent.get("symptom_name") or "").strip()
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

        return " ".join([x for x in query_parts if str(x).strip()])

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

    def _is_asking_about_additional_symptoms(self, text: str) -> bool:
        if not text:
            return False
        keywords = [
            "除了这些",
            "还有其他不舒服",
            "还有没有其他不舒服",
            "还有别的不舒服",
            "还有其他症状",
            "还有别的症状",
            "还有哪里不舒服",
            "还有没有别的",
            "还有什么不舒服"
        ]
        return any(k in text for k in keywords)

    def _patient_explicitly_declines_more_symptoms(self, text: str) -> bool:
        if not text:
            return False

        normalized = (
            text.strip()
            .replace("，", "")
            .replace("。", "")
            .replace("！", "")
            .replace("？", "")
            .replace(" ", "")
        )

        exact_finish_words = {
            "没有", "没有了", "没了", "没", "无",
            "暂无", "暂时没有",
            "没有其他了", "没有其他不舒服", "没有其他症状",
            "没别的", "没其他",
            "就这些", "就这样", "差不多了"
        }

        return normalized in exact_finish_words

    def _get_history_related_targets(self, current_entities: List[dict], long_term_memory_str: str) -> List[dict]:
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
            long_term_memory_str: str = "",
            qa_trace_str: str = "",
            pending_question_target: Optional[Dict[str, str]] = None
    ) -> List[dict]:
        try:
            current_entities = current_entities or []
            pending_question_target = pending_question_target or {}
            if pending_question_target.get("type") == "guideline_slot":
                forced_update = self._build_forced_slot_update(
                    pending_question_target=pending_question_target,
                    patient_input=patient_input,
                    last_ai_message=last_ai_message
                )
                return [forced_update]

            llm_visible_entities = self._get_agent1_visible_entities(
                current_entities=current_entities,
                pending_question_target=pending_question_target
            )



            if self._is_history_comparison_question(last_ai_message):
                targeted_updates = self._build_history_relation_updates(
                    current_entities=current_entities,
                    long_term_memory_str=long_term_memory_str,
                    patient_input=patient_input
                )
                if targeted_updates:
                    return targeted_updates
                return []

            entities_str = json.dumps(llm_visible_entities, ensure_ascii=False) if llm_visible_entities else "暂无记录"

            # result_dict = self.extract_chain.invoke({
            #     "patient_input": patient_input,
            #     "current_entities": entities_str,
            #     "chat_history_str": chat_history_str,
            #     "long_term_memory_str": long_term_memory_str
            # })
            raw_result = self.extract_chain.invoke({
                "patient_input": patient_input,
                "current_entities": entities_str,
                "last_ai_message": last_ai_message,
                "chat_history_str": chat_history_str,
                "long_term_memory_str": long_term_memory_str,
                "qa_trace_str": qa_trace_str
            })

            if isinstance(raw_result, BaseModel):
                raw_result = raw_result.model_dump()

            if isinstance(raw_result, dict):
                raw_symptoms = raw_result.get("symptoms", [])
            elif isinstance(raw_result, list):
                raw_symptoms = raw_result
            else:
                print(f"[Extractor] 非预期抽取结果类型: {type(raw_result).__name__} | raw_result={raw_result}")
                raw_symptoms = []

            if isinstance(raw_symptoms, dict):
                raw_symptoms = [raw_symptoms]
            elif not isinstance(raw_symptoms, list):
                raw_symptoms = []

            question_target = self._infer_question_target(last_ai_message, current_entities)

            extracted_symptoms = self._sanitize_extracted_entities(
                extracted_symptoms=raw_symptoms,
                current_entities=current_entities,
                question_target=question_target,
                patient_input=patient_input,
                last_ai_message=last_ai_message
            )

            # if pending_question_target.get("type") == "guideline_slot":
            #     extracted_symptoms.append(
            #         self._build_forced_slot_update(
            #             pending_question_target=pending_question_target,
            #             patient_input=patient_input,
            #             last_ai_message=last_ai_message
            #         )
            #     )


            if self.debug:
                print("\n" + "#" * 80)
                print(f"[Extractor] 患者输入: {patient_input}")
                print(f"[Extractor] LLM抽取结果(映射前): {json.dumps(extracted_symptoms, ensure_ascii=False)}")

            for idx, ent in enumerate(extracted_symptoms, start=1):
                status = ent.get("status", "active")
                symptom_name = (ent.get("symptom_name") or "").strip()

                if not symptom_name:
                    if self.debug:
                        print(
                            f"[Extractor] 症状{idx} 跳过映射 | "
                            f"symptom_name={symptom_name} | status={status}"
                        )
                    continue

                # 对撤销实体，优先尝试命中已有活动实体，复用它的 standard_term 和 symptom_name
                if status == "revoked":
                    existing_ent = self._find_existing_entity_for_revocation(
                        new_ent=ent,
                        current_entities=current_entities
                    )

                    if existing_ent:
                        ent["symptom_name"] = existing_ent.get("symptom_name")
                        ent["standard_term"] = existing_ent.get("standard_term")

                        if self.debug:
                            print(
                                f"[Extractor] 症状{idx} 撤销命中已有实体 | "
                                f"symptom_name={ent['symptom_name']} | "
                                f"standard_term={ent['standard_term']}"
                            )
                        continue

                    # 如果没命中已有实体，再尝试做一次标准术语映射，避免后续 merge 失败
                    query = self._build_mapping_query(ent)
                    standard_term = self.retriever.get_standard_term(
                        query=query,
                        top_k=5,
                        raw_top_k=30,
                        use_rerank=True
                    )
                    ent["standard_term"] = standard_term

                    if self.debug:
                        print(
                            f"[Extractor] 症状{idx} 撤销实体映射完成 | "
                            f"symptom_name={symptom_name} | standard_term={standard_term}"
                        )
                    continue

                existing_ent = self._find_existing_entity_by_symptom_name(
                    symptom_name=symptom_name,
                    current_entities=current_entities
                )

                if existing_ent and str(existing_ent.get("standard_term", "")).strip():
                    ent["standard_term"] = existing_ent.get("standard_term")
                    if self.debug:
                        print(
                            f"[Extractor] 症状{idx} 复用既有术语，不重新映射 | "
                            f"symptom_name={symptom_name} | "
                            f"standard_term={ent['standard_term']}"
                        )
                    continue

                query = self._build_mapping_query(ent)

                if self.debug:
                    print(
                        f"[Extractor] 症状{idx} 首次出现，开始映射 | "
                        f"symptom_name={symptom_name} | status={status} | query={query}"
                    )

                standard_term = self.retriever.get_standard_term(
                    query=query,
                    top_k=5,
                    raw_top_k=30,
                    use_rerank=True
                )
                ent["standard_term"] = standard_term

                if self.debug:
                    print(
                        f"[Extractor] 症状{idx} 映射完成 | "
                        f"symptom_name={symptom_name} | standard_term={standard_term}"
                    )

                existing_ent = self._find_existing_entity_by_symptom_name(
                    symptom_name=symptom_name,
                    current_entities=current_entities
                )

                if existing_ent and str(existing_ent.get("standard_term", "")).strip():
                    ent["standard_term"] = existing_ent.get("standard_term")
                    if self.debug:
                        print(
                            f"[Extractor] 症状{idx} 复用既有术语，不重新映射 | "
                            f"symptom_name={symptom_name} | "
                            f"standard_term={ent['standard_term']}"
                        )
                    continue

                query = self._build_mapping_query(ent)

                if self.debug:
                    print(
                        f"[Extractor] 症状{idx} 首次出现，开始映射 | "
                        f"symptom_name={symptom_name} | status={status} | query={query}"
                    )

                standard_term = self.retriever.get_standard_term(
                    query=query,
                    top_k=5,
                    raw_top_k=30,
                    use_rerank=True
                )
                ent["standard_term"] = standard_term

                if self.debug:
                    print(
                        f"[Extractor] 症状{idx} 映射完成 | "
                        f"symptom_name={symptom_name} | standard_term={standard_term}"
                    )

            if self.debug:
                print(f"[Extractor] 最终结果(映射后): {json.dumps(extracted_symptoms, ensure_ascii=False)}")
                print("#" * 80 + "\n")

            return extracted_symptoms

        except Exception as e:
            print(f"提取出错: {e}")
            return []

    # =========================================================
    # 模块 B + C：规划与生成回复（合并版）
    # =========================================================
    def generate_reply(
            self,
            patient_input: str,
            current_entities: List[dict],
            last_ai_message: str = "",
            chat_history_str: str = "",
            long_term_memory_str: str = "",
            qa_trace_str: str = ""
    ) -> str:
        try:
            active_entities = self._get_active_entities(current_entities)

            if not active_entities:
                return "您好，系统似乎没有捕捉到您的不适症状，能请您再具体描述一下哪里不舒服吗？"

            # 1. 先用代码锁定“下一条唯一该问的 guideline 要点”
            next_target = self._pick_next_guideline_target(active_entities)
            if next_target:
                question = self._generate_single_slot_question(
                    target_symptom=next_target["symptom"],
                    target_slot=next_target["slot"],
                    allowed_keywords=next_target["allowed_keywords"],
                    last_ai_message=last_ai_message,
                    chat_history_str=chat_history_str
                )

                if self.debug:
                    print(
                        f"\n[Single Slot Question] symptom={next_target['standard_term']} "
                        f"| slot={next_target['slot']} | question={question}\n"
                    )

                return question

            # 2. 到这里说明：当前所有 guideline 要点已经问完
            #    这时绝对不能再把控制权交回 dialogue_chain，否则模型会自由发挥补问
            history_targets = self._get_history_related_targets(active_entities, long_term_memory_str)

            # 2.1 如果还有相关既往史没比较完，就只做一次宽泛对比追问
            if history_targets and not self._all_history_targets_completed(active_entities, long_term_memory_str):
                return "我看到您之前也有过类似不适，我再确认一下，这次和之前比有没有更重一些，或者有没有新出现的不舒服？"

            # 2.2 如果上一轮已经在做“还有没有其他症状”的收尾确认
            if self._is_asking_about_additional_symptoms(last_ai_message):
                if self._patient_explicitly_declines_more_symptoms(patient_input):
                    return "好的，您的病情信息已收集完毕，系统正在为您生成病历草稿。"
                else:
                    # 患者说还有，但本轮没形成新的有效症状实体时，用一个宽泛承接
                    return "好的，您再具体说一下还有什么不舒服，好吗？"

            # 2.3 如果所有当前症状都已问完，且既往史比较也完成/无需比较
            #     下一步只能进入“新增症状确认”，不能再问当前症状细节
            return "除了这些，您还有其他不舒服的地方吗？"

        except Exception as e:
            print(f"生成回复出错: {e}")
            return "不好意思，系统开了个小差，您能再说一遍吗？"


