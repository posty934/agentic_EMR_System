import os
import json
import re
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
        self.llm = create_llm(temperature=0.2)
        self.parser = JsonOutputParser(pydantic_object=MedicalRecordDraft)

        self.field_label_map = {
            "onset_time": "时间",
            "location": "部位/范围",
            "characteristics": "性质/表现",
            "duration_pattern": "持续/规律",
            "severity": "程度/影响",
            "frequency": "频率/次数",
            "inducement": "诱因",
            "alleviating_factors": "缓解/加重",
            "relation_to_food": "与进食关系",
            "relation_to_bowel": "与排便关系",
            "relation_to_position": "与体位关系",
            "associated_symptoms": "伴随症状",
            "negative_symptoms": "否认症状",
            "progression": "变化趋势",
        }

        self.audit_priority_fields = {
            "severity",
            "progression",
            "associated_symptoms",
            "negative_symptoms",
            "inducement",
            "alleviating_factors",
            "relation_to_food",
            "relation_to_bowel",
            "relation_to_position",
            "dynamic_details",
            "slot_answers",
        }

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的临床住院医师。
你的任务是根据智能问诊系统收集到的【患者结构化症状数据】和【必须覆盖事实清单】，撰写一份专业的门诊病历草稿（包含主诉和现病史）。

【患者结构化症状数据 JSON】
{patient_data}

【必须覆盖事实清单】
{fact_checklist}

【撰写要求】
1. 主诉 (Chief Complaint)：
   - 提取最主要的 1-3 个标准术语及最长发病时间，尽量控制在20字以内。
   - 主诉只需简练概括，不必塞入全部细节。

2. 现病史 (History of Present Illness)：
   - 必须且只能基于传入的【结构化数据】撰写，严禁脑补、编造。
   - 必须覆盖【必须覆盖事实清单】中的全部非空事实；任何已确认事实都不能遗漏。
   - 将患者的“症状名”优先替换为“标准医学术语 (standard_term)”后再行文。
   - 对于程度/影响、诱因、伴随症状、否认症状、与进食/排便/体位关系、变化趋势、dynamic_details、slot_answers 中的已确认信息，原则上都要写入现病史。
   - 对于结构化数据中的口语化表达，必须转写为客观、书面、专业的医学表述；禁止直接照抄患者口语原句。
   - 禁止出现“患者自述‘快拉虚脱了’”“自述可能因‘辣到了’诱发”这类带引号或明显口语色彩的写法。
   - 例如：“快拉虚脱了”应改写为“症状较重，伴明显虚弱感”；“辣到了”应改写为“发病前有进食辛辣刺激食物史”。


3. 行文要求：
   - 客观、准确、符合医疗文书规范。
   - 可自然整合，但不能因为追求文风而丢失事实。
   - 可以将相近信息合并表达，但不能漏掉任何一个已确认点。
   - 不要生成诊断结论、检查结论、治疗建议或处方内容。
    - 口语化转写时必须保留患者原始回答中的程度、数量、频率、时间、主观性限定词。
    - “有点/一点点/轻微”可以写为“轻微/稍有”，但不能省略为单纯“有”。
    - “感觉/觉得/好像/疑似/不确定”必须保留为“自觉/感觉/诉”等主观表达，不能写成完全确定事实。
    - “一些/少量/一点”必须保留为“少量/部分/一些”，不能省略。
    - 对 slot_answers 中的短答，允许医学化，但不得改变或丢失程度。例如“有点”应写为“稍有/轻微”，不能写成未限定程度的“有”。


4. 严格遵循以下输出格式：
{format_instructions}"""),
            ("human", "请开始撰写病历。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

        self.repair_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的临床住院医师，现在需要根据质控反馈，对病历草稿做“内部自动修正”。

【当前有效结构化症状数据 JSON】
{patient_data}

【必须覆盖事实清单】
{fact_checklist}

【上一版病历草稿】
{draft_data}

【质控反馈 / 修正指令】
{repair_instruction}

【修正规则】
1. 只能基于【当前有效结构化症状数据】修改，严禁新增不存在的信息。
2. 删除与被撤销症状有关的内容，删除与实体不一致、与时间不一致、与症状特征不一致的内容。
3. 必须把【必须覆盖事实清单】中的遗漏事实补回现病史。
4. 对于程度/影响、伴随症状、否认症状、诱因、关系类字段、dynamic_details、slot_answers，只要结构化数据中有值，就不能遗漏。
5. 所有口语化、非医学书面表达都必须改写为专业表述，禁止直接保留引号内原话。
6. 若原草稿里有正确内容，应尽量保留。
7.
- 口语化转写时必须保留患者原始回答中的程度、数量、频率、时间、主观性限定词。
- “有点/一点点/轻微”可以写为“轻微/稍有”，但不能省略为单纯“有”。
- “感觉/觉得/好像/疑似/不确定”必须保留为“自觉/感觉/诉”等主观表达，不能写成完全确定事实。
- “一些/少量/一点”必须保留为“少量/部分/一些”，不能省略。
- 对 slot_answers 中的短答，允许医学化，但不得改变或丢失程度。例如“有点”应写为“稍有/轻微”，不能写成未限定程度的“有”。

8. 不要生成诊断、检查结果、治疗方案或处方。
9. 输出仍然只包含：主诉 + 现病史。
10. 严格按以下 JSON 格式输出：
{format_instructions}"""),
            ("human", "请直接给出修正后的病历草稿。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.repair_chain = self.repair_prompt | self.llm | self.parser

    def _get_active_entities(self, current_entities: list) -> list:
        if not current_entities:
            return []
        return [ent for ent in current_entities if ent.get("status", "active") != "revoked"]

    def _normalize_text(self, text) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text)
        return text.lower()

    def _split_clauses(self, text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        parts = re.split(r"[，。；;、/]+", raw)
        return [p.strip() for p in parts if p and p.strip()]

    def _professionalize_fact_text(self, symptom: str, field: str, label: str, content: str) -> str:
        text = str(content or "").strip()
        if not text:
            return text

        replacements = [
            (r"快拉虚脱了|拉虚脱了|快虚脱了", "症状较重，伴明显虚弱感"),
            (r"辣到了|吃辣后|吃火锅辣到了|前几天吃火锅辣到了", "发病前有进食辛辣刺激食物史"),
            (r"吃了不干净的东西", "发病前有进食不洁饮食史"),
            (r"拉不出来", "排便困难"),
            (r"总感觉没排干净", "排便不尽感"),
            (r"拉完还想拉", "里急后重"),
            (r"总想吐", "恶心明显"),
            (r"吐得厉害", "呕吐频繁"),
            (r"疼得直不起腰", "疼痛剧烈"),
            (r"完全吃不下|一点都吃不下", "无法进食"),
            (r"眼前发黑", "黑矇"),
            (r"心慌", "心悸"),
            (r"有这种感觉", "有"),
            (r"没什么变化", "无明显变化"),
        ]

        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)

        if field == "severity":
            if symptom == "腹泻" and "明显虚弱感" in text:
                text = "腹泻程度较重，伴明显虚弱感"
            elif symptom == "呕吐" and "明显虚弱感" in text:
                text = "呕吐较重，伴明显虚弱感"

        if field == "inducement":
            if "辛辣刺激食物" in text and "发病前" not in text:
                text = f"发病前{text}"
            if "不洁饮食" in text and "发病前" not in text:
                text = f"发病前{text}"

        text = text.strip("，。；; ")
        return text

    def _build_strict_entity_payload(self, ent: dict) -> dict:
        symptom_name = (ent.get("symptom_name") or "").strip()
        standard_term = (ent.get("standard_term") or symptom_name).strip()

        strict_slot_answers = {}
        slot_answers = ent.get("slot_answers", {})
        if isinstance(slot_answers, dict):
            for slot, answer in slot_answers.items():
                slot = str(slot).strip()
                answer = str(answer).strip()
                if slot and answer:
                    strict_slot_answers[slot] = answer

        strict_dynamic_details = {}
        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            history_relation = str(dynamic_details.get("与过往病史关联", "")).strip()
            if history_relation:
                strict_dynamic_details["与过往病史关联"] = history_relation

        return {
            "symptom_name": symptom_name,
            "standard_term": standard_term,
            "status": ent.get("status", "active"),
            "slot_answers": strict_slot_answers,
            "dynamic_details": strict_dynamic_details,
        }

    def _iter_fact_items(self, ent: dict) -> list[dict]:
        facts = []
        seen = set()

        symptom = (ent.get("standard_term") or ent.get("symptom_name") or "未知症状").strip()

        slot_answers = ent.get("slot_answers", {})
        if isinstance(slot_answers, dict):
            for slot, answer in slot_answers.items():
                slot = str(slot).strip()
                answer = str(answer).strip()
                if not slot or not answer:
                    continue

                key = (
                    symptom,
                    f"slot::{slot}",
                    self._normalize_text(answer),
                )
                if key in seen:
                    continue
                seen.add(key)

                facts.append({
                    "symptom": symptom,
                    "field": "slot_answers",
                    "label": f"问诊要点:{slot}",
                    "content": self._professionalize_fact_text(
                        symptom,
                        "slot_answers",
                        slot,
                        answer
                    ),
                    "audit_priority": True,
                })

        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            history_relation = str(dynamic_details.get("与过往病史关联", "")).strip()
            if history_relation:
                key = (
                    symptom,
                    "dynamic::与过往病史关联",
                    self._normalize_text(history_relation),
                )
                if key not in seen:
                    facts.append({
                        "symptom": symptom,
                        "field": "dynamic_details",
                        "label": "与过往病史关联",
                        "content": self._professionalize_fact_text(
                            symptom,
                            "dynamic_details",
                            "与过往病史关联",
                            history_relation
                        ),
                        "audit_priority": True,
                    })

        return facts

    def _build_fact_checklist(self, active_entities: list) -> str:
        if not active_entities:
            return "无"

        lines = []
        for idx, ent in enumerate(active_entities, start=1):
            symptom_name = (ent.get("symptom_name") or "").strip()
            standard_term = (ent.get("standard_term") or symptom_name or "未知术语").strip()
            lines.append(f"症状{idx}: 原始症状名={symptom_name or '未提供'}；标准术语={standard_term or '未映射'}")

            facts = self._iter_fact_items(ent)
            if not facts:
                lines.append("- 无可用细节")
                continue

            for fact in facts:
                lines.append(f"- [{fact['symptom']}] {fact['label']}: {fact['content']}")

        return "\n".join(lines)

    def _build_patient_payload(self, active_entities: list) -> tuple[str, str]:
        strict_entities = [
            self._build_strict_entity_payload(ent)
            for ent in active_entities
            if ent.get("status", "active") != "revoked"
        ]
        data_str = json.dumps(strict_entities, ensure_ascii=False, indent=2)
        fact_checklist = self._build_fact_checklist(strict_entities)
        return data_str, fact_checklist

    def _fact_is_covered(self, fact: dict, draft_text: str) -> bool:
        draft_norm = self._normalize_text(draft_text)
        content = str(fact.get("content") or "").strip()
        if not content:
            return True

        content_norm = self._normalize_text(content)
        if content_norm and content_norm in draft_norm:
            return True

        clauses = self._split_clauses(content)
        clause_norms = [self._normalize_text(x) for x in clauses if len(self._normalize_text(x)) >= 2]

        if clause_norms and all(c in draft_norm for c in clause_norms):
            return True

        return False

    def _find_missing_facts(self, active_entities: list, draft_record: dict) -> list[dict]:
        draft_text = (
            f"{draft_record.get('chief_complaint', '')} "
            f"{draft_record.get('history_of_present_illness', '')}"
        ).strip()

        missing = []
        for ent in active_entities:
            for fact in self._iter_fact_items(ent):
                if self._fact_is_covered(fact, draft_text):
                    continue

                # 所有字段都喂给 LLM，但优先拦截高价值字段遗漏
                if fact["audit_priority"]:
                    missing.append(fact)

        return missing

    def _enforce_fact_coverage(self, active_entities: list, draft_record: dict) -> dict:
        if not active_entities or not draft_record:
            return draft_record

        missing = self._find_missing_facts(active_entities, draft_record)
        if not missing:
            return draft_record

        patient_data_str, fact_checklist = self._build_patient_payload(active_entities)
        draft_str = json.dumps(draft_record, ensure_ascii=False, indent=2)

        missing_lines = [
            f"- [{fact['symptom']}] {fact['label']}: {fact['content']}"
            for fact in missing
        ]
        repair_instruction = (
            "上一版草稿遗漏了以下已确认结构化事实，请在不新增任何无依据信息的前提下，"
            "把这些遗漏点自然、完整地补回现病史；尤其不要遗漏严重程度、伴随症状、否认症状、诱因、关系类字段和动态问诊细节。\n"
            "遗漏事实如下：\n" +
            "\n".join(missing_lines)
        )

        try:
            repaired = self.repair_chain.invoke({
                "patient_data": patient_data_str,
                "fact_checklist": fact_checklist,
                "draft_data": draft_str,
                "repair_instruction": repair_instruction
            })
            return repaired
        except Exception as e:
            print(f"病历覆盖性修正出错: {e}")
            return draft_record

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

            patient_data_str, fact_checklist = self._build_patient_payload(active_entities)

            result = self.chain.invoke({
                "patient_data": patient_data_str,
                "fact_checklist": fact_checklist
            })

            result = self._enforce_fact_coverage(active_entities, result)
            return result

        except Exception as e:
            print(f"病历生成出错: {e}")
            return {
                "chief_complaint": "生成失败",
                "history_of_present_illness": "系统生成病历时发生错误，请重试。"
            }

    def revise_record(self, current_entities: list, draft_record: dict, repair_instruction: str) -> dict:
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

            patient_data_str, fact_checklist = self._build_patient_payload(active_entities)
            draft_str = json.dumps(draft_record, ensure_ascii=False, indent=2)
            instruction = repair_instruction.strip() if repair_instruction else "请严格基于现有实体重写病历草稿，删除不一致内容，补回遗漏事实，保留正确内容。"

            result = self.repair_chain.invoke({
                "patient_data": patient_data_str,
                "fact_checklist": fact_checklist,
                "draft_data": draft_str,
                "repair_instruction": instruction
            })

            result = self._enforce_fact_coverage(active_entities, result)
            return result

        except Exception as e:
            print(f"病历自动修正出错: {e}")
            return draft_record if draft_record else {
                "chief_complaint": "生成失败",
                "history_of_present_illness": "系统修正病历时发生错误，请重试。"
            }
