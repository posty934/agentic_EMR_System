import re
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from core.graph import EMRState
from langchain_core.messages import AIMessage, HumanMessage

from agents.agent1_extractor import Agent1Extractor
from agents.agent2_generator import Agent2Generator
from agents.agent3_reviewer import Agent3Reviewer


class EMRWorkflow:
    def __init__(self):
        self.agent1 = Agent1Extractor()
        self.agent2 = Agent2Generator()
        self.agent3 = Agent3Reviewer()
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(EMRState)

        workflow.add_node("Agent1_Extractor", self.node_extractor)
        workflow.add_node("Agent2_Generator", self.node_generator)
        workflow.add_node("Agent2_Refiner", self.node_refiner)
        workflow.add_node("Agent3_Reviewer", self.node_reviewer)

        workflow.set_entry_point("Agent1_Extractor")

        workflow.add_conditional_edges(
            "Agent1_Extractor",
            self.route_after_extractor,
            {
                "continue_asking": END,
                "generate_record": "Agent2_Generator"
            }
        )

        workflow.add_edge("Agent2_Generator", "Agent3_Reviewer")
        workflow.add_edge("Agent2_Refiner", "Agent3_Reviewer")

        workflow.add_conditional_edges(
            "Agent3_Reviewer",
            self.route_after_reviewer,
            {
                "pass": END,
                "auto_repair": "Agent2_Refiner",
                "rollback": END
            }
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    # =========================
    # 文本合并工具函数
    # =========================
    def _normalize_text(self, value) -> str:
        if value is None:
            return ""

        text = str(value).strip()

        # 统一空白
        text = re.sub(r"\s+", "", text)

        # 统一常见分隔符
        text = (
            text.replace("、", "，")
            .replace(",", "，")
            .replace("；", "，")
            .replace(";", "，")
        )

        # 压缩重复逗号
        text = re.sub(r"，+", "，", text).strip("，")
        return text

    def _split_clauses(self, value) -> list[str]:
        text = self._normalize_text(value)
        if not text:
            return []

        # 适度按中文常见并列分隔切分
        parts = re.split(r"[，。/]+", text)
        return [p.strip() for p in parts if p and p.strip()]

    def _merge_text_field(self, old_val, new_val) -> str:
        """
        合并文本字段，避免：
        1. 完全重复
        2. 新值包含旧值时再次整段拼接
        3. 旧值包含新值时再次拼接子串
        4. 枚举项重复
        """
        old_text = str(old_val).strip() if old_val else ""
        new_text = str(new_val).strip() if new_val else ""

        if not old_text:
            return new_text
        if not new_text:
            return old_text

        old_norm = self._normalize_text(old_text)
        new_norm = self._normalize_text(new_text)

        # 完全一样
        if old_norm == new_norm:
            return old_text

        # 新值是“更完整版本” -> 直接覆盖
        if old_norm and old_norm in new_norm:
            return new_text

        # 新值只是旧值的一部分 -> 保留旧值
        if new_norm and new_norm in old_norm:
            return old_text

        # 否则按短语去重合并
        merged = []
        for item in self._split_clauses(old_text) + self._split_clauses(new_text):
            if item and item not in merged:
                merged.append(item)

        return "，".join(merged) if merged else new_text

    def _merge_dynamic_details(self, old_details: dict, new_details: dict) -> dict:
        result = dict(old_details or {})
        for dk, dv in (new_details or {}).items():
            if not dv:
                continue
            result[dk] = self._merge_text_field(result.get(dk), dv)
        return result
    def _normalize_qa_text(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[，。！？；：、“”\"'（）()【】\[\]<>《》]", "", text)
        return text.lower()

    def _looks_like_question(self, text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False

        if "？" in text or "?" in text:
            return True

        question_keywords = [
            "吗", "么", "是否", "有无", "有没有", "会不会", "是不是",
            "能不能", "还有没有", "哪里", "多久", "什么时候",
            "多长时间", "几次", "什么样", "有没有加重", "有没有缓解"
        ]
        return any(k in text for k in question_keywords)

    def _build_qa_trace(self, messages, max_pairs: int = 4) -> str:
        pairs = []

        for i in range(len(messages) - 1):
            q_msg = messages[i]
            a_msg = messages[i + 1]

            if not isinstance(q_msg, AIMessage) or not isinstance(a_msg, HumanMessage):
                continue

            q = str(q_msg.content or "").strip()
            a = str(a_msg.content or "").strip()

            if not q or not a or not self._looks_like_question(q):
                continue

            pairs.append((q, a))

        if not pairs:
            return "暂无已完成问答轨迹。"

        latest_by_fp = {}
        order = []

        for q, a in pairs:
            fp = self._normalize_qa_text(q)
            if fp not in latest_by_fp:
                order.append(fp)
            latest_by_fp[fp] = (q, a)

        lines = []
        for idx, fp in enumerate(order[-max_pairs:], start=1):
            q, a = latest_by_fp[fp]
            lines.append(f"{idx}. AI问：{q}")
            lines.append(f"   患者答：{a}")
            lines.append("   结论：该问点已经出现过明确患者回答，除非后续信息冲突，否则禁止同义重复追问。")

        return "\n".join(lines)

    def _find_existing_entity_for_merge(self, new_ent: dict, updated_entities: list):
        sym_name = (new_ent.get("symptom_name") or "").strip()
        standard_term = (new_ent.get("standard_term") or "").strip()
        status = (new_ent.get("status") or "active").strip()

        active_entities = [
            ent for ent in updated_entities
            if ent.get("status", "active") != "revoked"
        ]

        # 1. 先按 symptom_name 精确匹配
        for ent in active_entities:
            if sym_name and (ent.get("symptom_name") or "").strip() == sym_name:
                return ent

        # 2. 再按 standard_term 精确匹配
        if standard_term and standard_term != "未知术语":
            same_term_entities = [
                ent for ent in active_entities
                if (ent.get("standard_term") or "").strip() == standard_term
            ]
            if len(same_term_entities) == 1:
                return same_term_entities[0]

        # 3. 对撤销实体额外放宽：
        #    如果它的 symptom_name 恰好就是已有活动实体的 standard_term，也认为是同一个症状
        if status == "revoked":
            for ent in active_entities:
                ent_standard = (ent.get("standard_term") or "").strip()
                if sym_name and ent_standard and sym_name == ent_standard:
                    return ent

        return None

    def node_extractor(self, state: EMRState):
        messages = state.get("messages", [])
        entities = state.get("entities", [])
        pending_question_target = state.get("pending_question_target", {})

        long_term_memory_str = state.get("long_term_memory", "无既往病史记录。")

        # 滑动窗口机制
        window_size = 4
        recent_messages = messages[-window_size:] if len(messages) >= window_size else messages
        chat_history_str = "\n".join([
            f"{'患者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent_messages
        ])
        qa_trace_str = self._build_qa_trace(messages)

        last_human_msg = messages[-1].content if messages else ""
        last_ai_msg = ""
        if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
            last_ai_msg = messages[-2].content

        extracted_data = self.agent1.extract(
            last_human_msg,
            entities,
            last_ai_msg,
            chat_history_str,
            long_term_memory_str,
            qa_trace_str,
            pending_question_target
        )

        updated_entities = list(entities)

        if extracted_data:
            mergeable_fields = [
                "onset_time",
                "characteristics",
                "inducement",
                "frequency",
                "alleviating_factors",
                "location",
                "duration_pattern",
                "severity",
                "associated_symptoms",
                "negative_symptoms",
                "relation_to_food",
                "relation_to_bowel",
                "relation_to_position",
                "progression",
            ]

            for new_ent in extracted_data:
                sym_name = (new_ent.get("symptom_name") or "").strip()
                if not sym_name:
                    continue

                existing = self._find_existing_entity_for_merge(new_ent, updated_entities)

                if existing:
                    if new_ent.get("status") == "revoked":
                        existing["status"] = "revoked"
                        existing["is_locked"] = False
                        continue

                    if existing.get("is_locked", False):
                        continue

                    existing.setdefault("status", "active")
                    existing.setdefault("slot_answers", {})
                    existing.setdefault("slot_display_answers", {})
                    existing.setdefault("dynamic_details", {})

                    if not existing.get("standard_term") and new_ent.get("standard_term"):
                        existing["standard_term"] = new_ent.get("standard_term")

                    for k in mergeable_fields:
                        new_val = new_ent.get(k)
                        if not new_val:
                            continue
                        existing[k] = self._merge_text_field(existing.get(k), new_val)

                    existing["dynamic_details"] = self._merge_dynamic_details(
                        existing.get("dynamic_details", {}),
                        new_ent.get("dynamic_details", {})
                    )

                    for slot, answer in (new_ent.get("slot_answers", {}) or {}).items():
                        if not answer:
                            continue
                        existing["slot_answers"][slot] = self._merge_text_field(
                            existing["slot_answers"].get(slot),
                            answer
                        )
                    for slot, display_answer in (new_ent.get("slot_display_answers", {}) or {}).items():
                        if not display_answer:
                            continue
                        existing["slot_display_answers"][slot] = str(display_answer).strip()

                else:
                    new_ent.setdefault("status", "active")
                    new_ent.setdefault("dynamic_details", {})
                    new_ent.setdefault("slot_answers", {})
                    new_ent.setdefault("slot_display_answers", {})
                    new_ent.setdefault("is_locked", False)
                    updated_entities.append(new_ent)

        updated_entities = self.agent1.refresh_entities_slot_state(updated_entities)

        turn_plan = self.agent1.plan_next_turn(
            last_human_msg,
            updated_entities,
            last_ai_msg,
            chat_history_str,
            long_term_memory_str,
            qa_trace_str
        )

        updated_entities = turn_plan["entities"]
        reply = turn_plan["reply"]
        next_pending_question_target = turn_plan.get("pending_question_target", {})

        is_finished = "病情信息已收集完毕" in reply

        return {
            "messages": [AIMessage(content=reply)],
            "entities": updated_entities,
            "missing_slots": [],
            "is_finished": is_finished,
            "revision_count": 0,
            "max_revision_count": state.get("max_revision_count", 2),
            "auto_revision_possible": False,
            "need_user_input": False,
            "feedback": "",
            "rollback_question": "",
            "repair_instruction": "",
            "pending_question_target": next_pending_question_target
        }

    def node_generator(self, state: EMRState):
        entities = state.get("entities", [])
        draft = self.agent2.generate_record(entities)
        return {"draft_record": draft}

    def node_refiner(self, state: EMRState):
        entities = state.get("entities", [])
        draft = state.get("draft_record", {})
        repair_instruction = state.get("repair_instruction", "") or state.get("feedback", "")
        current_revision = state.get("revision_count", 0)

        revised_draft = self.agent2.revise_record(
            current_entities=entities,
            draft_record=draft,
            repair_instruction=repair_instruction
        )

        return {
            "draft_record": revised_draft,
            "revision_count": current_revision + 1
        }

    def node_reviewer(self, state: EMRState):
        draft = state.get("draft_record", {})
        entities = state.get("entities", [])
        current_revision = state.get("revision_count", 0)
        max_revision = state.get("max_revision_count", 2)

        validation = self.agent3.validate(draft, entities)

        raw_auto_revision_possible = validation.get("auto_revision_possible", False)
        auto_revision_possible = raw_auto_revision_possible and (current_revision < max_revision)
        need_user_input = validation.get("need_user_input", False) or (
            raw_auto_revision_possible and current_revision >= max_revision
        )

        new_messages = []
        is_finished_state = True

        if not validation["final_pass"]:
            if auto_revision_possible:
                is_finished_state = False
            else:
                if raw_auto_revision_possible and current_revision >= max_revision:
                    warning_text = (
                        f"🚨 **【自动修正未通过，已转人工追问】** 🚨\n\n"
                        f"- 已自动修正次数: {current_revision}/{max_revision}\n"
                        f"- 图谱校验: {'通过' if validation['kg_pass'] else '未通过'}\n"
                        f"- LLM复核: {'通过' if validation['llm_pass'] else '未通过'}\n\n"
                        f"**问题摘要：** {validation.get('feedback')}"
                    )
                else:
                    warning_text = (
                        f"🚨 **【双重质控拦截】** 🚨\n\n"
                        f"- 图谱校验: {'通过' if validation['kg_pass'] else '未通过'}\n"
                        f"- LLM复核: {'通过' if validation['llm_pass'] else '未通过'}\n\n"
                        f"**问题摘要：** {validation.get('feedback')}"
                    )

                new_messages.append(AIMessage(content=warning_text))
                new_messages.append(AIMessage(content=validation["rollback_question"]))
                is_finished_state = False

        return {
            "is_valid": validation["final_pass"],
            "feedback": validation["feedback"],
            "rollback_question": validation["rollback_question"],
            "repair_instruction": validation.get("repair_instruction", ""),
            "auto_revision_possible": auto_revision_possible,
            "need_user_input": need_user_input,
            "messages": new_messages,
            "is_finished": is_finished_state,
            "revision_count": current_revision,
            "max_revision_count": max_revision
        }

    def route_after_extractor(self, state: EMRState):
        if state.get("is_finished") is True:
            return "generate_record"
        return "continue_asking"

    def route_after_reviewer(self, state: EMRState):
        if state.get("is_valid") is True:
            return "pass"
        if state.get("auto_revision_possible") is True:
            return "auto_repair"
        return "rollback"