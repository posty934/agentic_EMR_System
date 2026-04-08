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

    def node_extractor(self, state: EMRState):
        messages = state.get("messages", [])
        entities = state.get("entities", [])

        long_term_memory_str = state.get("long_term_memory", "无既往病史记录。")

        # 滑动窗口机制
        window_size = 4
        recent_messages = messages[-window_size:] if len(messages) >= window_size else messages
        chat_history_str = "\n".join([
            f"{'患者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent_messages
        ])

        last_human_msg = messages[-1].content if messages else ""
        last_ai_msg = ""
        if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
            last_ai_msg = messages[-2].content

        extracted_data = self.agent1.extract(
            last_human_msg,
            entities,
            last_ai_msg,
            chat_history_str,
            long_term_memory_str
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

                existing = next(
                    (e for e in updated_entities if (e.get("symptom_name") or "").strip() == sym_name),
                    None
                )

                if existing:
                    if new_ent.get("status") == "revoked":
                        existing["status"] = "revoked"
                        continue

                    existing.setdefault("status", "active")

                    # standard_term 只在原来没有时补，不拼接
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
                else:
                    new_ent.setdefault("status", "active")
                    new_ent.setdefault("dynamic_details", {})
                    updated_entities.append(new_ent)

        reply = self.agent1.generate_reply(
            last_human_msg,
            updated_entities,
            last_ai_msg,
            chat_history_str,
            long_term_memory_str
        )

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
            "repair_instruction": ""
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