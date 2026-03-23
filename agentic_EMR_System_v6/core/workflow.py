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

        # 质控回滚时，必须走到 END 等待用户回答
        workflow.add_conditional_edges(
            "Agent3_Reviewer",
            self.route_after_reviewer,
            {
                "pass": END,
                "rollback": END
            }
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def node_extractor(self, state: EMRState):
        messages = state.get("messages", [])
        entities = state.get("entities", [])

        # 提取患者向量记忆
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

        # 🚨 注意：给 extract 传入 long_term_memory_str
        extracted_data = self.agent1.extract(last_human_msg, entities, last_ai_msg, chat_history_str,
                                             long_term_memory_str)

        updated_entities = list(entities)
        if extracted_data:
            for new_ent in extracted_data:
                sym_name = new_ent.get("symptom_name")
                if not sym_name: continue

                existing = next((e for e in updated_entities if e.get("symptom_name") == sym_name), None)
                if existing:
                    if new_ent.get("status") == "revoked":
                        existing["status"] = "revoked"
                        continue
                    for k in ["onset_time", "characteristics", "inducement", "frequency", "alleviating_factors"]:
                        if new_ent.get(k):
                            if not existing.get(k):
                                existing[k] = new_ent.get(k)
                            elif new_ent.get(k) not in existing.get(k):
                                existing[k] = f"{existing[k]}，{new_ent.get(k)}"
                    if new_ent.get("dynamic_details"):
                        if not existing.get("dynamic_details"):
                            existing["dynamic_details"] = {}
                        for dk, dv in new_ent.get("dynamic_details").items():
                            if dk not in existing["dynamic_details"]:
                                existing["dynamic_details"][dk] = dv
                            elif dv not in existing["dynamic_details"][dk]:
                                existing["dynamic_details"][dk] = f"{existing['dynamic_details'][dk]}，{dv}"
                else:
                    updated_entities.append(new_ent)

        # 🚨 注意：给 generate_reply 传入 long_term_memory_str
        reply = self.agent1.generate_reply(last_human_msg, updated_entities, last_ai_msg, chat_history_str,
                                           long_term_memory_str)
        is_finished = "病情信息已收集完毕" in reply

        return {
            "messages": [AIMessage(content=reply)],
            "entities": updated_entities,
            "missing_slots": [],
            "is_finished": is_finished
        }

    def node_reviewer(self, state: EMRState):
        draft = state.get("draft_record", {})
        entities = state.get("entities", [])

        validation = self.agent3.validate(draft, entities)

        new_messages = []
        is_finished_state = True

        if not validation["final_pass"]:
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
            "messages": new_messages,
            "is_finished": is_finished_state
        }

    def node_generator(self, state: EMRState):
        entities = state.get("entities", [])
        draft = self.agent2.generate_record(entities)
        return {"draft_record": draft}

    # def node_reviewer(self, state: EMRState):
    #     draft = state.get("draft_record", {})
    #     entities = state.get("entities", [])
    #
    #     validation = self.agent3.validate(draft, entities)
    #
    #     new_messages = []
    #     is_finished_state = True
    #
    #     if not validation["is_valid"]:
    #         warning_text = f"⚠️ **触发系统内部质控拦截**\n\n**拦截原因：** {validation.get('feedback')}"
    #         new_messages.append(AIMessage(content=warning_text))
    #         new_messages.append(AIMessage(content=validation["rollback_question"]))
    #         is_finished_state = False
    #
    #     return {
    #         "is_valid": validation["is_valid"],
    #         "feedback": validation["feedback"],
    #         "rollback_question": validation["rollback_question"],
    #         "messages": new_messages,
    #         "is_finished": is_finished_state
    #     }

    def route_after_extractor(self, state: EMRState):
        if state.get("is_finished") is True:
            return "generate_record"
        return "continue_asking"

    def route_after_reviewer(self, state: EMRState):
        if state.get("is_valid") is True:
            return "pass"
        return "rollback"