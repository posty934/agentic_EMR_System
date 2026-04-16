from agents.agent1_extractor import Agent1Extractor


class FakeGraph:
    def has_symptom(self, symptom):
        return False


class FakeKGValidator:
    graph = FakeGraph()

    def _slot_is_filled(self, ent, slot):
        return bool((ent.get("slot_answers") or {}).get(slot))

    def _derive_keywords_from_slot(self, slot):
        return [slot]

    def _normalize_text(self, value):
        return str(value or "").strip()

    def _field_needs_slot_keyword_match(self, field, slot):
        return False

    def _field_value_matches_slot(self, value, slot):
        return True

    def _infer_field_from_slot(self, slot):
        return None


def build_agent1_without_llm() -> Agent1Extractor:
    agent = Agent1Extractor.__new__(Agent1Extractor)
    agent.kg_validator = FakeKGValidator()
    agent.clinical_guidelines = {
        "腹泻": {
            "必问核心要素": [
                "发病时间",
                "大便性状",
            ]
        }
    }
    return agent


def test_history_comparison_is_asked_after_guideline_slots_are_complete():
    agent = build_agent1_without_llm()
    entities = [
        {
            "symptom_name": "拉肚子",
            "standard_term": "腹泻",
            "status": "active",
            "slot_answers": {
                "发病时间": "3天前",
                "大便性状": "水样便",
            },
            "dynamic_details": {},
        }
    ]

    plan = agent.plan_next_turn(
        patient_input="水样便",
        current_entities=entities,
        long_term_memory_str="[2026-04-09 就诊记录] 主诉：腹泻3天。",
    )

    assert "之前" in plan["reply"]
    assert "更重" in plan["reply"]
    assert plan["pending_question_target"]["type"] == "history_comparison"
    assert plan["pending_question_target"]["targets"][0]["standard_term"] == "腹泻"
    assert plan["entities"][0]["is_locked"] is True


def test_history_comparison_answer_is_targeted_to_pending_symptom():
    agent = build_agent1_without_llm()
    entities = [
        {
            "symptom_name": "拉肚子",
            "standard_term": "腹泻",
            "status": "active",
            "dynamic_details": {},
        },
        {
            "symptom_name": "烧心",
            "standard_term": "烧心",
            "status": "active",
            "dynamic_details": {},
        },
    ]

    updates = agent.extract(
        patient_input="比上次更严重，没有新症状",
        current_entities=entities,
        pending_question_target={
            "type": "history_comparison",
            "symptom_name": "拉肚子",
            "standard_term": "腹泻",
            "targets": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                }
            ],
        },
    )

    assert len(updates) == 1
    assert updates[0]["standard_term"] == "腹泻"
    assert updates[0]["dynamic_details"]["与过往病史关联"] == "比上次更严重，没有新症状"
