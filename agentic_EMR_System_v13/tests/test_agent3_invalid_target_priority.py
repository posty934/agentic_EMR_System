from agents.agent3_reviewer import Agent3Reviewer


def build_reviewer_without_llm() -> Agent3Reviewer:
    return Agent3Reviewer.__new__(Agent3Reviewer)


def test_llm_invalid_target_reason_overrides_kg_reason_for_same_slot():
    reviewer = build_reviewer_without_llm()

    merged = reviewer._merge_invalid_targets(
        kg_targets=[
            {
                "symptom_name": "拉不出来",
                "standard_term": "便秘",
                "slot": "大便的性状(干硬便/球状便)",
                "current_answer": "水样便",
                "reason": "KG原因",
                "source": "kg",
            }
        ],
        llm_targets=[
            {
                "symptom_name": "拉不出来",
                "standard_term": "便秘",
                "slot": "大便的性状(干硬便/球状便)",
                "current_answer": "水样便",
                "reason": "LLM原因",
                "source": "llm",
            }
        ],
        llm_result={"is_valid": False, "feedback": "LLM总体反馈", "repair_mode": "ask_user"},
        llm_pass=False,
    )

    assert len(merged) == 1
    assert merged[0]["reason"] == "LLM原因"
    assert merged[0]["source"] == "llm"


def test_llm_feedback_overrides_kg_reason_when_llm_has_no_structured_target():
    reviewer = build_reviewer_without_llm()

    merged = reviewer._merge_invalid_targets(
        kg_targets=[
            {
                "symptom_name": "拉不出来",
                "standard_term": "便秘",
                "slot": "大便的性状(干硬便/球状便)",
                "current_answer": "水样便",
                "reason": "KG原因",
                "source": "kg",
            }
        ],
        llm_targets=[],
        llm_result={
            "is_valid": False,
            "feedback": "LLM认为患者修正后应以颗粒状便为准，不应继续按水样便判断。",
            "repair_mode": "ask_user",
            "rollback_question": "请再确认大便性状。",
        },
        llm_pass=False,
    )

    assert len(merged) == 1
    assert merged[0]["reason"] == "LLM认为患者修正后应以颗粒状便为准，不应继续按水样便判断。"
    assert merged[0]["kg_reason"] == "KG原因"
    assert merged[0]["source"] == "llm_feedback"
