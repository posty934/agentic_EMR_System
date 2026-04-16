from agents.agent2_generator import Agent2Generator


def build_agent2_without_llm() -> Agent2Generator:
    return Agent2Generator.__new__(Agent2Generator)


def test_slot_display_answer_becomes_fidelity_anchor():
    agent2 = build_agent2_without_llm()
    entity = {
        "symptom_name": "便秘",
        "standard_term": "便秘",
        "status": "active",
        "slot_answers": {
            "是否伴便血/黑便": "感觉有一些黑便",
        },
        "slot_display_answers": {
            "是否伴便血/黑便": "有少量黑便",
        },
    }

    patient_data, fact_checklist = agent2._build_patient_payload([entity])

    assert '"slot_display_answers"' in patient_data
    assert "有少量黑便" in fact_checklist
    assert "原始回答=感觉有一些黑便" in fact_checklist
    assert "保留数量限定" in fact_checklist
    assert "禁止省略成未限定的“有/伴有/出现”" in fact_checklist
    assert "保留主观/不确定性" in fact_checklist


def test_unqualified_black_stool_does_not_cover_limited_fact():
    agent2 = build_agent2_without_llm()
    entity = agent2._build_strict_entity_payload({
        "symptom_name": "便秘",
        "standard_term": "便秘",
        "status": "active",
        "slot_answers": {
            "是否伴便血/黑便": "感觉有一些黑便",
        },
        "slot_display_answers": {
            "是否伴便血/黑便": "有少量黑便",
        },
    })
    fact = agent2._iter_fact_items(entity)[0]

    assert not agent2._fact_is_covered(
        fact,
        {
            "chief_complaint": "便秘3天",
            "history_of_present_illness": "患者3天前出现便秘，伴有黑便。",
        },
    )
    assert not agent2._fact_is_covered(
        fact,
        {
            "chief_complaint": "便秘3天",
            "history_of_present_illness": "患者3天前出现便秘，有少量黑便。",
        },
    )
    assert agent2._fact_is_covered(
        fact,
        {
            "chief_complaint": "便秘3天",
            "history_of_present_illness": "患者3天前出现便秘，自觉有少量黑便。",
        },
    )


def test_record_style_polish_removes_field_labels_and_archaic_wording():
    agent2 = build_agent2_without_llm()
    result = agent2._polish_record_style({
        "chief_complaint": "吞咽困难伴胸痛1周",
        "history_of_present_illness": (
            "患者自述吞咽困难。伴随症状：自觉有轻微胸痛。"
            "否认症状：反酸、黑便。油腻饮食后亦明显。"
        ),
    })

    hpi = result["history_of_present_illness"]
    assert "伴随症状：" not in hpi
    assert "否认症状：" not in hpi
    assert "亦" not in hpi
    assert "伴轻微胸痛" in hpi
    assert "否认反酸、黑便" in hpi
