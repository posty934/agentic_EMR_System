from validators.kg_validator import KGValidator


def test_constipation_stool_slot_rejects_watery_stool():
    validator = KGValidator()
    result = validator.validate(
        [
            {
                "symptom_name": "便秘",
                "standard_term": "便秘",
                "status": "active",
                "slot_answers": {
                    "发病时间": "三天前开始",
                    "排便频率(几天一次)": "约七天排一次",
                    "大便的性状(干硬便/球状便)": "水样便",
                    "排便是否困难/费力": "有排便费力",
                    "是否伴腹胀或腹痛": "有腹胀",
                    "诱因(饮水少/膳食纤维少/活动减少/药物)": "最近喝水很少",
                    "是否伴便血/黑便": "无便血黑便",
                    "是否伴呕吐或停止排气": "无呕吐，未停止排气",
                },
                "slot_display_answers": {
                    "大便的性状(干硬便/球状便)": "水样便",
                },
            }
        ],
        {
            "chief_complaint": "便秘3天",
            "history_of_present_illness": "患者三天前开始便秘，约七天排一次。"
        },
    )

    issue_types = [issue["issue_type"] for issue in result["issues"]]
    assert result["kg_pass"] is False
    assert "slot_answer_conflict" in issue_types

    conflict_issue = next(issue for issue in result["issues"] if issue["issue_type"] == "slot_answer_conflict")
    target = conflict_issue["invalid_targets"][0]
    assert target["standard_term"] == "便秘"
    assert target["slot"] == "大便的性状(干硬便/球状便)"
    assert target["current_answer"] == "水样便"


def test_constipation_stool_slot_accepts_negated_watery_stool_with_hard_stool():
    validator = KGValidator()
    result = validator.validate(
        [
            {
                "symptom_name": "便秘",
                "standard_term": "便秘",
                "status": "active",
                "slot_answers": {
                    "发病时间": "三天前开始",
                    "排便频率(几天一次)": "约三天排一次",
                    "大便的性状(干硬便/球状便)": "不是水样便，是干硬便",
                    "排便是否困难/费力": "有排便费力",
                    "是否伴腹胀或腹痛": "有腹胀",
                    "诱因(饮水少/膳食纤维少/活动减少/药物)": "最近喝水很少",
                    "是否伴便血/黑便": "无便血黑便",
                    "是否伴呕吐或停止排气": "无呕吐，未停止排气",
                },
            }
        ],
        {
            "chief_complaint": "便秘3天",
            "history_of_present_illness": "患者三天前开始便秘，约三天排一次，大便干硬，排便费力。"
        },
    )

    issue_types = [issue["issue_type"] for issue in result["issues"]]
    assert result["kg_pass"] is True
    assert "slot_answer_conflict" not in issue_types
    assert "symptom_feature_conflict" not in issue_types


def test_current_slot_answer_overrides_stale_characteristics_for_feature_conflict():
    validator = KGValidator()
    result = validator.validate(
        [
            {
                "symptom_name": "便秘",
                "standard_term": "便秘",
                "status": "active",
                "characteristics": "水样便",
                "slot_answers": {
                    "发病时间": "约一周前开始",
                    "排便频率(几天一次)": "约五天排便一次",
                    "大便的性状(干硬便/球状便)": "大便呈颗粒状",
                    "排便是否困难/费力": "拉不出来",
                    "是否伴腹胀或腹痛": "有轻微腹胀，无腹痛",
                    "诱因(饮水少/膳食纤维少/活动减少/药物)": "最近喝水很少，活动量下降",
                    "是否伴便血/黑便": "大便有点发黑",
                    "是否伴呕吐或停止排气": "未伴呕吐或停止排气",
                },
                "slot_display_answers": {
                    "大便的性状(干硬便/球状便)": "大便呈颗粒状",
                },
            }
        ],
        {
            "chief_complaint": "便秘1周",
            "history_of_present_illness": "患者约一周前开始便秘，约五天排便一次，大便呈颗粒状，排便困难。"
        },
    )

    issue_types = [issue["issue_type"] for issue in result["issues"]]
    assert result["kg_pass"] is True
    assert "slot_answer_conflict" not in issue_types
    assert "symptom_feature_conflict" not in issue_types
