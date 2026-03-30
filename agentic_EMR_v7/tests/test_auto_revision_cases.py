from pprint import pprint
from validators.kg_validator import KGValidator


def print_case_result(case_name, result):
    print("\n" + "=" * 100)
    print(f"测试用例：{case_name}")
    print("=" * 100)
    print("KG 是否通过：", result["kg_pass"])
    print("问题数量：", len(result["issues"]))
    for idx, issue in enumerate(result["issues"], 1):
        print(f"\n[{idx}]")
        print("source      :", issue.get("source"))
        print("issue_type  :", issue.get("issue_type"))
        print("severity    :", issue.get("severity"))
        print("repair_mode :", issue.get("repair_mode"))
        print("message     :", issue.get("message"))
        print("question    :", issue.get("rollback_question"))


if __name__ == "__main__":
    validator = KGValidator()

    test_cases = [
        {
            "name": "1. 被撤销症状泄漏到草稿 -> 应判定为 auto_fix",
            "entities": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "3天前",
                    "characteristics": "水样便",
                    "inducement": "吃了辛辣火锅后",
                    "frequency": "每日4-5次",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否有里急后重感(拉完还想拉)": "无",
                        "是否伴有极度口渴/尿少(脱水风险)": "无",
                        "是否伴发热": "无"
                    }
                },
                {
                    "symptom_name": "便秘",
                    "standard_term": "便秘",
                    "status": "revoked",
                    "onset_time": "昨天",
                    "characteristics": "干硬便",
                    "inducement": None,
                    "frequency": "3天一次",
                    "alleviating_factors": None,
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "腹泻伴便秘3天",
                "history_of_present_illness": "患者3天前进食辛辣火锅后出现腹泻，大便呈水样便，每日4-5次，同时伴便秘。"
            }
        },
        {
            "name": "2. 草稿遗漏已确认症状 -> 应判定为 auto_fix",
            "entities": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "2天前",
                    "characteristics": "水样便",
                    "inducement": "饮食不洁后",
                    "frequency": "每日5次",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否有里急后重感(拉完还想拉)": "无",
                        "是否伴有极度口渴/尿少(脱水风险)": "无",
                        "是否伴发热": "无"
                    }
                },
                {
                    "symptom_name": "反酸",
                    "standard_term": "反酸",
                    "status": "active",
                    "onset_time": "1周前",
                    "characteristics": None,
                    "inducement": "进食油腻后明显",
                    "frequency": "饭后及夜间平卧时发作",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否伴有烧心感或胸骨后疼痛": "伴烧心感",
                        "有无吞咽困难": "无",
                        "有无呕血": "无"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "腹泻2天",
                "history_of_present_illness": "患者2天前饮食不洁后出现腹泻，大便呈水样便，每日5次，无发热。"
            }
        },
        {
            "name": "3. 缺少腹泻红旗征 -> 应判定为 ask_user",
            "entities": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "3天前",
                    "characteristics": "水样便",
                    "inducement": "吃了不干净的东西",
                    "frequency": "每日4次",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否有里急后重感(拉完还想拉)": "无"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "腹泻3天",
                "history_of_present_illness": "患者3天前出现腹泻，大便呈水样便，每日4次。"
            }
        },
        {
            "name": "4. 腹泻与便秘同时存在 -> 应判定为 ask_user",
            "entities": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "2天前",
                    "characteristics": "水样便",
                    "inducement": "饮食不洁后",
                    "frequency": "每日5次",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否有里急后重感(拉完还想拉)": "无",
                        "是否伴有极度口渴/尿少(脱水风险)": "无",
                        "是否伴发热": "无"
                    }
                },
                {
                    "symptom_name": "便秘",
                    "standard_term": "便秘",
                    "status": "active",
                    "onset_time": "2天前",
                    "characteristics": "干硬便",
                    "inducement": "饮水少",
                    "frequency": "3天一次",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "排便是否困难/费力": "明显费力",
                        "是否伴腹胀或腹痛": "腹胀",
                        "是否伴便血/黑便": "无",
                        "是否伴呕吐或停止排气": "无"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "腹泻伴便秘2天",
                "history_of_present_illness": "患者近2天既有腹泻，又有便秘。"
            }
        },
        {
            "name": "5. 腹痛缺高危红旗征 -> 应判定为 ask_user",
            "entities": [
                {
                    "symptom_name": "肚子痛",
                    "standard_term": "腹痛",
                    "status": "active",
                    "onset_time": "昨天",
                    "characteristics": "绞痛",
                    "inducement": None,
                    "frequency": None,
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "具体部位": "右上腹",
                        "持续时间(阵发/持续)": "阵发性",
                        "有无向其他部位放射(如背部/右肩)": "无",
                        "加重或缓解因素(如进食后/体位)": "进食后加重"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "腹痛1天",
                "history_of_present_illness": "患者昨日起出现右上腹阵发性绞痛，进食后加重，无明显放射痛。"
            }
        },
        {
            "name": "6. 信息完整且无泄漏 -> 应通过",
            "entities": [
                {
                    "symptom_name": "反酸",
                    "standard_term": "反酸",
                    "status": "active",
                    "onset_time": "2周前",
                    "characteristics": None,
                    "inducement": "进食油腻后明显",
                    "frequency": "饭后及夜间平卧时发作",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否伴有烧心感或胸骨后疼痛": "伴烧心感",
                        "有无吞咽困难": "无",
                        "有无呕血": "无"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "反酸2周",
                "history_of_present_illness": "患者2周前出现反酸，饭后及夜间平卧时发作，进食油腻后明显，伴烧心感，无吞咽困难及呕血。"
            }
        }
    ]

    for case in test_cases:
        result = validator.validate(case["entities"], case["draft"])
        print_case_result(case["name"], result)