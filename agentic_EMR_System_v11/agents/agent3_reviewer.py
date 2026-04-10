import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from core.llm_factory import create_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from validators.kg_validator import KGValidator

load_dotenv()


class ValidationResult(BaseModel):
    is_valid: bool = Field(description="病历逻辑是否合理、完整、无自相矛盾。True为通过，False为拦截。")
    feedback: str = Field(description="质控意见。如果通过，简述理由；如果拦截，一针见血地指出具体的矛盾或缺失点。")
    rollback_question: str = Field(
        description="如果必须依赖患者补充信息才能解决，请提供一句向患者追问的话术；如果当前仅需内部自动修正，则返回空字符串。"
    )
    repair_mode: str = Field(
        description="当 is_valid=False 时，必须返回 'auto_fix' 或 'ask_user'。如果只是草稿表达、遗漏、误写，可在现有结构化信息基础上内部修正，返回 auto_fix；如果必须依赖患者补充信息才能解决，返回 ask_user。"
    )



class Agent3Reviewer:
    def __init__(self):
        self.kg_validator = KGValidator()

        self.llm = create_llm(temperature=0.0)
        self.parser = JsonOutputParser(pydantic_object=ValidationResult)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名极其严谨的医院质控主任医师（Agent 3）。
你的任务是：审查 Agent 2 生成的【病历草稿】，并对照 Agent 1 提取的【结构化症状】，判断是否存在逻辑矛盾或重大缺失。

【当前结构化症状库 (事实基准)】:
{entities}

【待审查病历草稿】:
{draft}

【审查规则】:
1. 事实一致性：主诉和现病史里的描述，绝不能和【结构化症状库】里的记录冲突。
2. 逻辑自洽性：如果患者同时有“腹泻”和“便秘”，这就是明显需要澄清的矛盾点，必须拦截。
3. 信息完整性：如果有明显的医学逻辑断层（例如提到发热，但完全没记录体温/未做红旗排查），可以进行拦截并要求追问。
4. 已排除症状忽略：请注意，系统传入的症状库中，只包含了真实有效的症状。
5. 如果问题只是“病历文稿没有正确表达已有结构化信息”，优先在 feedback 中明确说明应如何重写。
6. 如果问题必须依赖患者补充信息才能解决，请给出温和、简洁、能直接对患者说的追问话术。
7. 本系统是预问诊系统，不要要求生成诊断结论。
8. 你必须额外判断当前问题属于哪一种修复路径：
- 如果现有结构化症状信息已经足够，只是病历草稿表达错误、遗漏已确认症状、保留了应删除内容、或语义表述与结构化事实不一致，但无需患者补充新信息即可修正，则 repair_mode 必须返回 "auto_fix"。
- 如果当前问题必须依赖患者进一步补充、澄清、确认，才能决定如何修正，则 repair_mode 必须返回 "ask_user"。
- 只有在 repair_mode="ask_user" 时，rollback_question 才应该是非空；如果 repair_mode="auto_fix"，rollback_question 应返回空字符串。


{format_instructions}"""),
            ("human", "请进行严格的病历质控审查。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def _build_strict_review_entity(self, ent: dict) -> dict:
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


    def validate(self, draft: dict, current_entities: list) -> dict:
        if not current_entities or not draft:
            return {
                "kg_pass": False,
                "llm_pass": False,
                "final_pass": False,
                "issues": [{
                    "source": "system",
                    "issue_type": "empty_input",
                    "severity": "high",
                    "message": "数据为空，无法质控",
                    "repair_mode": "ask_user"
                }],
                "feedback": "数据为空，无法质控",
                "rollback_question": "抱歉，系统数据缺失，能重新描述一下症状吗？",
                "repair_instruction": "",
                "auto_revision_possible": False,
                "need_user_input": True
            }

        try:
            active_entities = [
                ent for ent in current_entities
                if ent.get("status", "active") != "revoked"
            ]

            kg_result = self.kg_validator.validate(current_entities, draft)

            strict_entities = [
                self._build_strict_review_entity(ent)
                for ent in active_entities
            ]

            llm_result = self.chain.invoke({
                "entities": json.dumps(strict_entities, ensure_ascii=False),
                "draft": json.dumps(draft, ensure_ascii=False)
            })

            llm_pass = bool(llm_result.get("is_valid", False))
            final_pass = kg_result["kg_pass"] and llm_pass

            issues = list(kg_result["issues"])

            kg_gate = self._classify_kg_repair_gate(kg_result["issues"])

            if not llm_pass:
                llm_repair_mode = (llm_result.get("repair_mode") or "").strip()
                rollback_question = (llm_result.get("rollback_question") or "").strip()

                if llm_repair_mode not in {"auto_fix", "ask_user"}:
                    llm_repair_mode = self._infer_llm_repair_mode(llm_result)

                should_append_llm_issue = True

                if kg_gate == "need_user_input":
                    llm_repair_mode = "ask_user"
                elif kg_gate == "pure_auto_fix":
                    if llm_repair_mode == "ask_user" and rollback_question:
                        should_append_llm_issue = True
                    else:
                        llm_repair_mode = "auto_fix"
                        rollback_question = ""
                        should_append_llm_issue = False

                if should_append_llm_issue:
                    issues.append({
                        "source": "llm",
                        "issue_type": "semantic_review_failed",
                        "severity": "high",
                        "repair_mode": llm_repair_mode,
                        "message": llm_result.get("feedback", "LLM语义复核未通过"),
                        "rollback_question": rollback_question
                    })

            feedback = self._merge_feedback(kg_result, llm_result)
            rollback_question = self._pick_rollback_question(issues, llm_result)

            high_issues = [x for x in issues if x.get("severity") == "high"]
            high_auto_fix = [x for x in high_issues if x.get("repair_mode") == "auto_fix"]
            high_ask_user = [x for x in high_issues if x.get("repair_mode") == "ask_user"]

            auto_revision_possible = bool(high_issues) and bool(high_auto_fix) and not high_ask_user
            need_user_input = bool(high_ask_user)
            repair_instruction = self._build_repair_instruction(
                issues=issues,
                feedback=feedback,
                auto_revision_possible=auto_revision_possible
            )

            return {
                "kg_pass": kg_result["kg_pass"],
                "llm_pass": llm_pass,
                "final_pass": final_pass,
                "issues": issues,
                "feedback": feedback,
                "rollback_question": rollback_question,
                "repair_instruction": repair_instruction,
                "auto_revision_possible": auto_revision_possible,
                "need_user_input": need_user_input
            }

        except Exception as e:
            return {
                "kg_pass": False,
                "llm_pass": False,
                "final_pass": False,
                "issues": [{
                    "source": "system",
                    "issue_type": "validator_exception",
                    "severity": "high",
                    "message": f"质控模块异常：{e}",
                    "repair_mode": "ask_user"
                }],
                "feedback": "质控模块异常，已安全拦截",
                "rollback_question": "抱歉，我还需要重新确认一下您的关键信息，请再描述一下主要不适。",
                "repair_instruction": "",
                "auto_revision_possible": False,
                "need_user_input": True
            }

    def _classify_kg_repair_gate(self, kg_issues):
        """
        基于 KG 高严重度问题，判断当前场景属于：
        1. pure_auto_fix: 结构化事实已足够，只是草稿违背事实，可直接自动修
        2. need_user_input: 结构化事实本身不足/冲突，必须追问患者
        3. no_high_issue: KG 无高严重度问题
        """
        high_issues = [x for x in kg_issues if x.get("severity") == "high"]

        if not high_issues:
            return "no_high_issue"

        high_modes = {x.get("repair_mode") for x in high_issues}

        if high_modes == {"auto_fix"}:
            return "pure_auto_fix"

        return "need_user_input"

    def _infer_llm_repair_mode(self, llm_result: dict) -> str:
        rollback_question = (llm_result.get("rollback_question") or "").strip()
        feedback = (llm_result.get("feedback") or "").strip()

        if rollback_question:
            return "ask_user"

        ask_user_keywords = [
            "澄清", "追问", "补充", "确认", "核实",
            "无法判断", "信息不足", "需进一步确认", "需要进一步确认",
            "请患者", "请补充", "请说明", "不能直接判断"
        ]
        auto_fix_keywords = [
            "重写", "改写", "补回", "删除", "修正表述",
            "与结构化信息不一致", "按结构化信息改写", "草稿遗漏", "草稿表述不准确"
        ]

        if any(keyword in feedback for keyword in ask_user_keywords):
            return "ask_user"

        if any(keyword in feedback for keyword in auto_fix_keywords):
            return "auto_fix"

        return "auto_fix"

    def _decide_repair_path(self, issues):
        """
        分流原则：
        1. 高严重度问题全部是 auto_fix -> 可以内部自动修正
        2. 高严重度问题只要出现 ask_user -> 需要先追问患者
        3. 如果 ask_user 和 auto_fix 混合存在：
           - 当前不能直接自动重写最终稿
           - 但后续在追问完成后仍应继续修稿
        """
        high_issues = [x for x in issues if x.get("severity") == "high"]

        if not high_issues:
            return False, False

        repair_modes = {issue.get("repair_mode", "ask_user") for issue in high_issues}

        if repair_modes == {"auto_fix"}:
            return True, False

        return False, True

    def _build_repair_instruction(self, issues, feedback: str, auto_revision_possible: bool) -> str:
        high_issues = [x for x in issues if x.get("severity") == "high"]
        if not high_issues:
            return ""

        auto_fix_msgs = [
            issue.get("message", "")
            for issue in high_issues
            if issue.get("repair_mode") == "auto_fix"
        ]
        ask_user_msgs = [
            issue.get("message", "")
            for issue in high_issues
            if issue.get("repair_mode") == "ask_user"
        ]

        if auto_revision_possible:
            base_instruction = (
                "请严格基于当前有效结构化症状实体，内部自动重写病历草稿；"
                "删除被撤销症状、补回草稿遗漏但已确认存在的症状、修正与实体不一致的表述；"
                "保留已经正确的内容；严禁补充任何没有证据的新症状、新诱因、新时间、新红旗征或诊断信息。"
            )
            if auto_fix_msgs:
                return base_instruction + " 需要重点修正的问题包括：" + "；".join(auto_fix_msgs)
            if feedback:
                return base_instruction + " 质控反馈如下：" + feedback
            return base_instruction

        if ask_user_msgs and auto_fix_msgs:
            return (
                    "当前存在必须先向患者澄清的问题，暂不能直接生成最终修订稿。"
                    " 请先根据 rollback_question 追问患者；待信息澄清后，再基于结构化症状实体修订草稿。"
                    " 当前待澄清问题包括：" + "；".join(ask_user_msgs)
                    + "。澄清后仍需同步修正的问题包括：" + "；".join(auto_fix_msgs)
            )

        if ask_user_msgs:
            return (
                    "当前存在必须先向患者澄清的问题，暂不能直接内部自动重写。"
                    " 请先根据 rollback_question 追问患者。待澄清问题包括：" + "；".join(ask_user_msgs)
            )

        if auto_fix_msgs:
            return (
                    "请基于当前有效结构化症状实体修订草稿。"
                    " 需要重点修正的问题包括：" + "；".join(auto_fix_msgs)
            )

        return feedback or ""

    def _pick_rollback_question(self, issues, llm_result):
        for issue in issues:
            q = issue.get("rollback_question", "").strip()
            if q:
                return q

        q = llm_result.get("rollback_question", "").strip()
        if q:
            return q

        return "我还需要再确认一下您的主要不适出现的时间、性质和伴随症状，您能再描述得具体一点吗？"

    def _merge_feedback(self, kg_result, llm_result):
        messages = []

        if not kg_result.get("kg_pass", False):
            kg_msgs = [item["message"] for item in kg_result.get("issues", [])]
            if kg_msgs:
                messages.append("图谱校验未通过：" + "；".join(kg_msgs))

        if not llm_result.get("is_valid", False):
            messages.append("LLM复核未通过：" + llm_result.get("feedback", ""))

        if not messages:
            return "图谱校验和LLM复核均通过"

        return " | ".join(messages)


if __name__ == "__main__":
    reviewer = Agent3Reviewer()

    test_cases = [
        {
            "name": "01_正常通过_腹痛信息完整",
            "expected_issue_types": [],
            "entities": [
                {
                    "symptom_name": "上腹部绞痛",
                    "standard_term": "腹痛",
                    "status": "active",
                    "onset_time": "昨天开始",
                    "location": "上腹部",
                    "characteristics": "阵发性绞痛",
                    "duration_pattern": "一阵一阵发作",
                    "radiation": "无放射痛",
                    "relation_to_food": "饭后更明显",
                    "relation_to_position": "平卧后稍缓解",
                    "negative_symptoms": "无黑便、无呕血、无发热",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "上腹痛1天",
                "history_of_present_illness": "患者自昨天开始出现上腹部阵发性绞痛，一阵一阵发作，饭后更明显，平卧后稍缓解，无放射痛，无黑便、呕血及发热。"
            }
        },
        {
            "name": "02_草稿遗漏已确认症状_应自动修正",
            "expected_issue_types": ["draft_missing_active_symptom"],
            "entities": [
                {
                    "symptom_name": "上腹部绞痛",
                    "standard_term": "腹痛",
                    "status": "active",
                    "onset_time": "昨天开始",
                    "location": "上腹部",
                    "characteristics": "阵发性绞痛",
                    "duration_pattern": "间断发作",
                    "radiation": "无放射痛",
                    "relation_to_food": "饭后加重",
                    "relation_to_position": "蜷卧稍缓解",
                    "negative_symptoms": "无黑便、无呕血、无发热",
                    "dynamic_details": {}
                },
                {
                    "symptom_name": "恶心",
                    "standard_term": "恶心",
                    "status": "active",
                    "onset_time": "昨天开始",
                    "duration_pattern": "间断出现",
                    "characteristics": "饭后恶心，反胃、想吐",
                    "severity": "轻度，仍可进食",
                    "associated_symptoms": "伴腹痛，无呕吐",
                    "inducement": "进食油腻后明显",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "上腹痛1天",
                "history_of_present_illness": "患者自昨天开始出现上腹部阵发性绞痛，饭后加重，蜷卧稍缓解，无放射痛，无黑便、呕血及发热。"
            }
        },
        {
            "name": "03_草稿保留已撤销症状_应自动修正",
            "expected_issue_types": ["revoked_symptom_leak"],
            "entities": [
                {
                    "symptom_name": "上腹部绞痛",
                    "standard_term": "腹痛",
                    "status": "active",
                    "onset_time": "昨天开始",
                    "location": "上腹部",
                    "characteristics": "绞痛",
                    "duration_pattern": "阵发性",
                    "radiation": "无放射痛",
                    "relation_to_food": "饭后加重",
                    "relation_to_position": "平卧稍缓解",
                    "negative_symptoms": "无黑便、无呕血、无发热",
                    "dynamic_details": {}
                },
                {
                    "symptom_name": "发热",
                    "standard_term": "发热",
                    "status": "revoked",
                    "onset_time": "昨晚",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "上腹痛伴发热1天",
                "history_of_present_illness": "患者自昨天开始出现上腹部绞痛，饭后加重，平卧稍缓解，同时有发热。否认黑便、呕血。"
            }
        },
        {
            "name": "04_腹泻与便秘并存_应追问澄清",
            "expected_issue_types": ["symptom_mutex_conflict", "symptom_conflicts_with"],
            "entities": [
                {
                    "symptom_name": "腹泻",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "3天前开始",
                    "characteristics": "水样便",
                    "frequency": "每天6次",
                    "associated_symptoms": "有里急后重",
                    "inducement": "进食不洁食物后出现",
                    "negative_symptoms": "无口渴尿少，无发热",
                    "dynamic_details": {}
                },
                {
                    "symptom_name": "便秘",
                    "standard_term": "便秘",
                    "status": "active",
                    "onset_time": "3天前开始",
                    "frequency": "3天1次",
                    "characteristics": "大便干硬",
                    "associated_symptoms": "排便费力，伴腹胀",
                    "inducement": "近期饮水少",
                    "negative_symptoms": "无便血黑便，无呕吐或停止排气",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "腹泻伴便秘3天",
                "history_of_present_illness": "患者3天前进食不洁食物后出现腹泻，每天约6次，为水样便，并有里急后重；同时又诉大便干硬、3天1次、排便费力并伴腹胀。无发热，无明显脱水表现。"
            }
        },
        {
            "name": "05_腹痛缺少必填槽位_应追问补充",
            "expected_issue_types": ["missing_required_slot"],
            "entities": [
                {
                    "symptom_name": "肚子痛",
                    "standard_term": "腹痛",
                    "status": "active",
                    "location": "上腹部",
                    "characteristics": "隐痛",
                    "duration_pattern": "持续存在",
                    "negative_symptoms": "无黑便、无呕血、无发热",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "腹痛",
                "history_of_present_illness": "患者诉上腹部隐痛，持续存在，无黑便、呕血及发热。"
            }
        },
        {
            "name": "06_腹痛红旗项缺失_应追问补充",
            "expected_issue_types": ["missing_redflag_check"],
            "entities": [
                {
                    "symptom_name": "上腹痛",
                    "standard_term": "腹痛",
                    "status": "active",
                    "onset_time": "昨天开始",
                    "location": "上腹部",
                    "characteristics": "绞痛",
                    "duration_pattern": "阵发性",
                    "radiation": "无放射痛",
                    "relation_to_food": "饭后加重",
                    "relation_to_position": "蜷卧后稍缓解",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "上腹痛1天",
                "history_of_present_illness": "患者自昨天开始出现上腹部阵发性绞痛，饭后加重，蜷卧后稍缓解，无放射痛。"
            }
        },
        {
            "name": "07_腹泻特征冲突_应追问澄清",
            "expected_issue_types": ["symptom_feature_conflict"],
            "entities": [
                {
                    "symptom_name": "腹泻",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "2天前",
                    "characteristics": "大便干硬，排便费力",
                    "frequency": "每天1次",
                    "associated_symptoms": "无明显里急后重",
                    "inducement": "无明确诱因",
                    "negative_symptoms": "无口渴尿少，无发热",
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "腹泻2天",
                "history_of_present_illness": "患者2天前出现所谓“腹泻”，但描述为大便干硬、排便费力，每天1次，无发热，无明显脱水。"
            }
        }
    ]

    def print_case_summary(case_name, expected_issue_types, result):
        actual_issue_types = [issue.get("issue_type") for issue in result.get("issues", [])]

        print("\n" + "=" * 100)
        print(f"测试用例: {case_name}")
        print("=" * 100)
        print(f"理论命中 issue_type: {expected_issue_types if expected_issue_types else '无'}")
        print(f"实际命中 issue_type: {actual_issue_types if actual_issue_types else '无'}")
        print(f"KG校验是否通过: {result.get('kg_pass')}")
        print(f"LLM复核是否通过: {result.get('llm_pass')}")
        print(f"最终是否通过: {result.get('final_pass')}")
        print(f"是否可自动修正: {result.get('auto_revision_possible')}")
        print(f"是否需要追问用户: {result.get('need_user_input')}")
        print(f"反馈意见: {result.get('feedback')}")
        print(f"回退追问: {result.get('rollback_question')}")
        print(f"修正指令: {result.get('repair_instruction')}")

    summary = []

    for idx, case in enumerate(test_cases, start=1):
        result = reviewer.validate(case["draft"], case["entities"])
        print_case_summary(case["name"], case["expected_issue_types"], result)

        summary.append({
            "idx": idx,
            "name": case["name"],
            "expected_issue_types": case["expected_issue_types"],
            "actual_issue_types": [x.get("issue_type") for x in result.get("issues", [])],
            "kg_pass": result.get("kg_pass"),
            "llm_pass": result.get("llm_pass"),
            "final_pass": result.get("final_pass"),
            "auto_revision_possible": result.get("auto_revision_possible"),
            "need_user_input": result.get("need_user_input")
        })

    print("\n" + "#" * 100)
    print("全部测试用例汇总")
    print("#" * 100)
    for item in summary:
        print(
            f"[{item['idx']:02d}] {item['name']} | "
            f"expected={item['expected_issue_types']} | "
            f"actual={item['actual_issue_types']} | "
            f"kg_pass={item['kg_pass']} | "
            f"llm_pass={item['llm_pass']} | "
            f"final_pass={item['final_pass']} | "
            f"auto_revision_possible={item['auto_revision_possible']} | "
            f"need_user_input={item['need_user_input']}"
        )
