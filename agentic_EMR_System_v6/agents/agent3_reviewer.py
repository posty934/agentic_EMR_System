import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from validators.kg_validator import KGValidator

load_dotenv()


# === 1. 定义质控输出的数据结构 ===
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="病历逻辑是否合理、完整、无自相矛盾。True为通过，False为拦截。")
    feedback: str = Field(description="质控意见。如果通过，简述理由；如果拦截，一针见血地指出具体的矛盾或缺失点。")
    rollback_question: str = Field(
        description="如果拦截，需提供一句向患者追问的话术（比如确认矛盾点）；如果通过，填空字符串。")


class Agent3Reviewer:
    def __init__(self):
        self.kg_validator=KGValidator()

        # 质控需要极高的严谨性，温度设为 0
        self.llm = ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL_NAME"),
            temperature=0.0,
        )
        self.parser = JsonOutputParser(pydantic_object=ValidationResult)

        # === 2. 核心提示词：设定质控规则 ===
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名极其严谨的医院质控主任医师（Agent 3）。
            你的任务是：审查 Agent 2 生成的【病历草稿】，并对照 Agent 1 提取的【结构化症状】，判断是否存在逻辑矛盾或重大缺失。

            【当前结构化症状库 (事实基准)】:
            {entities}

            【待审查病历草稿】:
            {draft}

            【审查规则】:
            1. 事实一致性：主诉和现病史里的描述，绝不能和【结构化症状库】里的记录冲突（如库里写饭后痛，病历里写空腹痛）。
            2. 逻辑自洽性：如果患者同时有“腹泻”和“便秘”，这就是明显需要澄清的矛盾点，必须拦截 (is_valid: false)。
            3. 信息完整性：如果有明显的医学逻辑断层（例如提到发热，但完全没记录体温），可以进行拦截并要求追问。
            4. 🚨【已排除症状忽略】：请注意，系统传入的症状库中，只包含了真实有效的症状。如果患者之前有过口误，系统已经在传入前将其清除了，所以你只需专注于当前传入的实体。
            5. 语气：如果需要拦截，rollback_question 必须用温和的医生口吻，向患者确认那个矛盾或缺失的信息。
            \n{format_instructions}"""),
            ("human", "请进行严格的病历质控审查。")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

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
                    "message": "数据为空，无法质控"
                }],
                "feedback": "数据为空，无法质控",
                "rollback_question": "抱歉，系统数据缺失，能重新描述一下症状吗？"
            }

        try:
            active_entities = [
                ent for ent in current_entities
                if ent.get("status", "active") != "revoked"
            ]

            # 第一层：知识图谱校验
            kg_result = self.kg_validator.validate(current_entities, draft)

            # 第二层：LLM语义复核
            llm_result = self.chain.invoke({
                "entities": json.dumps(active_entities, ensure_ascii=False),
                "draft": json.dumps(draft, ensure_ascii=False)
            })

            llm_pass = bool(llm_result.get("is_valid", False))
            final_pass = kg_result["kg_pass"] and llm_pass

            issues = list(kg_result["issues"])

            if not llm_pass:
                issues.append({
                    "source": "llm",
                    "issue_type": "semantic_review_failed",
                    "severity": "high",
                    "message": llm_result.get("feedback", "LLM语义复核未通过"),
                    "rollback_question": llm_result.get("rollback_question", "")
                })

            feedback = self._merge_feedback(kg_result, llm_result)
            rollback_question = self._pick_rollback_question(issues, llm_result)

            return {
                "kg_pass": kg_result["kg_pass"],
                "llm_pass": llm_pass,
                "final_pass": final_pass,
                "issues": issues,
                "feedback": feedback,
                "rollback_question": rollback_question
            }

        except Exception as e:
            # 安全策略：任何异常都默认拦截，不能默认放行
            return {
                "kg_pass": False,
                "llm_pass": False,
                "final_pass": False,
                "issues": [{
                    "source": "system",
                    "issue_type": "validator_exception",
                    "severity": "high",
                    "message": f"质控模块异常：{e}"
                }],
                "feedback": "质控模块异常，已安全拦截",
                "rollback_question": "抱歉，我还需要重新确认一下您的关键信息，请再描述一下主要不适。"
            }

    def _pick_rollback_question(self, issues, llm_result):
        for issue in issues:
            q = issue.get("rollback_question", "").strip()
            if q:
                return q

        q = llm_result.get("rollback_question", "").strip()
        if q:
            return q

        return "我需要再确认一下刚才的症状信息，您能再描述得具体一点吗？"

    def _merge_feedback(self, kg_result, llm_result):
        messages = []

        if not kg_result["kg_pass"]:
            kg_msgs = [item["message"] for item in kg_result["issues"]]
            messages.append("图谱校验未通过：" + "；".join(kg_msgs))

        if not llm_result.get("is_valid", False):
            messages.append("LLM复核未通过：" + llm_result.get("feedback", ""))

        if not messages:
            return "图谱校验和LLM复核均通过"

        return " | ".join(messages)

if __name__ == "__main__":
    from pprint import pprint

    reviewer = Agent3Reviewer()

    test_cases = [
        {
            "name": "1. 腹泻-完整信息-应通过",
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
                }
            ],
            "draft": {
                "chief_complaint": "腹泻3天",
                "history_of_present_illness": "患者于3天前进食辛辣火锅后出现腹泻，大便呈水样，每日约4-5次，无里急后重，无极度口渴或尿少，无发热。"
            }
        },
        {
            "name": "2. 腹泻-缺少核心要点-应拦截",
            "entities": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "3天前",
                    "characteristics": None,
                    "inducement": None,
                    "frequency": None,
                    "alleviating_factors": None,
                    "dynamic_details": {}
                }
            ],
            "draft": {
                "chief_complaint": "腹泻3天",
                "history_of_present_illness": "患者于3天前出现腹泻。"
            }
        },
        {
            "name": "3. 腹泻+便秘互斥-应拦截",
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
                        "排便是否困难/费力": "是",
                        "是否伴腹胀或腹痛": "腹胀",
                        "是否伴便血/黑便": "无",
                        "是否伴呕吐或停止排气": "无"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "腹泻伴便秘2天",
                "history_of_present_illness": "患者近2天自诉既腹泻又便秘。"
            }
        },
        {
            "name": "4. 腹泻但特征像便秘-应拦截",
            "entities": [
                {
                    "symptom_name": "拉肚子",
                    "standard_term": "腹泻",
                    "status": "active",
                    "onset_time": "1周前",
                    "characteristics": "干硬便、球状便",
                    "inducement": "无明显诱因",
                    "frequency": "3天一次",
                    "alleviating_factors": None,
                    "dynamic_details": {
                        "是否有里急后重感(拉完还想拉)": "无",
                        "是否伴有极度口渴/尿少(脱水风险)": "无",
                        "是否伴发热": "无",
                        "排便是否困难/费力": "明显费力"
                    }
                }
            ],
            "draft": {
                "chief_complaint": "腹泻1周",
                "history_of_present_illness": "患者诉腹泻1周，但大便干硬，排便费力。"
            }
        },
        {
            "name": "5. 反酸-完整信息-应通过",
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
                "history_of_present_illness": "患者2周前出现反酸，饭后及夜间平卧时明显，进食油腻后加重，伴烧心感，无吞咽困难及呕血。"
            }
        },
        {
            "name": "6. 腹痛-缺红旗排查-应拦截",
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
        }
    ]

    print("\n" + "=" * 80)
    print("Agent3 双重质控测试开始")
    print("=" * 80)

    for idx, case in enumerate(test_cases, 1):
        print(f"\n\n【测试用例 {idx}】{case['name']}")
        print("-" * 80)

        result = reviewer.validate(case["draft"], case["entities"])

        print("图谱校验是否通过:", result.get("kg_pass"))
        print("LLM复核是否通过:", result.get("llm_pass"))
        print("最终是否放行:", result.get("final_pass"))
        print("反馈:", result.get("feedback"))
        print("追问:", result.get("rollback_question"))

        issues = result.get("issues", [])
        print("问题数:", len(issues))
        if issues:
            print("\n详细问题列表:")
            for i, issue in enumerate(issues, 1):
                print(f"  [{i}] source={issue.get('source')} | type={issue.get('issue_type')}")
                print(f"      message={issue.get('message')}")
                if issue.get("rollback_question"):
                    print(f"      rollback_question={issue.get('rollback_question')}")

    print("\n" + "=" * 80)
    print("测试结束")
    print("=" * 80)