from typing import Dict, List
from knowledge.graph_engine import MedicalGraph


class KGValidator:
    def __init__(self):
        self.graph = MedicalGraph()
        self.logic_rules = self.graph.get_logic_rules()

        # 能直接映射到 entities 固定字段的槽位
        self.slot_field_map = {
            "发病时间": "onset_time",
            "疼痛性质(绞痛/胀痛/刺痛)": "characteristics",
            "大便的性状(水样便/黏液便/脓血便)": "characteristics",
            "每日发作次数": "frequency",
            "诱因(不洁饮食/受凉)": "inducement",
            "发作规律(如饭后/夜间平卧)": "frequency",
            "诱因(进食油腻/甜食/饮酒)": "inducement",
            "排便频率(几天一次)": "frequency",
            "大便的性状(干硬便/球状便)": "characteristics",
            "诱因(饮水少/膳食纤维少/活动减少/药物)": "inducement"
        }

        # 需要在 dynamic_details 中做关键词匹配的槽位
        self.slot_keywords = {
            "具体部位": ["部位", "位置", "左上腹", "右上腹", "左下腹", "右下腹", "上腹", "下腹", "肚脐周围"],
            "持续时间(阵发/持续)": ["持续", "阵发", "一阵一阵", "持续时间", "时长", "一直痛"],
            "有无向其他部位放射(如背部/右肩)": ["放射", "背部", "右肩", "牵扯到后背", "向背部放射", "向右肩放射"],
            "加重或缓解因素(如进食后/体位)": ["进食后加重", "进食后缓解", "体位改变", "弯腰", "平卧", "站立", "缓解因素", "加重因素"],
            "是否有里急后重感(拉完还想拉)": ["里急后重", "拉完还想拉", "总想上厕所", "没有里急后重", "无里急后重"],
            "是否伴有极度口渴/尿少(脱水风险)": ["极度口渴", "口渴", "尿少", "脱水", "不口渴", "没有尿少", "无尿少"],
            "是否伴发热": ["发热", "体温", "低热", "高热", "没有发热", "无发热"],
            "是否伴有黑便/呕血": ["黑便", "呕血", "柏油样便", "没有黑便", "没有呕血", "无黑便", "无呕血"],
            "是否伴有烧心感或胸骨后疼痛": ["烧心", "胸骨后疼痛", "胸口烧", "胸骨后烧灼感", "胸口疼"],
            "有无吞咽困难": ["吞咽困难", "吞不下", "卡住", "梗阻感", "无吞咽困难", "没有吞咽困难"],
            "有无呕血": ["呕血", "吐血", "无呕血", "没有呕血"],
            "排便是否困难/费力": ["排便困难", "费力", "使劲", "拉不出来", "不好解", "解便困难"],
            "是否伴腹胀或腹痛": ["腹胀", "腹痛", "肚子胀", "肚子痛"],
            "是否伴便血/黑便": ["便血", "黑便", "大便带血", "无便血", "无黑便", "没有便血", "没有黑便"],
            "是否伴呕吐或停止排气": ["呕吐", "停止排气", "不放屁", "排气停止", "没有呕吐", "没有停止排气"]
        }

    def validate(self, current_entities: List[Dict], draft: Dict) -> Dict:
        active_entities = [
            ent for ent in current_entities
            if ent.get("status", "active") != "revoked"
        ]
        revoked_entities = [
            ent for ent in current_entities
            if ent.get("status") == "revoked"
        ]

        issues = []

        issues.extend(self._check_required_slots(active_entities))
        issues.extend(self._check_redflag_slots(active_entities))
        issues.extend(self._check_symptom_mutex(active_entities))
        issues.extend(self._check_feature_conflicts(active_entities))
        issues.extend(self._check_revoked_leak(revoked_entities, draft))

        kg_pass = len([x for x in issues if x["severity"] == "high"]) == 0

        return {
            "kg_pass": kg_pass,
            "issues": issues
        }

    def _check_required_slots(self, active_entities: List[Dict]) -> List[Dict]:
        issues = []

        for ent in active_entities:
            symptom = self._get_symptom_term(ent)
            if not symptom or not self.graph.has_symptom(symptom):
                continue

            required_slots = self.graph.get_required_slots(symptom)
            for slot in required_slots:
                if not self._slot_is_filled(ent, slot):
                    issues.append({
                        "source": "kg",
                        "issue_type": "missing_required_slot",
                        "severity": "high",
                        "message": f"症状[{symptom}]缺少必填信息：{slot}",
                        "rollback_question": f"请再补充一下{symptom}的“{slot}”情况。"
                    })

        return issues

    def _check_redflag_slots(self, active_entities: List[Dict]) -> List[Dict]:
        issues = []

        for ent in active_entities:
            symptom = self._get_symptom_term(ent)
            if not symptom or not self.graph.has_symptom(symptom):
                continue

            redflag_slots = self.graph.get_redflag_slots(symptom)
            for slot in redflag_slots:
                if not self._slot_is_filled(ent, slot):
                    issues.append({
                        "source": "kg",
                        "issue_type": "missing_redflag_check",
                        "severity": "high",
                        "message": f"症状[{symptom}]尚未完成红旗项排查：{slot}",
                        "rollback_question": f"还需要确认一下：{slot}？"
                    })

        return issues

    def _check_symptom_mutex(self, active_entities: List[Dict]) -> List[Dict]:
        issues = []
        active_symptoms = []
        for ent in active_entities:
            symptom = self._get_symptom_term(ent)
            if symptom:
                active_symptoms.append(symptom)

        mutex_pairs = self.logic_rules.get("symptom_mutex", [])
        seen_pairs = set()

        for pair in mutex_pairs:
            if len(pair) != 2:
                continue

            a, b = pair[0], pair[1]
            if a in active_symptoms and b in active_symptoms:
                key = tuple(sorted([a, b]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                issues.append({
                    "source": "kg",
                    "issue_type": "symptom_mutex_conflict",
                    "severity": "high",
                    "message": f"当前同时存在[{a}]与[{b}]，两者通常不能作为同一时刻的主导排便状态直接并存，需要澄清是否为先后变化或口误。",
                    "rollback_question": f"我再确认一下，您现在主要是腹泻还是便秘？两者是先后变化，还是其中一个说错了？"
                })

        return issues

    def _check_feature_conflicts(self, active_entities: List[Dict]) -> List[Dict]:
        issues = []
        feature_rules = self.logic_rules.get("symptom_feature_conflicts", [])

        for ent in active_entities:
            symptom = self._get_symptom_term(ent)
            if not symptom:
                continue

            entity_text = self._collect_entity_text(ent)

            for rule in feature_rules:
                if rule.get("symptom") != symptom:
                    continue

                conflict_keywords = rule.get("conflict_keywords", [])
                if any(keyword in entity_text for keyword in conflict_keywords):
                    issues.append({
                        "source": "kg",
                        "issue_type": "symptom_feature_conflict",
                        "severity": "high",
                        "message": rule.get("message", f"症状[{symptom}]存在特征冲突。"),
                        "rollback_question": rule.get("rollback_question", f"我需要再确认一下关于{symptom}的具体表现。")
                    })

        return issues

    def _check_revoked_leak(self, revoked_entities: List[Dict], draft: Dict) -> List[Dict]:
        issues = []
        draft_text = (draft.get("chief_complaint", "") + " " + draft.get("history_of_present_illness", "")).strip()
        revoked_terms = [self._get_symptom_term(ent) for ent in revoked_entities]

        for term in revoked_terms:
            if term and term in draft_text:
                issues.append({
                    "source": "kg",
                    "issue_type": "revoked_symptom_leak",
                    "severity": "high",
                    "message": f"病历草稿中出现了已撤销症状：{term}",
                    "rollback_question": f"我再确认一下，关于{term}，您之前是否已经否认过这个症状？"
                })

        return issues

    def _get_symptom_term(self, ent: Dict) -> str:
        return ent.get("standard_term") or ent.get("symptom_name") or ""

    def _slot_is_filled(self, ent: Dict, slot: str) -> bool:
        # 1. 固定字段直接查
        mapped_field = self.slot_field_map.get(slot)
        if mapped_field:
            value = ent.get(mapped_field)
            if value is not None and str(value).strip() != "":
                return True

        # 2. dynamic_details + 全实体文本关键词查
        entity_text = self._collect_entity_text(ent)
        keywords = self.slot_keywords.get(slot, [slot])

        for kw in keywords:
            if kw in entity_text:
                return True

        return False

    def _collect_entity_text(self, ent: Dict) -> str:
        text_parts = []

        for key in ["onset_time", "characteristics", "inducement", "frequency", "alleviating_factors"]:
            value = ent.get(key)
            if value is not None:
                text_parts.append(str(value))

        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            for k, v in dynamic_details.items():
                text_parts.append(str(k))
                text_parts.append(str(v))

        return " ".join(text_parts)