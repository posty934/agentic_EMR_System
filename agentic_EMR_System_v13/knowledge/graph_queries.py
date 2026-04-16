from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple

from knowledge.case_graph_builder import CaseGraph
from knowledge.graph_engine import MedicalGraph


class GraphQueries:
    """
    现在所有校验都变成“查图”：
    - 领域图（ontology graph）定义规则
    - 病例图（case graph）表达当前病例事实
    - validate 只负责 orchestrate
    """

    def __init__(
        self,
        ontology: MedicalGraph,
        case_graph: CaseGraph,
        conflict_matcher: Callable[[str, str], bool],
    ):
        self.ontology = ontology
        self.case_graph = case_graph
        self.conflict_matcher = conflict_matcher

    def run_all(self) -> List[Dict]:
        issues: List[Dict] = []
        issues.extend(self.find_slot_answer_conflicts())
        issues.extend(self.find_missing_required_slots())
        issues.extend(self.find_missing_redflag_slots())
        issues.extend(self.find_mutex_conflicts())
        issues.extend(self.find_symptom_conflicts())
        issues.extend(self.find_feature_conflicts())
        issues.extend(self.find_revoked_leaks())
        issues.extend(self.find_active_symptom_coverage_gaps())
        return issues

    def _active_instances(self):
        return self.case_graph.symptom_instances(status="active")

    def _revoked_instances(self):
        return self.case_graph.symptom_instances(status="revoked")

    def find_missing_required_slots(self) -> List[Dict]:
        issues: List[Dict] = []

        for inst in self._active_instances():
            symptom = str(inst.props.get("symptom_term") or "").strip()
            if not symptom or not self.ontology.has_symptom(symptom):
                continue

            for slot in self.ontology.get_required_slots(symptom):
                if self.case_graph.has_filled_slot(inst.id, slot):
                    continue

                issues.append({
                    "source": "kg",
                    "issue_type": "missing_required_slot",
                    "severity": "high",
                    "repair_mode": "ask_user",
                    "message": f"症状[{symptom}]缺少必填信息：{slot}",
                    "rollback_question": f"请再补充一下{symptom}的“{slot}”情况。"
                })

        return issues

    def find_missing_redflag_slots(self) -> List[Dict]:
        issues: List[Dict] = []

        for inst in self._active_instances():
            symptom = str(inst.props.get("symptom_term") or "").strip()
            if not symptom or not self.ontology.has_symptom(symptom):
                continue

            for slot in self.ontology.get_redflag_slots(symptom):
                if self.case_graph.has_filled_slot(inst.id, slot):
                    continue

                issues.append({
                    "source": "kg",
                    "issue_type": "missing_redflag_check",
                    "severity": "high",
                    "repair_mode": "ask_user",
                    "message": f"症状[{symptom}]尚未完成红旗项排查：{slot}",
                    "rollback_question": f"还需要确认一下：{slot}"
                })

        return issues

    def find_mutex_conflicts(self) -> List[Dict]:
        issues: List[Dict] = []

        active_symptoms: Set[str] = set()
        for inst in self._active_instances():
            symptom = str(inst.props.get("symptom_term") or "").strip()
            if symptom:
                active_symptoms.add(symptom)

        seen_pairs: Set[Tuple[str, str]] = set()

        for a, b in self.ontology.get_mutex_pairs():
            if a in active_symptoms and b in active_symptoms:
                pair = tuple(sorted([a, b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                issues.append({
                    "source": "kg",
                    "issue_type": "symptom_mutex_conflict",
                    "severity": "high",
                    "repair_mode": "ask_user",
                    "message": f"当前同时存在[{a}]与[{b}]，两者通常不能作为同一时刻并存的主导症状直接成立，需要澄清。",
                    "rollback_question": f"我再确认一下，您现在主要是“{a}”还是“{b}”？如果两种情况都出现过，请说明是先后变化还是同时存在。"
                })

        return issues

    def find_symptom_conflicts(self) -> List[Dict]:
        issues: List[Dict] = []

        active_symptoms: Set[str] = set()
        for inst in self._active_instances():
            symptom = str(inst.props.get("symptom_term") or "").strip()
            if symptom:
                active_symptoms.add(symptom)

        mutex_pairs = self.ontology.get_mutex_pairs()
        seen_pairs: Set[Tuple[str, str]] = set()

        for symptom in active_symptoms:
            for other in self.ontology.get_conflict_symptoms(symptom):
                if other not in active_symptoms:
                    continue

                pair = tuple(sorted([symptom, other]))
                if pair in mutex_pairs:
                    continue
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                issues.append({
                    "source": "kg",
                    "issue_type": "symptom_conflicts_with",
                    "severity": "high",
                    "repair_mode": "ask_user",
                    "message": f"症状[{symptom}]与[{other}]在知识图谱中定义为冲突症状，通常不能直接作为同一时刻主导症状并存。",
                    "rollback_question": f"我再确认一下，您现在更符合“{symptom}”还是“{other}”？如果两种情况都出现过，请说明是先后变化还是同时存在。"
                })

        return issues

    def find_feature_conflicts(self) -> List[Dict]:
        issues: List[Dict] = []

        for inst in self._active_instances():
            symptom = str(inst.props.get("symptom_term") or "").strip()
            if not symptom:
                continue

            entity_text = str(inst.props.get("entity_text") or "")

            for rule in self.ontology.get_feature_conflict_rules(symptom):
                conflict_keywords = list(rule.get("conflict_keywords", []))

                if any(self.conflict_matcher(entity_text, kw) for kw in conflict_keywords):
                    issues.append({
                        "source": "kg",
                        "issue_type": "symptom_feature_conflict",
                        "severity": "high",
                        "repair_mode": "ask_user",
                        "message": rule.get("message", f"症状[{symptom}]存在特征冲突。"),
                        "rollback_question": rule.get("rollback_question", f"我需要再确认一下关于{symptom}的具体表现。")
                    })
                    break

        return issues

    def find_revoked_leaks(self) -> List[Dict]:
        issues: List[Dict] = []

        for inst in self._revoked_instances():
            if not self.case_graph.is_revoked_leaked_in_draft(inst.id):
                continue

            symptom = str(inst.props.get("symptom_term") or "").strip()
            if not symptom:
                continue

            issues.append({
                "source": "kg",
                "issue_type": "revoked_symptom_leak",
                "severity": "high",
                "repair_mode": "auto_fix",
                "message": f"病历草稿中出现了已撤销症状：{symptom}",
                "rollback_question": ""
            })

        return issues

    def find_active_symptom_coverage_gaps(self) -> List[Dict]:
        issues: List[Dict] = []

        draft_text = self.case_graph.draft_text()
        if not draft_text:
            return issues

        for inst in self._active_instances():
            if self.case_graph.is_mentioned_in_draft(inst.id):
                continue

            display_term = str(
                inst.props.get("standard_term")
                or inst.props.get("raw_name")
                or inst.props.get("symptom_term")
                or ""
            ).strip()

            if not display_term:
                continue

            issues.append({
                "source": "kg",
                "issue_type": "draft_missing_active_symptom",
                "severity": "high",
                "repair_mode": "auto_fix",
                "message": f"病历草稿遗漏了已确认症状：{display_term}",
                "rollback_question": ""
            })

        return issues

    def _normalize_text(self, value: str) -> str:
        text = str(value or "").strip()
        return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff").lower()

    def _slot_matches(self, actual_slot: str, rule_slot: str) -> bool:
        actual_norm = self._normalize_text(actual_slot)
        rule_norm = self._normalize_text(rule_slot)
        return bool(
            actual_norm
            and rule_norm
            and (actual_norm == rule_norm or actual_norm in rule_norm or rule_norm in actual_norm)
        )

    def find_slot_answer_conflicts(self) -> List[Dict]:
        issues: List[Dict] = []
        seen = set()

        for inst in self._active_instances():
            symptom = str(inst.props.get("symptom_term") or "").strip()
            if not symptom:
                continue

            raw_entity = inst.props.get("raw_entity") or {}
            symptom_name = str(raw_entity.get("symptom_name") or inst.props.get("raw_name") or "").strip()
            standard_term = str(raw_entity.get("standard_term") or symptom).strip()

            rules = self.ontology.get_slot_answer_constraint_rules(symptom)
            if not rules:
                continue

            for slot_node in self.case_graph.filled_slot_values(inst.id):
                slot = str(slot_node.props.get("slot") or "").strip()
                answer = str(slot_node.props.get("answer") or "").strip()
                display_answer = str(slot_node.props.get("display_answer") or "").strip()
                answer_text = " ".join(x for x in [answer, display_answer] if x)
                if not slot or not answer_text:
                    continue

                for rule in rules:
                    rule_slot = str(rule.get("slot") or "").strip()
                    if not self._slot_matches(slot, rule_slot):
                        continue

                    conflict_keywords = list(rule.get("conflict_keywords") or [])
                    conflict_hits = [
                        keyword
                        for keyword in conflict_keywords
                        if self.conflict_matcher(answer_text, str(keyword))
                    ]
                    if not conflict_hits:
                        continue

                    key = (inst.id, slot, tuple(conflict_hits))
                    if key in seen:
                        continue
                    seen.add(key)

                    message = rule.get("message") or f"症状[{symptom}]的问诊要点“{slot}”回答与该症状不匹配。"
                    rollback_question = rule.get("rollback_question") or f"我再确认一下，您这次{symptom}的“{slot}”到底是什么情况？"

                    issues.append({
                        "source": "kg",
                        "issue_type": "slot_answer_conflict",
                        "severity": "high",
                        "repair_mode": "ask_user",
                        "message": message,
                        "rollback_question": rollback_question,
                        "invalid_targets": [{
                            "symptom_name": symptom_name,
                            "standard_term": standard_term,
                            "slot": slot,
                            "current_answer": display_answer or answer,
                            "reason": message,
                            "source": "kg",
                        }],
                        "conflict_keywords": conflict_hits,
                    })

        return issues
