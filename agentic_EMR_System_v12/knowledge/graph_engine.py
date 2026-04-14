# import json
# import os
#
#
# class MedicalGraph:
#     def __init__(self):
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         graph_path = os.path.join(current_dir, "medical_graph.json")
#
#         with open(graph_path, "r", encoding="utf-8") as f:
#             self.graph = json.load(f)
#
#         self.symptom_rules = self.graph.get("symptom_rules", {})
#         self.logic_rules = self.graph.get("logic_rules", {})
#
#     def get_required_slots(self, symptom_name: str) -> list:
#         return self.symptom_rules.get(symptom_name, {}).get("requires_slots", [])
#
#     def get_redflag_slots(self, symptom_name: str) -> list:
#         return self.symptom_rules.get(symptom_name, {}).get("redflag_slots", [])
#
#     def get_conflict_symptoms(self, symptom_name: str) -> list:
#         return self.symptom_rules.get(symptom_name, {}).get("conflicts_with", [])
#
#     def get_logic_rules(self) -> dict:
#         return self.logic_rules
#
#     def has_symptom(self, symptom_name: str) -> bool:
#         return symptom_name in self.symptom_rules
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class GraphNode:
    id: str
    kind: str
    label: str
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    props: Dict[str, Any] = field(default_factory=dict)


class InMemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.outgoing: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.incoming: Dict[str, List[GraphEdge]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        if not self.has_edge(edge.source, edge.relation, edge.target):
            self.outgoing[edge.source].append(edge)
            self.incoming[edge.target].append(edge)

    def has_edge(self, source: str, relation: str, target: str) -> bool:
        for edge in self.outgoing.get(source, []):
            if edge.relation == relation and edge.target == target:
                return True
        return False

    def neighbors(self, node_id: str, relation: Optional[str] = None) -> List[GraphNode]:
        result: List[GraphNode] = []
        for edge in self.outgoing.get(node_id, []):
            if relation and edge.relation != relation:
                continue
            node = self.get_node(edge.target)
            if node:
                result.append(node)
        return result

    def outgoing_edges(self, node_id: str, relation: Optional[str] = None) -> List[GraphEdge]:
        edges = self.outgoing.get(node_id, [])
        if relation is None:
            return list(edges)
        return [edge for edge in edges if edge.relation == relation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "label": node.label,
                    "props": node.props,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "props": edge.props,
                }
                for edges in self.outgoing.values()
                for edge in edges
            ],
        }


class MedicalGraph:
    """
    这是新的“真正知识图谱”入口：
    - medical_graph.json 不再只是规则配置，而是本体源数据
    - 运行时会被编译成节点 + 边
    - 仍然保留旧接口，保证你现有上层调用不炸
    """

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(current_dir, "medical_graph.json")

        with open(graph_path, "r", encoding="utf-8") as f:
            self.raw_graph = json.load(f)

        self.symptom_rules: Dict[str, Dict[str, Any]] = self.raw_graph.get("symptom_rules", {})
        self.logic_rules: Dict[str, Any] = self.raw_graph.get("logic_rules", {})

        self.graph = InMemoryGraph()
        self._build_ontology_graph()

    def _symptom_id(self, symptom_name: str) -> str:
        return f"symptom:{symptom_name}"

    def _slot_id(self, slot_name: str) -> str:
        return f"slot:{slot_name}"

    def _feature_rule_id(self, symptom_name: str, idx: int) -> str:
        return f"feature_rule:{symptom_name}:{idx}"

    def _ensure_symptom_node(self, symptom_name: str) -> str:
        node_id = self._symptom_id(symptom_name)
        if not self.graph.get_node(node_id):
            self.graph.add_node(
                GraphNode(
                    id=node_id,
                    kind="symptom_concept",
                    label=symptom_name,
                    props={"symptom_name": symptom_name},
                )
            )
        return node_id

    def _ensure_slot_node(self, slot_name: str) -> str:
        node_id = self._slot_id(slot_name)
        if not self.graph.get_node(node_id):
            self.graph.add_node(
                GraphNode(
                    id=node_id,
                    kind="slot_concept",
                    label=slot_name,
                    props={"slot_name": slot_name},
                )
            )
        return node_id

    def _build_ontology_graph(self) -> None:
        for symptom_name, rule in self.symptom_rules.items():
            symptom_id = self._ensure_symptom_node(symptom_name)

            for slot in rule.get("requires_slots", []):
                slot_id = self._ensure_slot_node(slot)
                self.graph.add_edge(
                    GraphEdge(
                        source=symptom_id,
                        relation="REQUIRES_SLOT",
                        target=slot_id,
                    )
                )

            for slot in rule.get("redflag_slots", []):
                slot_id = self._ensure_slot_node(slot)
                self.graph.add_edge(
                    GraphEdge(
                        source=symptom_id,
                        relation="NEEDS_REDFLAG_CHECK",
                        target=slot_id,
                    )
                )

            for other in rule.get("conflicts_with", []):
                other_id = self._ensure_symptom_node(other)
                self.graph.add_edge(
                    GraphEdge(
                        source=symptom_id,
                        relation="CONFLICTS_WITH",
                        target=other_id,
                    )
                )

        for pair in self.logic_rules.get("symptom_mutex", []):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            a, b = pair
            a_id = self._ensure_symptom_node(a)
            b_id = self._ensure_symptom_node(b)
            self.graph.add_edge(GraphEdge(source=a_id, relation="MUTEX_WITH", target=b_id))
            self.graph.add_edge(GraphEdge(source=b_id, relation="MUTEX_WITH", target=a_id))

        for idx, rule in enumerate(self.logic_rules.get("symptom_feature_conflicts", [])):
            symptom = str(rule.get("symptom") or "").strip()
            if not symptom:
                continue

            symptom_id = self._ensure_symptom_node(symptom)
            rule_id = self._feature_rule_id(symptom, idx)

            self.graph.add_node(
                GraphNode(
                    id=rule_id,
                    kind="feature_conflict_rule",
                    label=f"{symptom}_feature_conflict_{idx}",
                    props={
                        "symptom": symptom,
                        "conflict_keywords": list(rule.get("conflict_keywords", [])),
                        "message": rule.get("message", ""),
                        "rollback_question": rule.get("rollback_question", ""),
                    },
                )
            )
            self.graph.add_edge(
                GraphEdge(
                    source=symptom_id,
                    relation="HAS_FEATURE_CONFLICT_RULE",
                    target=rule_id,
                )
            )

    # ===== 下面这些方法保留旧接口，兼容你现在所有上层代码 =====

    def has_symptom(self, symptom_name: str) -> bool:
        return self.graph.get_node(self._symptom_id(symptom_name)) is not None

    def get_required_slots(self, symptom_name: str) -> List[str]:
        symptom_id = self._symptom_id(symptom_name)
        return [node.label for node in self.graph.neighbors(symptom_id, "REQUIRES_SLOT")]

    def get_redflag_slots(self, symptom_name: str) -> List[str]:
        symptom_id = self._symptom_id(symptom_name)
        return [node.label for node in self.graph.neighbors(symptom_id, "NEEDS_REDFLAG_CHECK")]

    def get_conflict_symptoms(self, symptom_name: str) -> List[str]:
        symptom_id = self._symptom_id(symptom_name)
        return [node.label for node in self.graph.neighbors(symptom_id, "CONFLICTS_WITH")]

    def get_feature_conflict_rules(self, symptom_name: str) -> List[Dict[str, Any]]:
        symptom_id = self._symptom_id(symptom_name)
        rules: List[Dict[str, Any]] = []
        for node in self.graph.neighbors(symptom_id, "HAS_FEATURE_CONFLICT_RULE"):
            rules.append(dict(node.props))
        return rules

    def get_mutex_pairs(self) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for node in self.graph.nodes.values():
            if node.kind != "symptom_concept":
                continue
            for edge in self.graph.outgoing_edges(node.id, "MUTEX_WITH"):
                left = node.label
                right_node = self.graph.get_node(edge.target)
                if right_node:
                    pairs.add(tuple(sorted([left, right_node.label])))
        return pairs

    def symptoms_conflict(self, symptom_a: str, symptom_b: str) -> bool:
        a_id = self._symptom_id(symptom_a)
        b_id = self._symptom_id(symptom_b)
        return self.graph.has_edge(a_id, "CONFLICTS_WITH", b_id)

    def get_logic_rules(self) -> Dict[str, Any]:
        return self.logic_rules