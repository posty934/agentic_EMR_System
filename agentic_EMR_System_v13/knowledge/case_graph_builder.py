from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional

from knowledge.graph_engine import GraphEdge, GraphNode, InMemoryGraph, MedicalGraph


class CaseGraph:
    def __init__(self, case_id: str = "case:current"):
        self.case_id = case_id
        self.draft_node_id = "draft:current"
        self.graph = InMemoryGraph()

        self.graph.add_node(
            GraphNode(
                id=self.case_id,
                kind="case",
                label="current_case",
                props={},
            )
        )

    def add_node(self, node: GraphNode) -> None:
        self.graph.add_node(node)

    def add_edge(self, edge: GraphEdge) -> None:
        self.graph.add_edge(edge)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.graph.get_node(node_id)

    def outgoing_edges(self, node_id: str, relation: Optional[str] = None):
        return self.graph.outgoing_edges(node_id, relation)

    def has_edge(self, source: str, relation: str, target: str) -> bool:
        return self.graph.has_edge(source, relation, target)

    def symptom_instances(self, status: Optional[str] = None) -> List[GraphNode]:
        nodes = [n for n in self.graph.nodes.values() if n.kind == "symptom_instance"]
        if status is None:
            return nodes
        return [n for n in nodes if n.props.get("status") == status]

    def has_filled_slot(self, instance_id: str, slot_name: str) -> bool:
        for edge in self.outgoing_edges(instance_id, "FILLS_SLOT"):
            if edge.props.get("slot") == slot_name:
                return True
        return False

    def filled_slot_values(self, instance_id: str) -> List[GraphNode]:
        values: List[GraphNode] = []
        for edge in self.outgoing_edges(instance_id, "FILLS_SLOT"):
            node = self.get_node(edge.target)
            if node and node.kind == "slot_value":
                values.append(node)
        return values

    def is_mentioned_in_draft(self, instance_id: str) -> bool:
        return self.has_edge(self.draft_node_id, "MENTIONS_ACTIVE_SYMPTOM", instance_id)

    def is_revoked_leaked_in_draft(self, instance_id: str) -> bool:
        return self.has_edge(self.draft_node_id, "LEAKS_REVOKED_SYMPTOM", instance_id)

    def draft_text(self) -> str:
        draft_node = self.get_node(self.draft_node_id)
        if not draft_node:
            return ""
        return str(draft_node.props.get("text", "") or "")

    def to_dict(self) -> Dict[str, Any]:
        return self.graph.to_dict()


class CaseGraphBuilder:
    """
    把“当前病例实体 + 草稿”编译成病例图。
    这里最关键的是：slot grounding 仍然调用你现有的 _slot_is_filled，
    所以不会损失你现在已经打磨好的字段映射与关键词兼容逻辑。
    """

    def __init__(
        self,
        ontology: MedicalGraph,
        slot_resolver: Callable[[Dict[str, Any], str], bool],
        symptom_resolver: Callable[[Dict[str, Any]], str],
        entity_text_collector: Callable[[Dict[str, Any]], str],
        slot_value_resolver: Optional[Callable[[Dict[str, Any], str], Dict[str, str]]] = None,
    ):
        self.ontology = ontology
        self.slot_resolver = slot_resolver
        self.symptom_resolver = symptom_resolver
        self.entity_text_collector = entity_text_collector
        self.slot_value_resolver = slot_value_resolver

    def _compose_draft_text(self, draft: Dict[str, Any]) -> str:
        if not isinstance(draft, dict):
            return ""
        return (
            str(draft.get("chief_complaint", "") or "").strip()
            + " "
            + str(draft.get("history_of_present_illness", "") or "").strip()
        ).strip()

    def _slot_value_node_id(self, instance_id: str, slot_name: str) -> str:
        digest = hashlib.md5(slot_name.encode("utf-8")).hexdigest()[:10]
        return f"{instance_id}:slot:{digest}"

    def build(self, current_entities: List[Dict[str, Any]], draft: Dict[str, Any]) -> CaseGraph:
        case_graph = CaseGraph()
        draft_text = self._compose_draft_text(draft)

        case_graph.add_node(
            GraphNode(
                id=case_graph.draft_node_id,
                kind="draft",
                label="current_draft",
                props={
                    "text": draft_text,
                    "raw_draft": draft or {},
                },
            )
        )
        case_graph.add_edge(
            GraphEdge(
                source=case_graph.case_id,
                relation="HAS_DRAFT",
                target=case_graph.draft_node_id,
            )
        )

        for idx, ent in enumerate(current_entities or []):
            if not isinstance(ent, dict):
                continue

            symptom_term = str(self.symptom_resolver(ent) or "").strip()
            raw_name = str(ent.get("symptom_name") or "").strip()
            standard_term = str(ent.get("standard_term") or "").strip()
            status = str(ent.get("status") or "active").strip()
            entity_text = self.entity_text_collector(ent)

            instance_id = f"symptom_instance:{idx}"

            case_graph.add_node(
                GraphNode(
                    id=instance_id,
                    kind="symptom_instance",
                    label=symptom_term or raw_name or f"symptom_{idx}",
                    props={
                        "symptom_term": symptom_term,
                        "raw_name": raw_name,
                        "standard_term": standard_term,
                        "status": status,
                        "entity_text": entity_text,
                        "raw_entity": ent,
                    },
                )
            )
            case_graph.add_edge(
                GraphEdge(
                    source=case_graph.case_id,
                    relation="HAS_SYMPTOM_INSTANCE",
                    target=instance_id,
                )
            )

            if symptom_term and self.ontology.has_symptom(symptom_term):
                case_graph.add_edge(
                    GraphEdge(
                        source=instance_id,
                        relation="INSTANCE_OF",
                        target=f"symptom:{symptom_term}",
                    )
                )

                all_slots = (
                    self.ontology.get_required_slots(symptom_term)
                    + self.ontology.get_redflag_slots(symptom_term)
                )

                for slot in all_slots:
                    if not self.slot_resolver(ent, slot):
                        continue

                    slot_value = {}
                    if self.slot_value_resolver:
                        slot_value = self.slot_value_resolver(ent, slot) or {}

                    slot_value_id = self._slot_value_node_id(instance_id, slot)
                    case_graph.add_node(
                        GraphNode(
                            id=slot_value_id,
                            kind="slot_value",
                            label=slot,
                            props={
                                "slot": slot,
                                "instance_id": instance_id,
                                "answer": slot_value.get("answer", ""),
                                "display_answer": slot_value.get("display_answer", ""),
                                "source": slot_value.get("source", ""),
                            },
                        )
                    )
                    case_graph.add_edge(
                        GraphEdge(
                            source=instance_id,
                            relation="FILLS_SLOT",
                            target=slot_value_id,
                            props={
                                "slot": slot,
                                "answer": slot_value.get("answer", ""),
                                "display_answer": slot_value.get("display_answer", ""),
                                "source": slot_value.get("source", ""),
                            },
                        )
                    )
                    case_graph.add_edge(
                        GraphEdge(
                            source=slot_value_id,
                            relation="INSTANCE_OF_SLOT",
                            target=f"slot:{slot}",
                        )
                    )

            # active symptom coverage：保持你当前逻辑
            if status != "revoked":
                coverage_candidates: List[str] = []
                if standard_term and standard_term != "未知术语":
                    coverage_candidates.append(standard_term)
                if raw_name:
                    coverage_candidates.append(raw_name)

                if any(term and term in draft_text for term in coverage_candidates):
                    case_graph.add_edge(
                        GraphEdge(
                            source=case_graph.draft_node_id,
                            relation="MENTIONS_ACTIVE_SYMPTOM",
                            target=instance_id,
                        )
                    )

            # revoked leak：严格保持你当前逻辑，只看 _get_symptom_term 的结果
            if status == "revoked":
                revoked_term = symptom_term
                if revoked_term and revoked_term in draft_text:
                    case_graph.add_edge(
                        GraphEdge(
                            source=case_graph.draft_node_id,
                            relation="LEAKS_REVOKED_SYMPTOM",
                            target=instance_id,
                        )
                    )

        return case_graph
