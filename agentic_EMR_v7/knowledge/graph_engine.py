import json
import os


class MedicalGraph:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(current_dir, "medical_graph.json")

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        self.symptom_rules = self.graph.get("symptom_rules", {})
        self.logic_rules = self.graph.get("logic_rules", {})

    def get_required_slots(self, symptom_name: str) -> list:
        return self.symptom_rules.get(symptom_name, {}).get("requires_slots", [])

    def get_redflag_slots(self, symptom_name: str) -> list:
        return self.symptom_rules.get(symptom_name, {}).get("redflag_slots", [])

    def get_conflict_symptoms(self, symptom_name: str) -> list:
        return self.symptom_rules.get(symptom_name, {}).get("conflicts_with", [])

    def get_logic_rules(self) -> dict:
        return self.logic_rules

    def has_symptom(self, symptom_name: str) -> bool:
        return symptom_name in self.symptom_rules