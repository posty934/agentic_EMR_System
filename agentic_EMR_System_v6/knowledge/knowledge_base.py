# knowledge/knowledge_base.py
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(filename: str):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_symptom_dict():
    return _load_json("symptoms.json")

def load_clinical_guidelines():
    return _load_json("clinical_guidelines.json")

def load_record_templates():
    return _load_json("record_templates.json")

def load_logic_rules():
    return _load_json("logic_rules.json")

def load_medical_graph():
    return _load_json("medical_graph.json")