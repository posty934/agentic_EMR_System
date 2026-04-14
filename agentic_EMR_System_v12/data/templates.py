from knowledge.knowledge_base import load_record_templates


def get_outpatient_template() -> dict:
    templates = load_record_templates()
    return templates["outpatient_emr"]


def get_outpatient_template_context() -> str:
    template = get_outpatient_template()
    section_lines = [
        f"- {section['label']} ({section['key']}): {section['instruction']}"
        for section in template["sections"]
    ]
    rule_lines = [f"- {rule}" for rule in template["writing_rules"]]

    context = [
        f"模板名称: {template['name']}",
        "模板结构:",
        *section_lines,
        "书写规则:",
        *rule_lines,
        f"阴性信息规则: {template['negative_finding_rule']}"
    ]
    return "\n".join(context)
