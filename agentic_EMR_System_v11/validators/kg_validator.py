import re
from typing import Dict, List
from knowledge.graph_engine import MedicalGraph


class KGValidator:
    def __init__(self):
        self.graph = MedicalGraph()
        self.logic_rules = self.graph.get_logic_rules()

        # 稳定固定字段映射
        self.slot_field_map = {
            # ========= 时间 / 起病 =========
            "发病时间": "onset_time",
            "起病时间": "onset_time",
            "起病时间及病程": "onset_time",
            "起病时间及急缓": "onset_time",
            "起病时间及持续时长": "onset_time",
            "起病时间及持续性": "onset_time",
            "起病时间及进展快慢": "onset_time",
            "起病时间及每日发作次数": "onset_time",
            "起病时间及频率": "onset_time",
            "发现时间": "onset_time",
            "首次出现时间": "onset_time",
            "首次发现时间": "onset_time",
            "开始时间及持续多久": "onset_time",
            "开始时间及进展速度": "onset_time",
            "开始时间和下降速度": "onset_time",
            "首次发现时间及进展速度": "onset_time",

            # ========= 时长 / 频率 / 规律 =========
            "持续时间": "duration_pattern",
            "持续时间(阵发/持续)": "duration_pattern",
            "持续时间(偶发/持续超过48小时)": "duration_pattern",
            "症状持续时间及发作频率": "duration_pattern",
            "发作频率": "frequency",
            "发作频率和持续时间": "frequency",
            "每日发作次数": "frequency",
            "每日次数": "frequency",
            "发病时间及次数": "frequency",
            "排便频率(几天一次)": "frequency",
            "呕吐频次和每次量": "frequency",
            "发作规律(如饭后/夜间平卧)": "frequency",
            "起病时间及频率": "frequency",
            "起病时间及每日发作次数": "frequency",

            # ========= 性质 / 表现 =========
            "疼痛性质(绞痛/胀痛/刺痛)": "characteristics",
            "疼痛性质(刀割样/胀痛/搏动痛)": "characteristics",
            "大便的性状(水样便/黏液便/脓血便)": "characteristics",
            "大便的性状(干硬便/球状便)": "characteristics",
            "大便中脓液、血液、黏液的大致情况": "characteristics",
            "呕吐物性质(食物残渣/胆汁/咖啡样/粪臭)": "characteristics",
            "血色(鲜红/暗红/紫红)": "characteristics",
            "血液颜色(鲜红/暗红/咖啡样)及是否有血块": "characteristics",
            "异味特点(腐臭/酸味/苦味)": "characteristics",
            "反流上来的是否为未消化食物": "characteristics",
            "灰白色大便是偶发还是持续": "characteristics",
            "主要不适类型(餐后饱胀/早饱/上腹痛或烧灼感)": "characteristics",
            "失禁的是稀便还是成形便": "characteristics",
            "具体表现(淡漠/烦躁/嗜睡/定向力差)": "characteristics",

            # ========= 部位 =========
            "具体部位": "location",
            "包块部位和大致大小": "location",
            "分布部位和范围": "location",
            "分布部位及数量": "location",
            "单侧还是双侧": "location",
            "异物感部位": "location",
            "症状部位(胸骨后/上腹部)": "location",
            "疼痛位于咽部还是胸骨后": "location",
            "以上腹、下腹还是全腹为主": "location",
            "是局部还是全腹膨隆": "location",

            # ========= 程度 / 影响 =========
            "出血量": "severity",
            "次数及大致出血量": "severity",
            "乏力程度及对日常活动的影响": "severity",
            "食欲下降程度(少吃/厌食/几乎不能进食)": "severity",
            "近1-6个月大约减轻多少": "severity",
            "流涎程度及是否需频繁吐口水": "severity",
            "坠胀持续时间及严重程度": "severity",
            "是否凹陷性水肿": "severity",
            "每次呕血量": "severity",
            "每次排便量是否很少": "severity",
            "是否影响进食": "severity",
            "是否影响进食饮水": "severity",
            "是否影响睡眠、进食或饮水": "severity",
            "是否影响进食量": "severity",
            "近期体重有无下降": "severity",
            "是否为非刻意减重": "severity",
            "近期是否增大": "severity",
            "是否较平时明显增多": "severity",

            # ========= 诱因 / 关系 =========
            "诱因(不洁饮食/受凉)": "inducement",
            "诱因(进食油腻/甜食/饮酒)": "inducement",
            "诱因(进食油腻/异味刺激/妊娠可能/药物)": "inducement",
            "诱因(饮水少/膳食纤维少/活动减少/药物)": "inducement",
            "有无肝炎、胆结石、饮酒、输血或新近用药史": "inducement",
            "有无慢性肝病、长期饮酒、妊娠或雌激素相关用药史": "inducement",

            "加重或缓解因素(如进食后/体位)": "alleviating_factors",
            "与进食、平卧或弯腰的关系": "alleviating_factors",
            "与进食的关系": "relation_to_food",
            "与进食或体位的关系": "alleviating_factors",
            "与排便的关系": "relation_to_bowel",
            "与排便的关系(排前/排后/持续存在)": "relation_to_bowel",
            "与饥饿、进食或腹泻的关系": "alleviating_factors",
            "与进食、排气排便的关系": "alleviating_factors",
            "发生于进食后多久": "relation_to_food",
            "夜间或平卧后是否加重": "relation_to_position",
            "餐后是否加重，排气或排便后是否缓解": "alleviating_factors",
            "抗酸药或抬高床头后能否缓解": "alleviating_factors",
            "休息后能否缓解": "alleviating_factors",

            # ======== 这部分是你现在最需要补的 ========
            "与特定食物的关系": "relation_to_food",
            "是否与豆类、乳制品、进食过快等相关": "relation_to_food",
            "是否摄入乳制品、豆类、甜味剂或进食过快": "relation_to_food",
            "油腻饮食、饮酒、咖啡、夜宵后是否更明显": "relation_to_food",
            "是否多在饭后或吞气后出现": "relation_to_food",

            # ========= 伴随 / 否认 =========
            "是否伴腹痛、腹泻、发热、盗汗或排便习惯改变": "associated_symptoms",
            "是否伴腹痛、黏液便、脓血便或发热": "associated_symptoms",
            "是否伴腹痛、腹胀、排气增多或稀便": "associated_symptoms",
            "是否伴腹胀或腹痛": "associated_symptoms",
            "是否伴有烧心感或胸骨后疼痛": "associated_symptoms",
            "是否伴反酸、口中酸苦或胸骨后痛": "associated_symptoms",
            "是否伴反酸、烧心、恶心或腹胀": "associated_symptoms",
            "是否伴吞咽困难或咽下痛": "negative_symptoms",
            "是否伴黑便、呕血或非刻意体重下降": "negative_symptoms",
            "是否伴发热": "negative_symptoms",
            "是否伴有黑便/呕血": "negative_symptoms",
            "是否伴有极度口渴/尿少(脱水风险)": "negative_symptoms",
            "有无吞咽困难": "negative_symptoms",
            "有无呕血": "negative_symptoms",
            "排便是否困难/费力": "associated_symptoms",
            "是否伴便血/黑便": "negative_symptoms",
            "是否伴呕吐或停止排气": "negative_symptoms",
        }

        # 高价值关键词补充
        self.slot_keywords = {
            "具体部位": [
                "上腹", "下腹", "左上腹", "右上腹", "左下腹", "右下腹",
                "脐周", "剑突下", "胸骨后", "咽部", "喉咙", "肛门", "胸口"
            ],
            "症状部位(胸骨后/上腹部)": [
                "胸骨后", "上腹", "上腹部", "胸口", "胸前", "胸口正后方"
            ],
            "持续时间(阵发/持续)": [
                "阵发", "持续", "一阵一阵", "一阵阵", "间断", "反复", "一直", "持续存在"
            ],
            "发作频率和持续时间": [
                "每次", "分钟", "小时", "半小时", "每天", "每日", "反复", "发作", "持续"
            ],
            "有无向其他部位放射(如背部/右肩)": [
                "放射", "牵扯", "背部", "后背", "右肩", "肩背"
            ],
            "加重或缓解因素(如进食后/体位)": [
                "饭后加重", "饭后缓解", "进食后加重", "进食后缓解",
                "平卧", "躺下", "躺着", "卧位", "坐起", "站立", "弯腰",
                "舒服一点", "好一点", "缓解", "加重"
            ],
            "与进食、平卧或弯腰的关系": [
                "进食后", "饭后", "平卧", "躺下", "弯腰", "更明显", "更厉害", "加重", "缓解"
            ],
            "是否有里急后重感(拉完还想拉)": [
                "里急后重", "拉完还想拉", "总想上厕所", "排不尽"
            ],
            "是否伴有极度口渴/尿少(脱水风险)": [
                "口渴", "极度口渴", "尿少", "少尿", "脱水", "没有口渴", "没有尿少", "无口渴", "无尿少"
            ],
            "是否伴发热": [
                "发热", "发烧", "体温", "高热", "低热", "没有发热", "无发热", "未发热"
            ],
            "是否伴有黑便/呕血": [
                "黑便", "柏油样", "呕血", "吐血", "咖啡样",
                "没有黑便", "没有呕血", "无黑便", "无呕血"
            ],
            "是否伴有烧心感或胸骨后疼痛": [
                "烧心", "胸骨后疼痛", "胸口烧", "胸口痛"
            ],
            "是否伴反酸、口中酸苦或胸骨后痛": [
                "反酸", "口中酸苦", "嘴里发酸", "嘴里发苦", "嘴里发酸发苦",
                "胸骨后痛", "胸骨后疼痛", "胸口正后方疼痛",
                "没有反酸", "无反酸", "没有口中酸苦", "无口中酸苦",
                "没有胸骨后痛", "无胸骨后痛"
            ],
            "有无吞咽困难": [
                "吞咽困难", "吞不下", "卡住", "梗阻感", "无吞咽困难", "没有吞咽困难"
            ],
            "是否伴吞咽困难或咽下痛": [
                "吞咽困难", "吞不下", "卡住", "咽下痛", "吞咽痛", "咽的时候疼",
                "无吞咽困难", "没有吞咽困难", "无咽下痛", "没有咽下痛"
            ],
            "有无呕血": [
                "呕血", "吐血", "咖啡样呕吐", "无呕血", "没有呕血"
            ],
            "是否伴黑便、呕血或非刻意体重下降": [
                "黑便", "柏油样", "呕血", "吐血", "体重下降", "消瘦", "掉秤",
                "没有黑便", "无黑便", "没有呕血", "无呕血",
                "没有体重下降", "无体重下降"
            ],
            "排便是否困难/费力": [
                "排便困难", "费力", "使劲", "拉不出来", "解便困难"
            ],
            "是否伴腹胀或腹痛": [
                "腹胀", "腹痛", "肚子胀", "肚子痛", "肚子疼"
            ],
            "是否伴便血/黑便": [
                "便血", "大便带血", "黑便", "无便血", "无黑便", "没有便血", "没有黑便"
            ],
            "是否伴呕吐或停止排气": [
                "呕吐", "停止排气", "不放屁", "排气停止", "无呕吐", "没有呕吐"
            ],
            "与特定食物的关系": [
                "油腻", "辛辣", "甜食", "豆类", "乳制品", "牛奶", "豆浆", "咖啡",
                "夜宵", "宵夜", "碳酸饮料", "进食过快"
            ],
            "是否与豆类、乳制品、进食过快等相关": [
                "豆类", "乳制品", "牛奶", "豆浆", "奶制品", "进食过快", "吃太快"
            ],
            "是否摄入乳制品、豆类、甜味剂或进食过快": [
                "乳制品", "牛奶", "豆类", "甜味剂", "无糖饮料", "代糖", "进食过快", "吃太快"
            ],
            "油腻饮食、饮酒、咖啡、夜宵后是否更明显": [
                "油腻", "油大的东西", "油腻的东西", "吃完油腻", "吃油后", "进食油腻后",
                "饮酒", "喝酒", "酒后",
                "咖啡", "喝咖啡", "咖啡后",
                "夜宵", "宵夜", "夜宵后", "宵夜后",
                "更明显", "更厉害", "更难受", "加重"
            ],
            "是否多在饭后或吞气后出现": [
                "饭后", "餐后", "吃完", "吞气", "咽气"
            ],
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
        issues.extend(self._check_symptom_conflicts(active_entities))
        issues.extend(self._check_feature_conflicts(active_entities))
        issues.extend(self._check_revoked_leak(revoked_entities, draft))
        issues.extend(self._check_active_symptom_coverage(active_entities, draft))

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

            for slot in self.graph.get_required_slots(symptom):
                if not self._slot_is_filled(ent, slot):
                    issues.append({
                        "source": "kg",
                        "issue_type": "missing_required_slot",
                        "severity": "high",
                        "repair_mode": "ask_user",
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

            for slot in self.graph.get_redflag_slots(symptom):
                if not self._slot_is_filled(ent, slot):
                    issues.append({
                        "source": "kg",
                        "issue_type": "missing_redflag_check",
                        "severity": "high",
                        "repair_mode": "ask_user",
                        "message": f"症状[{symptom}]尚未完成红旗项排查：{slot}",
                        "rollback_question": f"还需要确认一下：{slot}"
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
                    "repair_mode": "ask_user",
                    "message": f"当前同时存在[{a}]与[{b}]，两者通常不能作为同一时刻并存的主导症状直接成立，需要澄清。",
                    "rollback_question": f"我再确认一下，您现在主要是“{a}”还是“{b}”？如果两种情况都出现过，请说明是先后变化还是同时存在。"
                })

        return issues

    def _check_symptom_conflicts(self, active_entities: List[Dict]) -> List[Dict]:
        issues = []
        active_symptoms = []
        mutex_pairs = {
            tuple(sorted(pair))
            for pair in self.logic_rules.get("symptom_mutex", [])
            if isinstance(pair, list) and len(pair) == 2
        }

        for ent in active_entities:
            symptom = self._get_symptom_term(ent)
            if symptom:
                active_symptoms.append(symptom)

        seen_pairs = set()

        for symptom in active_symptoms:
            conflict_terms = self.graph.get_conflict_symptoms(symptom)
            for other in conflict_terms:
                if other in active_symptoms:
                    key = tuple(sorted([symptom, other]))
                    if key in mutex_pairs:
                        continue
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    issues.append({
                        "source": "kg",
                        "issue_type": "symptom_conflicts_with",
                        "severity": "high",
                        "repair_mode": "ask_user",
                        "message": f"症状[{symptom}]与[{other}]在知识图谱中定义为冲突症状，通常不能直接作为同一时刻主导症状并存。",
                        "rollback_question": f"我再确认一下，您现在更符合“{symptom}”还是“{other}”？如果两种情况都出现过，请说明是先后变化还是同时存在。"
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

                if any(self._contains_conflict_keyword(entity_text, keyword) for keyword in conflict_keywords):
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

    def _contains_conflict_keyword(self, entity_text: str, keyword: str) -> bool:
        entity_text = self._normalize_text(entity_text)
        keyword_norm = self._normalize_text(keyword)

        if not entity_text or not keyword_norm:
            return False

        # 1. 先做直接子串匹配
        if keyword_norm in entity_text:
            return True

        # 2. 再做常见语序/表达变体匹配
        for variant in self._expand_conflict_keyword_variants(keyword):
            variant_norm = self._normalize_text(variant)
            if variant_norm and variant_norm in entity_text:
                return True

        # 3. 最后做“关键词拆件”匹配
        token_groups = self._expand_conflict_keyword_token_groups(keyword)
        for tokens in token_groups:
            if tokens and all(self._normalize_text(token) in entity_text for token in tokens):
                return True

        return False

    def _expand_conflict_keyword_variants(self, keyword: str) -> List[str]:
        variant_map = {
            "干硬便": ["大便干硬", "便干硬", "粪便干硬", "大便偏干硬"],
            "球状便": ["大便呈球状", "便球状", "颗粒样大便"],
            "排便困难": ["解便困难", "大便困难", "排便不畅"],
            "费力排便": ["排便费力", "解便费力", "拉大便费力"],
            "几天一次": ["数天一次", "隔几天一次", "两三天一次"],
            "三天一次": ["3天1次", "每三天一次", "三天才一次"],
            "便次减少": ["排便次数减少", "大便次数减少", "便次变少"],
            "无排便": ["未排便", "没有排便", "排不出来", "解不出来"],
            "停止排气": ["不排气", "未排气", "排气停止"],
            "大便变细": ["便变细", "粪便变细", "大便变窄"],
        }
        return variant_map.get(keyword, [])

    def _expand_conflict_keyword_token_groups(self, keyword: str) -> List[List[str]]:
        token_map = {
            "干硬便": [["干硬", "便"]],
            "球状便": [["球状", "便"]],
            "排便困难": [["排便", "困难"]],
            "费力排便": [["排便", "费力"]],
            "几天一次": [["几天", "一次"]],
            "三天一次": [["三天", "一次"], ["3天", "1次"]],
            "便次减少": [["便次", "减少"], ["排便次数", "减少"]],
            "无排便": [["无", "排便"], ["没有", "排便"]],
            "停止排气": [["停止", "排气"], ["不", "排气"]],
            "大便变细": [["大便", "变细"]],
        }
        return token_map.get(keyword, [])

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
                    "repair_mode": "auto_fix",
                    "message": f"病历草稿中出现了已撤销症状：{term}",
                    "rollback_question": ""
                })

        return issues

    def _check_active_symptom_coverage(self, active_entities: List[Dict], draft: Dict) -> List[Dict]:
        issues = []
        draft_text = (draft.get("chief_complaint", "") + " " + draft.get("history_of_present_illness", "")).strip()

        if not draft_text:
            return issues

        for ent in active_entities:
            standard_term = (ent.get("standard_term") or "").strip()
            raw_name = (ent.get("symptom_name") or "").strip()

            term_candidates = []
            if standard_term and standard_term != "未知术语":
                term_candidates.append(standard_term)
            if raw_name:
                term_candidates.append(raw_name)

            if not term_candidates:
                continue

            if not any(term in draft_text for term in term_candidates):
                display_term = standard_term or raw_name
                issues.append({
                    "source": "kg",
                    "issue_type": "draft_missing_active_symptom",
                    "severity": "high",
                    "repair_mode": "auto_fix",
                    "message": f"病历草稿遗漏了已确认症状：{display_term}",
                    "rollback_question": ""
                })

        return issues

    def _get_symptom_term(self, ent: Dict) -> str:
        return ent.get("standard_term") or ent.get("symptom_name") or ""

    def _normalize_text(self, text: str) -> str:
        text = str(text or "").strip()
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", text)

    def _get_semantic_equivalent_fields(self, slot: str) -> List[str]:
        """
        让 KG 校验从“单字段”升级为“语义字段组”。
        这是修复“左侧已显示有值，但 KG 仍判空”的核心。
        """
        slot_norm = self._normalize_text(slot)
        fields = []

        # 进食 / 食物 / 饮食相关
        if any(x in slot_norm for x in ["进食", "食物", "饮食", "饮酒", "咖啡", "夜宵", "宵夜", "餐后", "空腹", "饥饿", "油腻"]):
            fields.extend(["relation_to_food", "inducement", "alleviating_factors"])

        # 排便 / 排气相关
        if any(x in slot_norm for x in ["排便", "排气", "便意", "便后", "里急后重", "失禁"]):
            fields.extend(["relation_to_bowel", "alleviating_factors"])

        # 体位相关
        if any(x in slot_norm for x in ["体位", "平卧", "弯腰", "坐起", "站立", "卧位", "不能平卧"]):
            fields.extend(["relation_to_position", "alleviating_factors"])

        # 伴随 / 否认相关
        if any(x in slot_norm for x in ["伴", "有无", "是否伴", "是否有", "是否出现", "是否存在"]):
            fields.extend(["associated_symptoms", "negative_symptoms"])

        # 时程 / 规律
        if any(x in slot_norm for x in ["持续", "阵发", "规律", "夜间", "白天", "晨起", "全天", "波动", "进行性", "逐渐"]):
            fields.extend(["duration_pattern", "frequency", "progression"])

        # 部位
        if any(x in slot_norm for x in ["部位", "范围", "侧", "胸骨后", "上腹", "下腹", "咽部", "肛门"]):
            fields.append("location")

        # 程度 / 量 / 影响
        if any(x in slot_norm for x in ["程度", "严重", "量", "大小", "影响", "减轻多少", "体重下降", "凹陷性"]):
            fields.append("severity")

        dedup_fields = []
        seen = set()
        for field in fields:
            if field and field not in seen:
                dedup_fields.append(field)
                seen.add(field)

        return dedup_fields

    def _infer_field_from_slot(self, slot: str) -> str:
        if slot in self.slot_field_map:
            return self.slot_field_map[slot]

        slot_norm = self._normalize_text(slot)

        if any(x in slot_norm for x in ["发病时间", "起病时间", "首次发现", "首次出现", "发现时间", "开始时间", "病程"]):
            return "onset_time"

        if any(x in slot_norm for x in ["持续", "阵发", "昼夜规律", "夜间", "白天", "晨轻暮重", "进行性", "逐渐", "波动"]):
            return "duration_pattern"

        if any(x in slot_norm for x in ["频率", "次数", "每日", "每次", "多久", "发作规律", "发作频率"]):
            return "frequency"

        if any(x in slot_norm for x in ["性质", "性状", "颜色", "类型", "特点", "何种", "血色", "呕吐物", "异味"]):
            return "characteristics"

        if any(x in slot_norm for x in ["部位", "范围", "侧", "胸骨后", "上腹", "下腹", "咽部", "肛门", "包块部位"]):
            return "location"

        if any(x in slot_norm for x in ["程度", "严重", "量", "大小", "体重下降多少", "影响进食", "影响睡眠", "影响活动", "凹陷性"]):
            return "severity"

        # ===== 关键修复：先判“与进食关系”，再判“诱因” =====
        if any(x in slot_norm for x in ["进食", "食物", "饮食", "饮酒", "咖啡", "夜宵", "宵夜", "餐后", "空腹", "饥饿", "油腻"]):
            if any(x in slot_norm for x in ["关系", "更明显", "加重", "缓解", "饭后", "餐后"]):
                return "relation_to_food"

        if any(x in slot_norm for x in ["排便", "排气", "便意", "便后", "里急后重", "失禁"]):
            if "关系" in slot_norm:
                return "relation_to_bowel"

        if any(x in slot_norm for x in ["体位", "平卧", "弯腰", "坐起", "站立", "不能平卧"]):
            if "关系" in slot_norm:
                return "relation_to_position"

        if any(x in slot_norm for x in ["诱因", "旅行史", "用药史", "有关", "相关", "受凉", "接触史", "基础病"]):
            return "inducement"

        if any(x in slot_norm for x in ["缓解", "加重", "体位", "饭后", "进食后", "平卧", "弯腰", "排便后", "休息后"]):
            return "alleviating_factors"

        if any(x in slot_norm for x in ["伴", "并", "有无", "是否有", "是否伴", "是否出现", "是否存在"]):
            return "associated_symptoms"

        return ""

    def _derive_keywords_from_slot(self, slot: str) -> List[str]:
        keywords = []

        if slot in self.slot_keywords:
            keywords.extend(self.slot_keywords[slot])

        bracket_parts = re.findall(r"[（(](.*?)[）)]", slot)
        for part in bracket_parts:
            for piece in re.split(r"[、/，,；;或及与和]+", part):
                piece = piece.strip()
                if piece:
                    keywords.append(piece)

        slot_main = re.sub(r"[（(].*?[）)]", "", slot).strip()
        for piece in re.split(r"[、/，,：:；;或及与和]+", slot_main):
            piece = piece.strip()
            if piece and len(piece) >= 2:
                keywords.append(piece)

        slot_norm = self._normalize_text(slot)

        if "部位" in slot_norm or "范围" in slot_norm:
            keywords.extend([
                "上腹", "下腹", "左上腹", "右上腹", "左下腹", "右下腹",
                "脐周", "剑突下", "胸骨后", "肛门", "咽部", "喉咙", "胸口"
            ])
        if "放射" in slot_norm:
            keywords.extend(["放射", "牵扯", "背部", "后背", "右肩", "肩背"])
        if "发热" in slot_norm:
            keywords.extend(["发热", "发烧", "高热", "低热", "无发热", "没有发热"])
        if "吞咽困难" in slot_norm:
            keywords.extend(["吞咽困难", "吞不下", "卡住", "梗阻感"])
        if "咽下痛" in slot_norm:
            keywords.extend(["咽下痛", "吞咽痛", "咽的时候疼"])
        if "黑便" in slot_norm:
            keywords.extend(["黑便", "柏油样", "无黑便", "没有黑便"])
        if "呕血" in slot_norm:
            keywords.extend(["呕血", "吐血", "咖啡样", "无呕血", "没有呕血"])
        if "便血" in slot_norm:
            keywords.extend(["便血", "大便带血", "鲜血", "暗红", "无便血", "没有便血"])
        if "尿少" in slot_norm:
            keywords.extend(["尿少", "少尿", "无尿少", "没有尿少"])
        if "口渴" in slot_norm:
            keywords.extend(["口渴", "极度口渴", "无口渴", "没有口渴"])
        if "腹痛" in slot_norm:
            keywords.extend(["腹痛", "肚子疼", "肚子痛"])
        if "腹胀" in slot_norm:
            keywords.extend(["腹胀", "肚子胀"])
        if "腹泻" in slot_norm:
            keywords.extend(["腹泻", "拉肚子", "水样便", "稀便"])
        if "便秘" in slot_norm:
            keywords.extend(["便秘", "干硬便", "球状便", "排便困难"])
        if "肛门疼痛" in slot_norm:
            keywords.extend(["肛门疼", "肛周疼", "排便疼"])
        if "瘙痒" in slot_norm:
            keywords.extend(["瘙痒", "发痒", "痒"])
        if "体重下降" in slot_norm:
            keywords.extend(["体重下降", "消瘦", "瘦了", "掉秤"])
        if "停止排气" in slot_norm:
            keywords.extend(["停止排气", "不放屁", "排气停止"])
        if "水肿" in slot_norm:
            keywords.extend(["水肿", "肿", "浮肿", "凹陷"])
        if "咳嗽" in slot_norm:
            keywords.extend(["咳嗽", "干咳"])
        if "流涎" in slot_norm:
            keywords.extend(["流口水", "流涎", "口水多"])
        if "异物感" in slot_norm:
            keywords.extend(["异物感", "像有东西卡着", "堵着"])
        if "呕吐" in slot_norm:
            keywords.extend(["呕吐", "吐", "干呕"])
        if "黄疸" in slot_norm:
            keywords.extend(["黄疸", "发黄", "尿黄", "皮肤黄", "眼黄"])
        if "意识改变" in slot_norm:
            keywords.extend(["嗜睡", "烦躁", "淡漠", "意识差", "定向力差", "意识改变"])
        if any(x in slot_norm for x in ["饮食", "进食", "食物", "餐后", "油腻", "咖啡", "夜宵", "宵夜", "饮酒"]):
            keywords.extend([
                "进食后", "饭后", "餐后", "油腻", "油腻的东西", "吃完油腻",
                "饮酒", "喝酒", "酒后", "咖啡", "喝咖啡", "夜宵", "宵夜",
                "更明显", "加重", "更厉害"
            ])

        deduped = []
        seen = set()
        for item in keywords:
            item = str(item).strip()
            if item and item not in seen:
                deduped.append(item)
                seen.add(item)

        return deduped
    def _field_value_matches_slot(self, value, slot: str) -> bool:
        value_norm = self._normalize_text(value)
        if not value_norm:
            return False

        keywords = self._derive_keywords_from_slot(slot)
        normalized_keywords = [self._normalize_text(k) for k in keywords if str(k).strip()]
        for kw in normalized_keywords:
            if kw and kw in value_norm:
                return True

        slot_main = re.sub(r"[（(].*?[）)]", "", str(slot)).strip()
        slot_main_norm = self._normalize_text(slot_main)
        if slot_main_norm and slot_main_norm in value_norm:
            return True

        return False

    def _collect_slot_evidence_text(self, ent: Dict) -> str:
        text_parts = []

        for key in [
            "onset_time",
            "location",
            "characteristics",
            "duration_pattern",
            "severity",
            "inducement",
            "frequency",
            "alleviating_factors",
            "relation_to_food",
            "relation_to_bowel",
            "relation_to_position",
            "associated_symptoms",
            "negative_symptoms",
            "progression",
        ]:
            value = ent.get(key)
            if value is not None and str(value).strip() != "":
                text_parts.append(str(value))

        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            for k, v in dynamic_details.items():
                if k is not None and str(k).strip() != "":
                    text_parts.append(str(k))
                if v is not None and str(v).strip() != "":
                    text_parts.append(str(v))

        return " ".join(text_parts)

    def _slot_is_filled(self, ent: Dict, slot: str) -> bool:
        # 0) 先看显式槽位记账：这是新的主判断依据
        slot_answers = ent.get("slot_answers", {})
        if isinstance(slot_answers, dict):
            value = slot_answers.get(slot)
            if value is not None and str(value).strip() != "":
                return True

        slot_norm = self._normalize_text(slot)

        candidate_fields = []

        mapped_field = self._infer_field_from_slot(slot)
        if mapped_field:
            candidate_fields.append(mapped_field)

        candidate_fields.extend(self._get_semantic_equivalent_fields(slot))

        dedup_fields = []
        seen_fields = set()
        for field_name in candidate_fields:
            if field_name and field_name not in seen_fields:
                dedup_fields.append(field_name)
                seen_fields.add(field_name)

        strict_match_fields = {
            "associated_symptoms",
            "negative_symptoms",
            "inducement",
            "alleviating_factors",
            "relation_to_food",
            "relation_to_bowel",
            "relation_to_position",
        }

        for field_name in dedup_fields:
            value = ent.get(field_name)
            if value is None or str(value).strip() == "":
                continue

            if field_name in strict_match_fields:
                if self._field_value_matches_slot(value, slot):
                    return True
            else:
                return True

        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            for dk, dv in dynamic_details.items():
                dk_norm = self._normalize_text(dk)
                if not dk_norm or dv is None or str(dv).strip() == "":
                    continue

                if dk_norm == slot_norm or dk_norm in slot_norm or slot_norm in dk_norm:
                    return True

                if self._field_value_matches_slot(dv, slot):
                    return True

        entity_text = self._normalize_text(self._collect_slot_evidence_text(ent))
        keywords = self._derive_keywords_from_slot(slot)
        normalized_keywords = [self._normalize_text(k) for k in keywords if str(k).strip()]

        for kw in normalized_keywords:
            if kw and kw in entity_text:
                return True

        return False

    def _collect_entity_text(self, ent: Dict) -> str:
        text_parts = []

        for key in [
            "symptom_name",
            "standard_term",
            "onset_time",
            "location",
            "characteristics",
            "duration_pattern",
            "severity",
            "inducement",
            "frequency",
            "alleviating_factors",
            "relation_to_food",
            "relation_to_bowel",
            "relation_to_position",
            "associated_symptoms",
            "negative_symptoms",
            "progression",
        ]:
            value = ent.get(key)
            if value is not None and str(value).strip() != "":
                text_parts.append(str(value))

        dynamic_details = ent.get("dynamic_details", {})
        if isinstance(dynamic_details, dict):
            for k, v in dynamic_details.items():
                if k is not None and str(k).strip() != "":
                    text_parts.append(str(k))
                if v is not None and str(v).strip() != "":
                    text_parts.append(str(v))

        return " ".join(text_parts)