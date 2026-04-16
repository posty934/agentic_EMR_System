# import os
# import json
# import faiss
# import numpy as np
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
#
#
# class PatientLongTermMemory:
#     def __init__(self, patient_id: str):
#         self.patient_id = patient_id
#         self.memory_dir = "data/patient_memories"
#         os.makedirs(self.memory_dir, exist_ok=True)
#
#         self.index_path = f"{self.memory_dir}/{patient_id}.index"
#         self.text_path = f"{self.memory_dir}/{patient_id}_texts.json"
#
#         self.encoder = SentenceTransformer("shibing624/text2vec-base-chinese")
#         self.dimension = self.encoder.get_sentence_embedding_dimension()
#
#         if os.path.exists(self.index_path) and os.path.exists(self.text_path):
#             self.index = faiss.read_index(self.index_path)
#             with open(self.text_path, "r", encoding="utf-8") as f:
#                 self.memory_texts = json.load(f)
#         else:
#             self.index = faiss.IndexFlatL2(self.dimension)
#             self.memory_texts = []
#
#     def get_all_memories(self) -> list:
#         return self.memory_texts
#
#     def save_memory(self, record_summary: str):
#         timestamp = datetime.now().strftime("%Y-%m-%d")
#         memory_entry = f"[{timestamp} 就诊记录] {record_summary}"
#
#         vector = self.encoder.encode([memory_entry]).astype("float32")
#         self.index.add(vector)
#         self.memory_texts.append(memory_entry)
#
#         faiss.write_index(self.index, self.index_path)
#         with open(self.text_path, "w", encoding="utf-8") as f:
#             json.dump(self.memory_texts, f, ensure_ascii=False)
#
#     def _build_symptom_candidates(self, current_entities: list) -> list:
#         candidates = []
#         if not current_entities:
#             return candidates
#
#         for ent in current_entities:
#             if ent.get("status", "active") == "revoked":
#                 continue
#
#             standard_term = (ent.get("standard_term") or "").strip()
#             symptom_name = (ent.get("symptom_name") or "").strip()
#
#             if standard_term and standard_term != "未知术语":
#                 candidates.append(standard_term)
#             if symptom_name:
#                 candidates.append(symptom_name)
#
#         # 去重但保序
#         deduped = []
#         seen = set()
#         for item in candidates:
#             if item not in seen:
#                 deduped.append(item)
#                 seen.add(item)
#         return deduped
#
#     def _filter_memories_by_symptoms(self, current_entities: list) -> list:
#         """
#         只保留“文本中明确出现当前症状词”的既往病史。
#         这是修复“腹泻/便秘误召回腹痛既往史”的关键。
#         """
#         symptom_candidates = self._build_symptom_candidates(current_entities)
#         if not symptom_candidates:
#             return []
#
#         matched = []
#         for idx, text in enumerate(self.memory_texts):
#             if any(symptom in text for symptom in symptom_candidates):
#                 matched.append((idx, text))
#         return matched
#
#     def retrieve_memory(self, current_query: str, current_entities: list = None, top_k: int = 2) -> str:
#         """
#         根据当前症状检索真正相关的历史病历：
#         1. 先按当前 active 症状做字面过滤
#         2. 再在过滤后的结果里做向量排序
#         """
#         if not self.memory_texts:
#             return "该患者为首次就诊，无既往病史记录。"
#
#         current_entities = current_entities or []
#         matched_memories = self._filter_memories_by_symptoms(current_entities)
#
#         # 没有字面相关的既往史，就直接视为“无相关既往史”
#         if not matched_memories:
#             return "该患者暂无与当前症状直接相关的既往史。"
#
#         candidate_indices = [idx for idx, _ in matched_memories]
#         candidate_texts = [text for _, text in matched_memories]
#
#         query_vector = self.encoder.encode([current_query]).astype("float32")
#         candidate_vectors = self.encoder.encode(candidate_texts).astype("float32")
#
#         sub_index = faiss.IndexFlatL2(self.dimension)
#         sub_index.add(candidate_vectors)
#
#         k = min(top_k, len(candidate_texts))
#         D, I = sub_index.search(query_vector, k)
#
#         retrieved_memories = [candidate_texts[i] for i in I[0] if i != -1]
#
#         if retrieved_memories:
#             return "\n".join(retrieved_memories)
#
#         return "该患者暂无与当前症状直接相关的既往史。"
#
#
# if __name__ == "__main__":
#     ok = PatientLongTermMemory("10001")
#     print(ok.retrieve_memory("腹泻", current_entities=[
#         {"symptom_name": "拉肚子", "standard_term": "腹泻", "status": "active"}
#     ]))
import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer


class PatientLongTermMemory:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.memory_dir = "data/patient_memories"
        os.makedirs(self.memory_dir, exist_ok=True)

        self.index_path = f"{self.memory_dir}/{patient_id}.index"
        self.text_path = f"{self.memory_dir}/{patient_id}_texts.json"
        self.vector_path = f"{self.memory_dir}/{patient_id}_vectors.npy"
        self.profile_path = f"{self.memory_dir}/{patient_id}_profile.json"

        self.encoder = SentenceTransformer("shibing624/text2vec-base-chinese")
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        self.memory_texts = self._load_memory_texts()
        self.patient_profile = self._load_patient_profile()
        self.memory_vectors = self._load_or_rebuild_vectors()

        self.index = faiss.IndexFlatL2(self.dimension)
        if len(self.memory_vectors) > 0:
            self.index.add(self.memory_vectors)

    def _encode(self, texts: list) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        vectors = self.encoder.encode(
            texts,
            show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    def _load_memory_texts(self) -> list:
        if os.path.exists(self.text_path):
            with open(self.text_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _load_patient_profile(self) -> dict:
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                return profile if isinstance(profile, dict) else {}
            except Exception:
                return {}
        return {}

    def _load_or_rebuild_vectors(self) -> np.ndarray:
        if not self.memory_texts:
            return np.empty((0, self.dimension), dtype="float32")

        if os.path.exists(self.vector_path):
            try:
                vectors = np.load(self.vector_path)
                if len(vectors) == len(self.memory_texts):
                    return vectors.astype("float32")
            except Exception:
                pass

        vectors = self._encode(self.memory_texts)
        np.save(self.vector_path, vectors)
        return vectors

    def _persist(self):
        faiss.write_index(self.index, self.index_path)

        with open(self.text_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_texts, f, ensure_ascii=False)

        np.save(self.vector_path, self.memory_vectors)

    def get_all_memories(self) -> list:
        return self.memory_texts

    def get_patient_profile(self) -> dict:
        return dict(self.patient_profile)

    def has_patient_profile(self) -> bool:
        age = str(self.patient_profile.get("age") or "").strip()
        sex = str(self.patient_profile.get("sex") or "").strip()
        return bool(age and sex)

    def save_patient_profile(self, name: str, age: str, sex: str):
        self.patient_profile = {
            "name": str(name or "").strip(),
            "age": str(age or "").strip(),
            "sex": str(sex or "").strip(),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self.patient_profile, f, ensure_ascii=False, indent=2)

    def save_memory(self, record_summary: str):
        timestamp = datetime.now().strftime("%Y-%m-%d")
        memory_entry = f"[{timestamp} 就诊记录] {record_summary}"

        vector = self._encode([memory_entry])

        self.memory_texts.append(memory_entry)
        if len(self.memory_vectors) == 0:
            self.memory_vectors = vector
        else:
            self.memory_vectors = np.vstack([self.memory_vectors, vector]).astype("float32")

        self.index.add(vector)
        self._persist()

    def _build_symptom_candidates(self, current_entities: list) -> list:
        candidates = []
        if not current_entities:
            return candidates

        for ent in current_entities:
            if ent.get("status", "active") == "revoked":
                continue

            standard_term = (ent.get("standard_term") or "").strip()
            symptom_name = (ent.get("symptom_name") or "").strip()

            if standard_term and standard_term != "未知术语":
                candidates.append(standard_term)
            if symptom_name:
                candidates.append(symptom_name)

        deduped = []
        seen = set()
        for item in candidates:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _filter_memories_by_symptoms(self, current_entities: list) -> list:
        """
        只保留“文本中明确出现当前症状词”的既往病史。
        """
        symptom_candidates = self._build_symptom_candidates(current_entities)
        if not symptom_candidates:
            return []

        matched = []
        for idx, text in enumerate(self.memory_texts):
            if any(symptom in text for symptom in symptom_candidates):
                matched.append((idx, text))
        return matched

    def _search_vectors(self, current_query: str, candidate_texts: list, candidate_vectors: np.ndarray, top_k: int) -> list:
        if not current_query or not candidate_texts or len(candidate_vectors) == 0:
            return []

        query_vector = self._encode([current_query])

        sub_index = faiss.IndexFlatL2(self.dimension)
        sub_index.add(candidate_vectors.astype("float32"))

        k = min(top_k, len(candidate_texts))
        _, I = sub_index.search(query_vector, k)

        return [candidate_texts[i] for i in I[0] if i != -1]

    def retrieve_memory(self, current_query: str, current_entities: list = None, top_k: int = 2) -> str:
        """
        根据当前症状检索真正相关的历史病历：
        1. 先按当前 active 症状做字面过滤
        2. 再在过滤后的结果里做向量排序
        3. 如果当前轮还没有抽取出症状实体，则用患者原始输入做一次语义兜底

        与旧版相比：
        - 不再每轮对 candidate_texts 重新 encode
        - 直接复用保存时已经持久化的 memory_vectors
        """
        if not self.memory_texts:
            return "该患者为首次就诊，无既往病史记录。"

        current_entities = current_entities or []
        matched_memories = self._filter_memories_by_symptoms(current_entities)

        if not matched_memories:
            if not current_entities:
                retrieved_memories = self._search_vectors(
                    current_query=current_query,
                    candidate_texts=self.memory_texts,
                    candidate_vectors=self.memory_vectors,
                    top_k=top_k
                )
                if retrieved_memories:
                    return "\n".join(retrieved_memories)
            return "该患者暂无与当前症状直接相关的既往史。"

        candidate_indices = [idx for idx, _ in matched_memories]
        candidate_texts = [text for _, text in matched_memories]
        candidate_vectors = self.memory_vectors[candidate_indices]

        if len(candidate_vectors) == 0:
            return "该患者暂无与当前症状直接相关的既往史。"

        retrieved_memories = self._search_vectors(
            current_query=current_query,
            candidate_texts=candidate_texts,
            candidate_vectors=candidate_vectors,
            top_k=top_k
        )

        if retrieved_memories:
            return "\n".join(retrieved_memories)

        return "该患者暂无与当前症状直接相关的既往史。"


if __name__ == "__main__":
    ok = PatientLongTermMemory("10001")
    print(ok.retrieve_memory(
        "腹泻",
        current_entities=[
            {"symptom_name": "拉肚子", "standard_term": "腹泻", "status": "active"}
        ]
    ))
