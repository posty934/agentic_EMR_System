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

        # 复用已有的轻量级编码器
        self.encoder = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        # 加载患者专属的向量库（如果存在）
        if os.path.exists(self.index_path) and os.path.exists(self.text_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.text_path, 'r', encoding='utf-8') as f:
                self.memory_texts = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.memory_texts = []

    def get_all_memories(self) -> list:
        """供前端 UI 侧边栏展示：获取该患者所有的既往病史"""
        return self.memory_texts

    def save_memory(self, record_summary: str):
        """问诊结束时，将病历核心内容化为向量存入"""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        memory_entry = f"[{timestamp} 就诊记录] {record_summary}"

        vector = self.encoder.encode([memory_entry]).astype('float32')
        self.index.add(vector)
        self.memory_texts.append(memory_entry)

        # 持久化到本地硬盘
        faiss.write_index(self.index, self.index_path)
        with open(self.text_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_texts, f, ensure_ascii=False)

    def retrieve_memory(self, current_query: str, top_k: int = 2) -> str:
        """问诊交互时，根据当前症状检索最相关的历史病历，供 Planner 参考"""
        if self.index.ntotal == 0:
            return "该患者为首次就诊，无既往病史记录。"

        query_vector = self.encoder.encode([current_query]).astype('float32')
        D, I = self.index.search(query_vector, top_k)

        # 过滤掉无效索引
        retrieved_memories = [self.memory_texts[i] for i in I[0] if i != -1]

        if retrieved_memories:
            return "\n".join(retrieved_memories)
        return "该患者暂无与当前症状高度相关的既往史。"