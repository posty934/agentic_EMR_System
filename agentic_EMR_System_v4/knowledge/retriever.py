import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class MedicalRetriever:
    def __init__(self):
        print("⏳ 正在加载向量召回模型 (Bi-Encoder)...")
        # 提示：由于采用了“口语 vs 专业描述”的对比，推荐后续替换为医疗微调模型
        self.encoder = SentenceTransformer('BAAI/bge-base-zh')

        print("⏳ 正在加载现成的重排基座模型 (Cross-Encoder)...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')

        # ==========================================
        # 1 & 2. 加载医学标准术语库 (Term + Description)
        # ==========================================
        self.knowledge_base = self._load_medical_dictionary()

        # 提取用于构建 Faiss 的文本： "术语：描述"
        self.index_texts = [f"{item['term']}：{item['desc']}" for item in self.knowledge_base]

        # ==========================================
        # 3. 构建 Faiss 索引
        # ==========================================
        self.index = None
        self._build_index()
        print(f"✅ 检索-重排模块加载完毕！共索引 {len(self.knowledge_base)} 条医学定义。\n")

    def _load_medical_dictionary(self) -> list:
        """
        加载标准医学字典。
        """

        with open("symptoms.json", "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        return dictionary

    def _build_index(self):
        """将【术语+描述】整体向量化，构建 Faiss 高速检索索引"""
        embeddings = self.encoder.encode(self.index_texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        4. 阶段一：向量粗召回
        只返回包含 term, desc 和用于比对的完整 text 的候选字典列表。
        """
        query_vector = self.encoder.encode([query])
        D, I = self.index.search(np.array(query_vector).astype('float32'), top_k)

        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.knowledge_base):
                results.append(self.knowledge_base[idx])
        return results

    def rerank(self, query: str, candidates: list) -> dict:
        """
        5. 阶段二：BERT 精重排 (query vs 医学描述)
        """
        if not candidates:
            return {"term": "未知术语", "desc": "无匹配"}

        # 核心逻辑：将 [患者大白话] 与 [标准术语+医学描述] 拼接成对进行交叉打分
        # 例如: ["拉肚子像水一样", "腹泻：排便次数明显超过平日习惯的频率..."]
        sentence_pairs = [[query, f"{cand['term']}：{cand['desc']}"] for cand in candidates]

        # 重排打分
        scores = self.reranker.predict(sentence_pairs)

        # 找出得分最高
        best_idx = np.argmax(scores)
        return candidates[best_idx]

    def get_standard_term(self, query: str) -> str:
        """对外暴露的接口：只吐出最终的标准术语"""
        if not query or len(query) < 2:
            return query

        candidates = self.retrieve(query, top_k=5)
        best_match = self.rerank(query, candidates)
        return best_match["term"]


# === 独立测试代码 ===
if __name__ == "__main__":
    retriever = MedicalRetriever()

    print("--- 🔬 开始测试 [定义级语义匹配] 检索-重排 流水线 ---")
    while True:
        query = input("\n请输入患者原声 (输入 q 退出): ")
        if query.lower() == 'q':
            break
        if query:
            # 阶段一：召回
            candidates = retriever.retrieve(query, top_k=5)
            # 阶段二：重排
            best_match = retriever.rerank(query, candidates)

            print(f"🗣️ [患者原声]: {query}")
            print("   ┣ 🔍 [第一阶段 - 召回候选 (Top-5)]:")
            for cand in candidates:
                print(f"   ┃    - {cand['term']} ({cand['desc'][:15]}...)")

            print(f"   ┣ 🎯 [第二阶段 - CrossEncoder 重排最高分]: {best_match['term']}：{best_match['desc']}")
            print(f"   ┗ ✅ [最终输出]: >>> {best_match['term']} <<<")