import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class MedicalRetriever:
    def __init__(self):
        # ==========================================
        # 1. 召回模型 (Bi-Encoder) - 负责从海量库中快速捞出 Top-K
        # ==========================================
        print("⏳ 正在加载向量召回模型 (Bi-Encoder)...")
        self.encoder = SentenceTransformer('shibing624/text2vec-base-chinese')

        # ==========================================
        # 2. 重排模型 (Cross-Encoder) - 负责精准打分，优中选优
        # ==========================================
        print("⏳ 正在加载现成的重排基座模型 (Cross-Encoder)...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')

        # ==========================================
        # 3. 加载消化内科标准术语库
        # ==========================================
        self.standard_terms = self._load_standard_terms("data/digestive_terms.txt")

        # 初始化并构建 Faiss 索引
        self.index = None
        self._build_index()
        print("✅ 检索-重排模块全部加载完毕！\n")

    def _load_standard_terms(self, file_path: str) -> list:
        """从外部文件加载术语，如果文件不存在则使用内置消化科字典"""
        terms = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip()
                    if term:
                        terms.append(term)
            print(f"✅ 成功从 {file_path} 加载了 {len(terms)} 个标准术语！")
        else:
            print("⚠️ 未找到外部术语文件，自动启用内置【消化内科】标准词库。")
            terms = [
                "腹痛", "腹泻", "恶心", "呕吐", "反酸",
                "烧心", "便秘", "消化不良", "食欲不振", "腹胀",
                "吞咽困难", "黑便", "呕血", "黄疸", "胃痛"
            ]
        return terms

    def _build_index(self):
        """将标准术语转化为向量，并构建 Faiss 高速检索索引"""
        embeddings = self.encoder.encode(self.standard_terms)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """阶段一：向量粗召回 (扩大范围捞取候选词)"""
        query_vector = self.encoder.encode([query])
        D, I = self.index.search(np.array(query_vector).astype('float32'), top_k)

        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.standard_terms):
                results.append(self.standard_terms[idx])
        return results

    def rerank(self, query: str, candidate_terms: list) -> str:
        """阶段二：BERT 精重排 (Cross-Encoder 深度交叉打分)"""
        if not candidate_terms:
            return "未知术语"

        # 核心逻辑：将患者原话与每一个候选术语拼接成对 (Pairs)
        sentence_pairs = [[query, term] for term in candidate_terms]

        # 交给重排模型进行交叉注意力打分
        scores = self.reranker.predict(sentence_pairs)

        # 找出得分最高的候选词的索引
        best_idx = np.argmax(scores)
        return candidate_terms[best_idx]

    def get_standard_term(self, query: str) -> str:
        """
        对外暴露的完整流水线接口：原声实体 -> [召回] -> [重排] -> 标准术语 Top 1
        """
        # 如果大白话太短或者没有意义，直接返回兜底
        if not query or len(query) < 2:
            return query

        # 1. 先召回 5 个大概率相关的
        candidates = self.retrieve(query, top_k=5)

        # 2. 再让重排模型进行最终的精准匹配
        best_term = self.rerank(query, candidates)

        return best_term


# === 独立测试代码 ===
if __name__ == "__main__":
    retriever = MedicalRetriever()

    # 模拟几种消化内科极其口语化的患者表达


    print("--- 🔬 开始测试 [检索-重排] 完整流水线 ---")
    while True:
        query=input()
        if query:
            # 分步查看处理过程
            candidates = retriever.retrieve(query, top_k=5)
            best_match = retriever.rerank(query, candidates)

            print(f"🗣️ [患者原声]: {query}")
            print(f"   ┣ 🔍 [第一阶段 - Faiss 粗召回 Top-5]: {candidates}")
            print(f"   ┗ 🎯 [第二阶段 - BERT 精重排 Top-1]: >>> {best_match} <<<\n")