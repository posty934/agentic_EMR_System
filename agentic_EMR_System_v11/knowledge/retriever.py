import os
import json
from typing import List, Dict, Any

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder


class RemoteQwenReranker:
    """
    远程重排器客户端。

    约定接口：
    POST {base_url}/rerank
    request:
    {
        "query": "...",
        "documents": ["doc1", "doc2", ...]
    }

    response:
    {
        "scores": [0.91, 0.42, ...]
    }
    """

    def __init__(self, base_url: str, timeout: int = 15, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def score(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []

        url = f"{self.base_url}/rerank"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "query": query,
            "documents": documents,
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        scores = data.get("scores")

        if not isinstance(scores, list):
            raise ValueError(f"远程 reranker 返回格式错误: {data}")

        if len(scores) != len(documents):
            raise ValueError(
                f"远程 reranker 返回 score 数量不匹配: "
                f"docs={len(documents)}, scores={len(scores)}"
            )

        return [float(x) for x in scores]


class MedicalRetriever:
    def __init__(self):
        print("正在加载向量召回模型 (Bi-Encoder)...")
        self.encoder = SentenceTransformer("BAAI/bge-base-zh")

        self.enable_reranker = os.getenv("ENABLE_RERANKER", "true").strip().lower() == "true"
        self.reranker = None
        self.reranker_source = "disabled"

        if self.enable_reranker:
            self._init_reranker()

        self.knowledge_base = self._load_medical_dictionary()
        self.vector_entries = self._flatten_knowledge_base(self.knowledge_base)

        if not self.vector_entries:
            raise ValueError("症状库为空，无法构建索引。")

        self.index_texts = [entry["text"] for entry in self.vector_entries]

        self.index = None
        self._build_index()

        print(
            f"检索模块加载完毕！"
            f"共 {len(self.knowledge_base)} 个症状概念，"
            f"{len(self.vector_entries)} 条索引文本。"
        )
        print(f" reranker_source={self.reranker_source}\n")

    def _init_reranker(self):
        reranker_mode = os.getenv("RERANKER_MODE", "local").strip().lower()
        remote_url = os.getenv("REMOTE_RERANKER_URL", "").strip()
        remote_api_key = os.getenv("REMOTE_RERANKER_API_KEY", "").strip()
        remote_timeout = int(os.getenv("REMOTE_RERANKER_TIMEOUT", "15"))
        strict_remote = os.getenv("STRICT_REMOTE_RERANKER", "false").strip().lower() == "true"

        local_model_name = os.getenv("LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-base")

        if reranker_mode == "remote":
            if not remote_url:
                if strict_remote:
                    raise ValueError("RERANKER_MODE=remote 时必须设置 REMOTE_RERANKER_URL")
                print("未设置 REMOTE_RERANKER_URL，回退到本地重排器。")
                self.reranker = CrossEncoder(local_model_name)
                self.reranker_source = f"local:{local_model_name}"
                return

            try:
                print(f"正在初始化远程重排器: {remote_url}")
                self.reranker = RemoteQwenReranker(
                    base_url=remote_url,
                    timeout=remote_timeout,
                    api_key=remote_api_key or None,
                )
                self.reranker_source = f"remote:{remote_url}"
                print("远程重排器初始化完成。")
                return
            except Exception as e:
                if strict_remote:
                    raise RuntimeError(f"远程重排器初始化失败: {e}") from e
                print(f"远程重排器初始化失败，回退到本地模型: {e}")
                self.reranker = CrossEncoder(local_model_name)
                self.reranker_source = f"local:{local_model_name}"
                return

        print(f"正在加载本地重排器: {local_model_name}")
        self.reranker = CrossEncoder(local_model_name)
        self.reranker_source = f"local:{local_model_name}"

    def _load_medical_dictionary(self) -> List[Dict[str, Any]]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "symptoms.json")

        with open(json_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        if not isinstance(dictionary, list):
            raise ValueError("symptoms.json 顶层必须是 list。")

        return dictionary

    def _flatten_knowledge_base(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vector_entries: List[Dict[str, Any]] = []

        for concept_idx, item in enumerate(concepts):
            term = str(item.get("term", "")).strip()
            if not term:
                continue

            variants = item.get("index_variants", [])
            if not isinstance(variants, list):
                variants = []

            cleaned_variants = []
            for v in variants:
                v = str(v).strip()
                if v:
                    cleaned_variants.append(v)

            if not cleaned_variants:
                retrieval_text = str(item.get("retrieval_text", "")).strip()
                if retrieval_text:
                    cleaned_variants = [retrieval_text]
                else:
                    desc = str(item.get("desc", "")).strip()
                    fallback = f"{term}：{desc}" if desc else term
                    cleaned_variants = [fallback]

            for variant_idx, text in enumerate(cleaned_variants):
                vector_entries.append(
                    {
                        "entry_id": len(vector_entries),
                        "concept_id": concept_idx,
                        "variant_id": variant_idx,
                        "term": term,
                        "text": text,
                        "concept": item,
                    }
                )

        return vector_entries

    def _build_index(self):
        embeddings = self.encoder.encode(
            self.index_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5, raw_top_k: int = 30) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_vector = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        safe_raw_top_k = min(max(raw_top_k, top_k), len(self.vector_entries))
        scores, indices = self.index.search(query_vector, safe_raw_top_k)

        term_best: Dict[str, Dict[str, Any]] = {}
        debug_rows = []

        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.vector_entries):
                continue

            hit = self.vector_entries[idx]
            score = float(scores[0][rank])
            term = hit["term"]
            concept = hit["concept"]

            debug_rows.append(
                {
                    "rank": rank + 1,
                    "term": term,
                    "score": score,
                    "matched_text": hit["text"][:80],
                }
            )

            existing = term_best.get(term)
            if existing is None or score > existing["score"]:
                term_best[term] = {
                    "term": term,
                    "desc": concept.get("desc", ""),
                    "score": score,
                    "matched_text": hit["text"],
                    "matched_variant_id": hit["variant_id"],
                    "concept": concept,
                }

        results = sorted(
            term_best.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        print("\n" + "=" * 80)
        print(f"[Retriever] 原始查询: {query}")
        print(f"[Retriever] 第一阶段 variant 召回 raw_top_k={safe_raw_top_k}:")
        for row in debug_rows[:10]:
            print(
                f"  RawTop{row['rank']}: "
                f"term={row['term']} | score={row['score']:.4f} | "
                f"hit={row['matched_text']}..."
            )

        print("[Retriever] 第一阶段按 term 聚合后的 TopK:")
        if results:
            for i, item in enumerate(results, start=1):
                print(
                    f"  Top{i}: term={item['term']} | "
                    f"score={item['score']:.4f} | "
                    f"best_hit={item['matched_text'][:80]}..."
                )
        else:
            print("  [空]")

        return results

    def _predict_rerank_scores(self, query: str, candidate_texts: List[str]) -> List[float]:
        if not self.reranker:
            return []

        if hasattr(self.reranker, "score"):
            return [float(x) for x in self.reranker.score(query, candidate_texts)]

        if hasattr(self.reranker, "predict"):
            sentence_pairs = [[query, text] for text in candidate_texts]
            return [float(x) for x in self.reranker.predict(sentence_pairs)]

        raise TypeError(f"不支持的 reranker 类型: {type(self.reranker)}")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            print(f"[Retriever] 第二阶段重排跳过，候选为空。query={query}")
            return {"term": "未知术语", "desc": "无匹配", "score": None}

        if not self.enable_reranker or self.reranker is None:
            best = dict(candidates[0])
            best["reranker_source"] = self.reranker_source
            print(
                f"[Retriever] reranker 未启用，直接返回召回第一名: "
                f"term={best['term']} | score={best.get('score')}"
            )
            return best

        candidate_texts = []
        for cand in candidates:
            cand_text = (
                cand.get("matched_text")
                or cand.get("retrieval_text")
                or f"{cand.get('term', '')}：{cand.get('desc', '')}"
            )
            candidate_texts.append(cand_text)

        try:
            scores = self._predict_rerank_scores(query, candidate_texts)
        except Exception as e:
            print(f"rerank 调用失败，回退到召回第一名: {e}")
            best = dict(candidates[0])
            best["reranker_source"] = self.reranker_source
            return best

        print(f"[Retriever] 第二阶段重排结果: query={query} | source={self.reranker_source}")
        for idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
            print(
                f"  Rank{idx}: term={cand['term']} | "
                f"rerank_score={float(score):.4f} | "
                f"text={cand.get('matched_text', '')[:60]}..."
            )

        best_idx = int(np.argmax(scores))
        best = dict(candidates[best_idx])
        best["rerank_score"] = float(scores[best_idx])
        best["reranker_source"] = self.reranker_source

        print(
            f"[Retriever] 最终选择: term={best['term']} | "
            f"recall_score={best.get('score', None)} | "
            f"rerank_score={best['rerank_score']:.4f} | "
            f"source={self.reranker_source}"
        )
        print("=" * 80 + "\n")

        return best

    def get_standard_term(
        self,
        query: str,
        top_k: int = 5,
        raw_top_k: int = 30,
        use_rerank: bool = True,
    ) -> str:
        if not query or len(query.strip()) < 2:
            print(f"[Retriever] 查询过短，直接返回原词: {query}")
            return query

        candidates = self.retrieve(query=query, top_k=top_k, raw_top_k=raw_top_k)

        if not candidates:
            print(f"[Retriever] 无召回结果，返回 未知术语 | query={query}")
            return "未知术语"

        if use_rerank:
            best_match = self.rerank(query, candidates)
            final_term = best_match.get("term", "未知术语")
            final_score = best_match.get("rerank_score", best_match.get("score", None))
            print(
                f"[Retriever] 输出标准术语 | "
                f"query={query} | final_term={final_term} | final_score={final_score} | "
                f"source={best_match.get('reranker_source', self.reranker_source)}"
            )
        else:
            best_match = candidates[0]
            final_term = best_match.get("term", "未知术语")
            final_score = best_match.get("score", None)

            print(
                f"[Retriever] 跳过 rerank，直接使用召回第一名 | "
                f"query={query} | final_term={final_term} | final_score={final_score}"
            )
            print("=" * 80 + "\n")

        return final_term


if __name__ == "__main__":
    retriever = MedicalRetriever()

    print("--- 开始测试检索流水线 ---")
    while True:
        query = input("\n请输入患者原声 (输入 q 退出): ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        candidates = retriever.retrieve(query, top_k=5, raw_top_k=30)

        print(f"[患者原声]: {query}")
        print("[第一阶段候选]:")
        for cand in candidates:
            print(
                f"  - {cand['term']} | "
                f"score={cand['score']:.4f} | "
                f"hit={cand['matched_text'][:40]}..."
            )

        final_term = retriever.get_standard_term(
            query=query,
            top_k=5,
            raw_top_k=30,
            use_rerank=True,
        )

        print(f"[最终输出]: {final_term}")
