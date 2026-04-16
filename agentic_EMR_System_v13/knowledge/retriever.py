import os
import json
from typing import List, Dict, Any

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()


class MedicalRetriever:
    """
    多向量医学术语召回器

    设计目标：
    1. 支持每个症状 concept 下有多条 index_variants
    2. 先在 variant 粒度召回，再按 term 聚合
    3. 当前阶段优先召回率，get_standard_term 默认直接取召回第一名
    4. 使用云端 reranker 进行第二阶段重排
    """

    def __init__(self):
        print("⏳ 正在加载向量召回模型 (Bi-Encoder)...")
        self.encoder = SentenceTransformer("BAAI/bge-base-zh")

        # 当前阶段你更关心召回率，但为了兼容保留重排器开关
        self.enable_reranker = os.getenv("RERANKER_ENABLED", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.reranker = None
        self.reranker_url = os.getenv("RERANKER_API_URL", "").strip()
        self.reranker_api_key = os.getenv("RERANKER_API_KEY", "").strip()
        self.reranker_timeout = float(os.getenv("RERANKER_TIMEOUT", "30"))

        if self.enable_reranker:
            if not self.reranker_url:
                raise ValueError("已启用 reranker，但缺少环境变量 RERANKER_API_URL。")
            if not self.reranker_api_key:
                raise ValueError("已启用 reranker，但缺少环境变量 RERANKER_API_KEY。")

            print("⏳ 正在配置云端重排器 (AutoDL Reranker API)...")
            self.reranker = requests.Session()

        # 1. 加载症状 concept 库
        self.knowledge_base = self._load_medical_dictionary()

        # 2. 展平为多条可索引文本
        self.vector_entries = self._flatten_knowledge_base(self.knowledge_base)

        if not self.vector_entries:
            raise ValueError("症状库为空，无法构建索引。")

        self.index_texts = [entry["text"] for entry in self.vector_entries]

        # 3. 构建索引
        self.index = None
        self._build_index()

        print(
            f"✅ 检索模块加载完毕！"
            f"共 {len(self.knowledge_base)} 个症状概念，"
            f"{len(self.vector_entries)} 条索引文本。\n"
        )

    def _load_medical_dictionary(self) -> List[Dict[str, Any]]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "symptoms.json")

        with open(json_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        if not isinstance(dictionary, list):
            raise ValueError("symptoms.json 顶层必须是 list。")

        return dictionary

    def _flatten_knowledge_base(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 concept 级术语库展开成 variant 级索引文本。

        优先顺序：
        1. index_variants
        2. retrieval_text
        3. fallback: term + desc
        """
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
                vector_entries.append({
                    "entry_id": len(vector_entries),
                    "concept_id": concept_idx,
                    "variant_id": variant_idx,
                    "term": term,
                    "text": text,
                    "concept": item,
                })

        return vector_entries

    def _build_index(self):
        """
        将所有 variant 文本向量化，构建 Faiss 索引。
        使用归一化向量 + Inner Product，等价于 cosine 相似度检索。
        """
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
        """
        第一阶段：多向量粗召回

        流程：
        1. 在所有 index_variants 上搜索 raw_top_k
        2. 按 term 聚合，保留每个 term 的最高分 variant
        3. 返回 term-level top_k

        返回结果中的每项包含：
        - term
        - desc
        - score
        - matched_text      命中的最佳索引文本
        - matched_variant_id
        - concept           原始 concept 对象
        """
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

            debug_rows.append({
                "rank": rank + 1,
                "term": term,
                "score": score,
                "matched_text": hit["text"][:80]
            })

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
            reverse=True
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

    def _build_reranker_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.reranker_api_key}",
        }

    def _parse_reranker_scores(self, response_data: Any, expected_count: int) -> List[float]:
        def extract_score(item: Dict[str, Any]):
            for key in ("score", "relevance_score", "rerank_score", "probability", "logit"):
                if key in item:
                    return float(item[key])
            return None

        def parse_score_list(items: Any):
            if not isinstance(items, list):
                return None

            if len(items) == expected_count and all(isinstance(item, (int, float)) for item in items):
                return [float(item) for item in items]

            if all(isinstance(item, dict) for item in items):
                scores = [None] * expected_count

                for position, item in enumerate(items):
                    score = extract_score(item)
                    if score is None:
                        continue

                    index = item.get(
                        "index",
                        item.get(
                            "document_index",
                            item.get(
                                "corpus_id",
                                item.get("id", position)
                            )
                        )
                    )

                    try:
                        index = int(index)
                    except (TypeError, ValueError):
                        index = position

                    if 0 <= index < expected_count:
                        scores[index] = score

                if all(score is not None for score in scores):
                    return [float(score) for score in scores]

                if len(items) == expected_count:
                    ordered_scores = []
                    for item in items:
                        score = extract_score(item)
                        if score is None:
                            break
                        ordered_scores.append(score)

                    if len(ordered_scores) == expected_count:
                        return [float(score) for score in ordered_scores]

            return None

        scores = parse_score_list(response_data)
        if scores is not None:
            return scores

        if isinstance(response_data, dict):
            for key in ("scores", "rerank_scores", "relevance_scores"):
                scores = parse_score_list(response_data.get(key))
                if scores is not None:
                    return scores

            for key in ("results", "data", "documents", "rankings", "ranked_documents"):
                scores = parse_score_list(response_data.get(key))
                if scores is not None:
                    return scores

            for key in ("result", "output"):
                nested = response_data.get(key)
                if isinstance(nested, (dict, list)):
                    return self._parse_reranker_scores(nested, expected_count)

            data = response_data.get("data")
            if isinstance(data, dict):
                return self._parse_reranker_scores(data, expected_count)

        raise ValueError(f"无法解析云端 reranker 返回格式: {response_data}")

    def _call_cloud_reranker(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "query": query,
            "documents": documents,
        }

        try:
            response = self.reranker.post(
                self.reranker_url,
                headers=self._build_reranker_headers(),
                json=payload,
                timeout=self.reranker_timeout,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"云端 reranker 请求失败: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"云端 reranker 返回不是合法 JSON: {e}") from e

        return self._parse_reranker_scores(response_data, expected_count=len(documents))

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        第二阶段：云端 Reranker 重排
        """
        if not candidates:
            print(f"[Retriever] 第二阶段重排跳过，候选为空。query={query}")
            return {"term": "未知术语", "desc": "无匹配", "score": None}

        if not self.enable_reranker or self.reranker is None:
            best = dict(candidates[0])
            print(
                f"[Retriever] reranker 未启用，直接返回召回第一名: "
                f"term={best['term']} | score={best.get('score')}"
            )
            return best

        candidate_texts = []
        for cand in candidates:
            cand_text = f"{cand.get('term', '')}：{cand.get('desc', '')}"
            candidate_texts.append(cand_text)

        scores = np.asarray(
            self._call_cloud_reranker(query=query, documents=candidate_texts),
            dtype="float32",
        )

        print(f"[Retriever] 第二阶段重排结果: query={query}")
        for idx, (cand, score, cand_text) in enumerate(zip(candidates, scores, candidate_texts), start=1):
            print(
                f"  Rank{idx}: term={cand['term']} | "
                f"rerank_score={float(score):.4f} | "
                f"text={cand_text[:60]}..."
            )

        best_idx = int(np.argmax(scores))
        best = dict(candidates[best_idx])
        best["rerank_score"] = float(scores[best_idx])

        print(
            f"[Retriever] 最终选择: term={best['term']} | "
            f"recall_score={best.get('score', None)} | "
            f"rerank_score={best['rerank_score']:.4f}"
        )
        print("=" * 80 + "\n")

        return best

    def get_standard_term(
        self,
        query: str,
        top_k: int = 5,
        raw_top_k: int = 30,
        use_rerank: bool = True
    ) -> str:
        """
        对外暴露接口：返回最终标准术语。

        当前默认 use_rerank=False：
        - 你现在更关注召回率
        - 直接返回召回第一名，避免 CrossEncoder 把召回顶掉

        后续如果你要恢复两阶段：
        retriever.get_standard_term(query, use_rerank=True)
        """
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
        else:
            best_match = candidates[0]
            final_term = best_match.get("term", "未知术语")
            final_score = best_match.get("score", None)

            print(
                f"当前use_rerank为{use_rerank}| "
                f"[Retriever] 跳过 rerank，直接使用召回第一名 | "
                f"query={query} | final_term={final_term} | final_score={final_score}"
            )
            print("=" * 80 + "\n")

        return final_term


# === 独立测试代码 ===
if __name__ == "__main__":
    retriever = MedicalRetriever()

    print("--- 🔬 开始测试 [多向量定义级语义匹配] 检索流水线 ---")
    while True:
        query = input("\n请输入患者原声 (输入 q 退出): ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        candidates = retriever.retrieve(query, top_k=5, raw_top_k=30)

        print(f"🗣️ [患者原声]: {query}")
        print("   ┣ 🔍 [第一阶段 - term 聚合后候选 (Top-5)]:")
        for cand in candidates:
            print(
                f"   ┃    - {cand['term']} | "
                f"score={cand['score']:.4f} | "
                f"hit={cand['matched_text'][:40]}..."
            )

        final_term = retriever.get_standard_term(
            query=query,
            top_k=5,
            raw_top_k=30,
            use_rerank=True
        )

        print(f"   ┗ ✅ [最终输出]: >>> {final_term} <<<")
