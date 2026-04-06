import os
import json
from collections import Counter

import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


def _load_raw_test_cases():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    project_root = os.path.dirname(base_dir)
    eval_cases_path = os.path.join(project_root, "scripts", "data", "reranker_eval_set.json")

    with open(eval_cases_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    return [(item["query"], item["term"]) for item in eval_cases]


RAW_TEST_CASES = _load_raw_test_cases()

TEST_CASES = [{"query": query, "term": term} for query, term in RAW_TEST_CASES]


class QwenReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        max_length: int = 2048,
        batch_size: int = 4,
        instruction: str = (
            "Given a patient's colloquial symptom description, judge whether "
            "the medical term and its definition match the patient's meaning."
        ),
    ):
        print("⏳ 正在加载 Qwen3 重排模型 (Reranker)...")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model()

        self.token_false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.token_true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]

        self.prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n'
            '<|im_start|>user\n'
        )
        self.suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _load_model(self):
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                ).eval()
                return model
            except Exception as e:
                print(f"⚠️ device_map='auto' 加载失败，回退到单卡 CUDA 模式: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                ).eval()
                model = model.to("cuda")
                return model

        print("⚠️ 未检测到 CUDA，将使用 CPU 加载 Qwen3-Reranker-4B，速度会较慢且内存占用较高。")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        return model

    def _get_model_device(self):
        return next(self.model.parameters()).device

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _build_inputs(self, pairs: list):
        texts = [self._format_pair(query, doc) for query, doc in pairs]

        max_body_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_body_length <= 0:
            raise ValueError("max_length 过小，无法容纳 Qwen reranker 的前后缀模板。")

        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_body_length,
        )

        for i, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + token_ids + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )

        device = self._get_model_device()
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        return inputs

    @torch.no_grad()
    def score(self, query: str, documents: list) -> list:
        if not documents:
            return []

        all_scores = []

        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start:start + self.batch_size]
            pairs = [(query, doc) for doc in batch_docs]
            inputs = self._build_inputs(pairs)

            logits = self.model(**inputs).logits[:, -1, :]
            true_vector = logits[:, self.token_true_id]
            false_vector = logits[:, self.token_false_id]

            batch_logits = torch.stack([false_vector, true_vector], dim=1)
            batch_log_probs = torch.nn.functional.log_softmax(batch_logits, dim=1)
            batch_scores = batch_log_probs[:, 1].exp().detach().cpu().tolist()

            all_scores.extend(batch_scores)

        return all_scores


class MedicalRetriever:
    def __init__(self, symptoms_path: str = None):
        print("⏳ 正在加载向量召回模型 (Bi-Encoder)...")
        self.encoder = SentenceTransformer("BAAI/bge-base-zh")

        self.reranker = QwenReranker(
            model_name="Qwen/Qwen3-Reranker-4B",
            max_length=2048,
            batch_size=4,
            instruction=(
                "Given a patient's colloquial symptom description, judge whether "
                "the medical term and its definition match the patient's meaning."
            ),
        )

        self.symptoms_path = symptoms_path or self._resolve_symptoms_path()
        self.knowledge_base = self._load_medical_dictionary()
        self.term_set = {item["term"] for item in self.knowledge_base}

        self._validate_test_cases(TEST_CASES)

        self.index_texts = [f"{item['term']}：{item['desc']}" for item in self.knowledge_base]

        self.index = None
        self._build_index()

        print(f"✅ 检索-重排模块加载完毕！共索引 {len(self.knowledge_base)} 条医学定义。")
        print(f"📚 症状库路径: {self.symptoms_path}")
        print(f"🧪 已加载高难度测试样本: {len(TEST_CASES)} 条\n")

    def _get_base_dir(self):
        if "__file__" in globals():
            return os.path.dirname(os.path.abspath(__file__))
        return os.getcwd()

    def _resolve_symptoms_path(self) -> str:
        base_dir = self._get_base_dir()
        candidates = [
            os.environ.get("SYMPTOMS_JSON"),
            "/kaggle/input/datasets/pposty/symptoms/symptoms.json",
            os.path.join(base_dir, "symptoms.json"),
            "/mnt/data/symptoms.json",
        ]

        for path in candidates:
            if path and os.path.exists(path):
                return path

        raise FileNotFoundError(
            "未找到 symptoms.json，请将其放在脚本同目录、/mnt/data/symptoms.json、"
            "Kaggle 数据集路径下，或设置环境变量 SYMPTOMS_JSON。"
        )

    def _load_medical_dictionary(self) -> list:
        with open(self.symptoms_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        return dictionary

    def _validate_test_cases(self, test_cases: list):
        invalid = [case for case in test_cases if case["term"] not in self.term_set]
        if invalid:
            bad_terms = sorted(set(case["term"] for case in invalid))
            raise ValueError(f"测试集里存在症状库中没有的标签: {bad_terms}")

    def _build_index(self):
        embeddings = self.encoder.encode(self.index_texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query: str, top_k: int = 5, verbose: bool = True) -> list:
        top_k = min(top_k, len(self.knowledge_base))
        query_vector = self.encoder.encode([query])
        D, I = self.index.search(np.array(query_vector).astype("float32"), top_k)

        results = []
        debug_rows = []

        for rank, idx in enumerate(I[0]):
            if idx != -1 and idx < len(self.knowledge_base):
                item = self.knowledge_base[idx]
                results.append(item)
                debug_rows.append({
                    "rank": rank + 1,
                    "term": item["term"],
                    "distance": float(D[0][rank]),
                    "desc_preview": item["desc"][:40],
                })

        if verbose:
            print("\n" + "=" * 100)
            print(f"[Retriever] 原始查询: {query}")
            print("[Retriever] 第一阶段召回 TopK:")
            for row in debug_rows:
                print(
                    f"  Top{row['rank']}: term={row['term']} | "
                    f"distance={row['distance']:.4f} | desc={row['desc_preview']}..."
                )
            if not debug_rows:
                print("[Retriever] 第一阶段召回为空")

        return results

    def rerank(self, query: str, candidates: list, verbose: bool = True) -> dict:
        if not candidates:
            if verbose:
                print(f"[Retriever] 第二阶段重排跳过，候选为空。query={query}")
            return {
                "term": "未知术语",
                "desc": "无匹配",
                "score": None,
                "ranked_candidates": [],
            }

        candidate_texts = [f"{cand['term']}：{cand['desc']}" for cand in candidates]
        scores = self.reranker.score(query, candidate_texts)

        scored_candidates = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["score"] = float(score)
            scored_candidates.append(item)

        ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
        best = dict(ranked_candidates[0])
        best["ranked_candidates"] = ranked_candidates

        if verbose:
            print(f"[Retriever] 第二阶段重排结果: query={query}")
            for idx, cand in enumerate(ranked_candidates, start=1):
                print(f"  Rank{idx}: term={cand['term']} | score={cand['score']:.4f}")
            print(
                f"[Retriever] 最终选择: term={best['term']} | "
                f"score={best['score']:.4f}"
            )
            print("=" * 100 + "\n")

        return best

    def predict(self, query: str, top_k: int = 5, verbose: bool = True) -> dict:
        if not query or len(query.strip()) < 2:
            return {
                "query": query,
                "final_term": query,
                "score": None,
                "best_match": {"term": query, "desc": "", "score": None, "ranked_candidates": []},
                "retrieval_candidates": [],
                "reranked_candidates": [],
            }

        retrieval_candidates = self.retrieve(query, top_k=top_k, verbose=verbose)
        best_match = self.rerank(query, retrieval_candidates, verbose=verbose)

        final_term = best_match.get("term", "未知术语")
        final_score = best_match.get("score", None)
        reranked_candidates = best_match.get("ranked_candidates", [])

        if verbose:
            print(
                f"[Retriever] predict 完成 | "
                f"query={query} | final_term={final_term} | final_score={final_score}"
            )

        return {
            "query": query,
            "final_term": final_term,
            "score": final_score,
            "best_match": best_match,
            "retrieval_candidates": retrieval_candidates,
            "reranked_candidates": reranked_candidates,
        }

    def get_standard_term(self, query: str, top_k: int = 5, verbose: bool = True) -> str:
        result = self.predict(query, top_k=top_k, verbose=verbose)
        return result["final_term"]

    def _print_error_group(self, title: str, errors: list, top_k: int, show_rerank: bool):
        print("\n" + "#" * 110)
        print(title)
        print("#" * 110)

        if not errors:
            print("无")
            return

        for err in errors:
            print("-" * 110)
            print(f"编号             : {err['index']}")
            print(f"用户原话         : {err['query']}")
            print(f"正确标签         : {err['expected_term']}")
            print(f"模型预测         : {err['pred_term']}")
            print(f"预测分数         : {err['pred_score']}")
            print(f"召回候选 Top{top_k} : {err['retrieval_candidate_terms']}")

            if show_rerank:
                print("重排排序 TopK    :")
                for rank, item in enumerate(err["reranked_candidates"], start=1):
                    print(f"  #{rank} {item['term']} | score={item['score']:.4f}")

    def run_test_suite(
        self,
        test_cases: list,
        top_k: int = 5,
        save_path: str = "benchmark_results.json",
    ) -> dict:
        total = len(test_cases)
        correct = 0
        recall_hit = 0

        retrieval_miss_errors = []
        rerank_errors = []
        all_results = []
        confusion_counter = Counter()

        print("\n" + "#" * 110)
        print(f"🚀 开始自动测试，共 {total} 个高难度样本 | top_k={top_k}")
        print("#" * 110)

        for idx, case in enumerate(test_cases, start=1):
            query = case["query"]
            expected_term = case["term"]

            result = self.predict(query, top_k=top_k, verbose=False)

            pred_term = result["final_term"]
            pred_score = result["score"]

            retrieval_candidates = result["retrieval_candidates"]
            reranked_candidates = result["reranked_candidates"]

            retrieval_candidate_terms = [item["term"] for item in retrieval_candidates]
            reranked_candidate_terms = [item["term"] for item in reranked_candidates]

            stage1_hit = expected_term in retrieval_candidate_terms
            final_ok = pred_term == expected_term

            if stage1_hit:
                recall_hit += 1
            if final_ok:
                correct += 1

            row = {
                "index": idx,
                "query": query,
                "expected_term": expected_term,
                "pred_term": pred_term,
                "pred_score": pred_score,
                "stage1_hit": stage1_hit,
                "final_ok": final_ok,
                "retrieval_candidate_terms": retrieval_candidate_terms,
                "reranked_candidates": reranked_candidates,
            }
            all_results.append(row)

            if not final_ok:
                confusion_counter[(expected_term, pred_term)] += 1

                if not stage1_hit:
                    retrieval_miss_errors.append(row)
                    fail_type = "召回失败"
                else:
                    rerank_errors.append(row)
                    fail_type = "重排失败"

                status = f"❌ {fail_type}"
            else:
                status = "✅ 正确"

            recall_status = "HIT" if stage1_hit else "MISS"
            print(
                f"[{idx:03d}/{total}] {status} | "
                f"Recall@{top_k}={recall_status} | "
                f"GT={expected_term} | Pred={pred_term} | Query={query}"
            )

        accuracy = correct / total if total else 0.0
        recall_at_k = recall_hit / total if total else 0.0
        rerank_accuracy_when_recalled = correct / recall_hit if recall_hit else 0.0

        summary = {
            "total": total,
            "top_k": top_k,
            "correct": correct,
            "final_accuracy": accuracy,
            "stage1_recall_at_k": recall_at_k,
            "retrieval_hit_count": recall_hit,
            "retrieval_miss_count": len(retrieval_miss_errors),
            "rerank_error_count": len(rerank_errors),
            "rerank_accuracy_when_recalled": rerank_accuracy_when_recalled,
            "confusions_top10": [
                {
                    "expected": gt,
                    "predicted": pred,
                    "count": cnt,
                }
                for (gt, pred), cnt in confusion_counter.most_common(10)
            ],
            "retrieval_miss_errors": retrieval_miss_errors,
            "rerank_errors": rerank_errors,
            "all_results": all_results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 110)
        print("📊 基准测试汇总")
        print("=" * 110)
        print(f"总样本数                         : {total}")
        print(f"第一阶段 Recall@{top_k}            : {recall_hit}/{total} = {recall_at_k:.2%}")
        print(f"最终 Top1 Accuracy               : {correct}/{total} = {accuracy:.2%}")
        print(f"召回失败数（正确标签不在 Top{top_k}）: {len(retrieval_miss_errors)}")
        print(f"重排失败数（已召回但 Top1 错）     : {len(rerank_errors)}")
        print(f"召回命中样本中的重排准确率         : {correct}/{recall_hit} = {rerank_accuracy_when_recalled:.2%}" if recall_hit else "召回命中样本中的重排准确率         : N/A")
        print(f"当前 Recall@{top_k} 下的理论准确率上限 : {recall_at_k:.2%}")
        print(f"结果已保存到                     : {save_path}")

        if confusion_counter:
            print("\n🔀 最常见混淆对 Top10：")
            for i, ((gt, pred), cnt) in enumerate(confusion_counter.most_common(10), start=1):
                print(f"  {i:02d}. GT={gt} -> Pred={pred} | {cnt} 次")
        else:
            print("\n🔀 最常见混淆对 Top10：无错误，无混淆。")

        self._print_error_group(
            title=f"A. 召回失败样例（正确标签根本没进 Top{top_k}）",
            errors=retrieval_miss_errors,
            top_k=top_k,
            show_rerank=False,
        )

        self._print_error_group(
            title=f"B. 重排失败样例（正确标签已在 Top{top_k}，但重排没把它排到 Top1）",
            errors=rerank_errors,
            top_k=top_k,
            show_rerank=True,
        )

        return summary


def interactive_demo(retriever: MedicalRetriever, top_k: int = 5):
    print("\n--- 🔬 进入手动测试模式 (输入 q 退出) ---")
    while True:
        query = input("\n请输入患者原声: ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        result = retriever.predict(query, top_k=top_k, verbose=True)
        print(f"🗣️ [患者原声]: {query}")
        print("   ┣ 🔍 [第一阶段 - 召回候选]:")
        for cand in result["retrieval_candidates"]:
            print(f"   ┃    - {cand['term']} ({cand['desc'][:18]}...)")
        print(f"   ┗ ✅ [最终输出]: >>> {result['final_term']} <<<")


if __name__ == "__main__":
    retriever = MedicalRetriever()

    benchmark_top_k = int(os.getenv("BENCHMARK_TOP_K", "5"))
    benchmark_save_path = os.getenv("BENCHMARK_SAVE_PATH", "benchmark_results.json")

    retriever.run_test_suite(
        test_cases=TEST_CASES,
        top_k=benchmark_top_k,
        save_path=benchmark_save_path,
    )

    # 如果你还想在自动测试结束后继续手动输入测试，把这个环境变量设为 1
    # 例如：RUN_INTERACTIVE=1 python your_file.py
    if os.getenv("RUN_INTERACTIVE", "0") == "1":
        interactive_demo(retriever, top_k=benchmark_top_k)
