import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n'
    '<|im_start|>user\n'
)

ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def load_jsonl_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {e}") from e
    return rows


def format_pair(instruction: str, query: str, document: str) -> str:
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


class QwenRerankerScorer:
    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 8,
        instruction: str = (
            "Given a patient's colloquial symptom description, judge whether "
            "the medical term and its definition match the patient's meaning."
        ),
        load_in_4bit: bool = False,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction
        self.load_in_4bit = load_in_4bit

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id 为空。")

        yes_ids = self.tokenizer("yes", add_special_tokens=False).input_ids
        no_ids = self.tokenizer("no", add_special_tokens=False).input_ids
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f"当前 tokenizer 下 yes/no 不是单 token。yes={yes_ids}, no={no_ids}"
            )
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]

        self.prefix_tokens = self.tokenizer.encode(
            SYSTEM_PREFIX,
            add_special_tokens=False,
        )
        self.suffix_tokens = self.tokenizer.encode(
            ASSISTANT_SUFFIX,
            add_special_tokens=False,
        )

        self.model = self._load_model()

    def _load_model(self):
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        model_kwargs = dict(
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["torch_dtype"] = compute_dtype
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs,
        ).eval()

        if self.adapter_path:
            model = PeftModel.from_pretrained(base, self.adapter_path).eval()
        else:
            model = base

        return model

    def unload(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_model_device(self):
        return next(self.model.parameters()).device

    def _build_inputs(self, pairs: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        texts = [format_pair(self.instruction, q, d) for q, d in pairs]

        max_body_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_body_length <= 0:
            raise ValueError("max_length 太小，装不下 prompt。")

        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_body_length,
        )

        for i, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + token_ids + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        device = self._get_model_device()
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        return inputs

    @torch.no_grad()
    def score_documents(self, query: str, documents: List[str]) -> List[float]:
        all_scores = []
        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start:start + self.batch_size]
            pairs = [(query, doc) for doc in batch_docs]
            inputs = self._build_inputs(pairs)

            logits = self.model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self.yes_token_id]
            no_logits = logits[:, self.no_token_id]

            two_class_logits = torch.stack([no_logits, yes_logits], dim=1)
            probs = torch.softmax(two_class_logits, dim=1)[:, 1]
            all_scores.extend(probs.detach().cpu().tolist())

        return all_scores


def evaluate_group_file(
    scorer: QwenRerankerScorer,
    group_rows: List[Dict],
) -> Dict:
    total_groups = 0
    top1_correct = 0
    reciprocal_rank_sum = 0.0

    pair_total = 0
    pair_correct = 0
    tp = tn = fp = fn = 0

    details = []

    for row_idx, row in enumerate(group_rows, start=1):
        query = row["query"]
        candidates = row["candidates"]

        documents = [x["document"] for x in candidates]
        labels = [int(x["label"]) for x in candidates]

        scores = scorer.score_documents(query, documents)
        ranked = sorted(
            [
                {
                    "document": doc,
                    "label": label,
                    "score": float(score),
                }
                for doc, label, score in zip(documents, labels, scores)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        total_groups += 1

        top1_hit = int(ranked[0]["label"] == 1)
        top1_correct += top1_hit

        rr = 0.0
        first_positive_rank = None
        for rank_idx, item in enumerate(ranked, start=1):
            if item["label"] == 1:
                first_positive_rank = rank_idx
                rr = 1.0 / rank_idx
                break
        reciprocal_rank_sum += rr

        for label, score in zip(labels, scores):
            pred = 1 if score >= 0.5 else 0
            pair_correct += int(pred == label)
            pair_total += 1

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 0 and label == 0:
                tn += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1

        details.append(
            {
                "row_idx": row_idx,
                "query": query,
                "top1_hit": top1_hit,
                "first_positive_rank": first_positive_rank,
                "ranked_candidates": ranked,
            }
        )

    top1_acc = top1_correct / total_groups if total_groups else 0.0
    mrr = reciprocal_rank_sum / total_groups if total_groups else 0.0
    pair_acc = pair_correct / pair_total if pair_total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "group_count": total_groups,
        "top1_acc": top1_acc,
        "mrr": mrr,
        "pair_acc_at_0.5": pair_acc,
        "pair_precision_at_0.5": precision,
        "pair_recall_at_0.5": recall,
        "pair_f1_at_0.5": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "details": details,
    }


def build_disagreement_report(
    base_result: Dict,
    lora_result: Dict,
) -> Dict:
    base_details = {x["row_idx"]: x for x in base_result["details"]}
    lora_details = {x["row_idx"]: x for x in lora_result["details"]}

    lora_fix_cases = []
    lora_worse_cases = []
    both_wrong_cases = []

    all_row_ids = sorted(base_details.keys())
    for row_id in all_row_ids:
        b = base_details[row_id]
        l = lora_details[row_id]

        if b["top1_hit"] == 0 and l["top1_hit"] == 1:
            lora_fix_cases.append(
                {
                    "row_idx": row_id,
                    "query": b["query"],
                    "base_first_positive_rank": b["first_positive_rank"],
                    "lora_first_positive_rank": l["first_positive_rank"],
                    "base_top3": b["ranked_candidates"][:3],
                    "lora_top3": l["ranked_candidates"][:3],
                }
            )
        elif b["top1_hit"] == 1 and l["top1_hit"] == 0:
            lora_worse_cases.append(
                {
                    "row_idx": row_id,
                    "query": b["query"],
                    "base_first_positive_rank": b["first_positive_rank"],
                    "lora_first_positive_rank": l["first_positive_rank"],
                    "base_top3": b["ranked_candidates"][:3],
                    "lora_top3": l["ranked_candidates"][:3],
                }
            )
        elif b["top1_hit"] == 0 and l["top1_hit"] == 0:
            both_wrong_cases.append(
                {
                    "row_idx": row_id,
                    "query": b["query"],
                    "base_first_positive_rank": b["first_positive_rank"],
                    "lora_first_positive_rank": l["first_positive_rank"],
                    "base_top3": b["ranked_candidates"][:3],
                    "lora_top3": l["ranked_candidates"][:3],
                }
            )

    return {
        "lora_fix_cases": lora_fix_cases,
        "lora_worse_cases": lora_worse_cases,
        "both_wrong_cases": both_wrong_cases,
    }


def print_summary(name: str, result: Dict):
    print(f"\n===== {name} =====")
    print(f"groups              : {result['group_count']}")
    print(f"top1_acc            : {result['top1_acc']:.4f}")
    print(f"mrr                 : {result['mrr']:.4f}")
    print(f"pair_acc@0.5        : {result['pair_acc_at_0.5']:.4f}")
    print(f"pair_precision@0.5  : {result['pair_precision_at_0.5']:.4f}")
    print(f"pair_recall@0.5     : {result['pair_recall_at_0.5']:.4f}")
    print(f"pair_f1@0.5         : {result['pair_f1_at_0.5']:.4f}")
    print(f"confusion_matrix    : TP={result['tp']} TN={result['tn']} FP={result['fp']} FN={result['fn']}")


def main():
    parser = argparse.ArgumentParser(description="Compare base model vs LoRA reranker on the same group eval file.")
    parser.add_argument("--base_model", required=True, help="基座模型目录")
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter 目录，可填最终输出目录或某个 checkpoint 目录")
    parser.add_argument("--group_file", required=True, help="group 验证集 jsonl")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--report_file", default=None, help="可选，保存详细对比结果 json")
    parser.add_argument(
        "--instruction",
        default=(
            "Given a patient's colloquial symptom description, judge whether "
            "the medical term and its definition match the patient's meaning."
        ),
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="推理时使用 4bit 量化，省显存")
    args = parser.parse_args()

    group_rows = load_jsonl_rows(args.group_file)
    if len(group_rows) == 0:
        raise ValueError(f"group_file 为空: {args.group_file}")

    print(f"加载 group 文件: {args.group_file}")
    print(f"group 数量: {len(group_rows)}")

    print("\n开始评估 Base model ...")
    base_scorer = QwenRerankerScorer(
        base_model=args.base_model,
        adapter_path=None,
        max_length=args.max_length,
        batch_size=args.batch_size,
        instruction=args.instruction,
        load_in_4bit=args.load_in_4bit,
    )
    base_result = evaluate_group_file(base_scorer, group_rows)
    print_summary("Base", base_result)
    base_scorer.unload()

    print("\n开始评估 LoRA model ...")
    lora_scorer = QwenRerankerScorer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        instruction=args.instruction,
        load_in_4bit=args.load_in_4bit,
    )
    lora_result = evaluate_group_file(lora_scorer, group_rows)
    print_summary("LoRA", lora_result)
    lora_scorer.unload()

    delta = {
        "delta_top1_acc": lora_result["top1_acc"] - base_result["top1_acc"],
        "delta_mrr": lora_result["mrr"] - base_result["mrr"],
        "delta_pair_acc_at_0.5": lora_result["pair_acc_at_0.5"] - base_result["pair_acc_at_0.5"],
        "delta_pair_f1_at_0.5": lora_result["pair_f1_at_0.5"] - base_result["pair_f1_at_0.5"],
    }

    print("\n===== Delta (LoRA - Base) =====")
    print(f"delta_top1_acc      : {delta['delta_top1_acc']:.4f}")
    print(f"delta_mrr           : {delta['delta_mrr']:.4f}")
    print(f"delta_pair_acc@0.5  : {delta['delta_pair_acc_at_0.5']:.4f}")
    print(f"delta_pair_f1@0.5   : {delta['delta_pair_f1_at_0.5']:.4f}")

    disagreement = build_disagreement_report(base_result, lora_result)
    print(f"\nLoRA 修正的 case 数量   : {len(disagreement['lora_fix_cases'])}")
    print(f"LoRA 变差的 case 数量   : {len(disagreement['lora_worse_cases'])}")
    print(f"两者都错的 case 数量   : {len(disagreement['both_wrong_cases'])}")

    if args.report_file:
        report = {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "group_file": args.group_file,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "instruction": args.instruction,
            "base_result": base_result,
            "lora_result": lora_result,
            "delta": delta,
            "disagreement": disagreement,
        }

        Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n详细报告已保存到: {args.report_file}")


if __name__ == "__main__":
    main()