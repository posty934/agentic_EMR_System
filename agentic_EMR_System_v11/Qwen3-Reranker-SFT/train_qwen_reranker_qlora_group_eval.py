import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


SYSTEM_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n'
    '<|im_start|>user\n'
)

ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def format_pair(instruction: str, query: str, document: str) -> str:
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


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


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for p in output_path.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
            checkpoints.append((step, p))
        except ValueError:
            continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


class PairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: List[Dict],
        tokenizer,
        instruction: str,
        max_length: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_length = max_length
        self.prefix_ids = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(ASSISTANT_SUFFIX, add_special_tokens=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        query = row["query"]
        document = row["document"]
        answer = str(row["label"]).strip().lower()

        if answer not in ("yes", "no"):
            raise ValueError(f"label 必须是 yes/no，收到: {answer}")

        body_text = format_pair(self.instruction, query, document)
        body_ids = self.tokenizer.encode(body_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id 为空，无法构造训练样本。")
        eos_ids = [self.tokenizer.eos_token_id]

        max_body_len = (
            self.max_length
            - len(self.prefix_ids)
            - len(self.suffix_ids)
            - len(answer_ids)
            - len(eos_ids)
        )
        if max_body_len <= 0:
            raise ValueError("max_length 太小，装不下模板与答案。")

        body_ids = body_ids[:max_body_len]

        input_ids = self.prefix_ids + body_ids + self.suffix_ids + answer_ids + eos_ids
        labels = (
            [-100] * (len(self.prefix_ids) + len(body_ids) + len(self.suffix_ids))
            + answer_ids
            + eos_ids
        )
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id 为空，无法进行 padding。")

        max_len = max(len(x["input_ids"]) for x in features)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feat in features:
            input_ids = feat["input_ids"]
            labels = feat["labels"]
            attention_mask = feat["attention_mask"]

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


class PrettyTrainCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        train_loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch")

        if train_loss is not None:
            msg = f"[Train] step={state.global_step}"
            if epoch is not None:
                msg += f" | epoch={epoch:.2f}"
            msg += f" | loss={train_loss:.6f}"
            if lr is not None:
                msg += f" | lr={lr:.2e}"
            print(msg)

        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            msg = f"[EvalLoss] step={state.global_step} | eval_loss={eval_loss:.6f}"
            print(msg)


class GroupRerankEvaluator:
    def __init__(
        self,
        tokenizer,
        instruction: str,
        max_length: int,
        yes_token_id: int,
        no_token_id: int,
        batch_size: int = 8,
    ):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_length = max_length
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.batch_size = batch_size

        self.prefix_ids = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(ASSISTANT_SUFFIX, add_special_tokens=False)

    def _build_prompt_ids(self, query: str, document: str) -> List[int]:
        body_text = format_pair(self.instruction, query, document)
        body_ids = self.tokenizer.encode(body_text, add_special_tokens=False)

        max_body_len = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        if max_body_len <= 0:
            raise ValueError("max_length 太小，装不下 prompt。")

        body_ids = body_ids[:max_body_len]
        return self.prefix_ids + body_ids + self.suffix_ids

    def _pad_batch(self, batch_input_ids: List[List[int]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id 为空。")

        max_len = max(len(x) for x in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []

        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

    @torch.no_grad()
    def score_documents(self, model, query: str, documents: List[str]) -> List[float]:
        device = next(model.parameters()).device
        scores = []

        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start:start + self.batch_size]
            batch_input_ids = [self._build_prompt_ids(query, doc) for doc in batch_docs]
            inputs = self._pad_batch(batch_input_ids)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits

            # 训练 tokenizer 是 right padding，不能直接 logits[:, -1, :]
            # 必须取每条样本最后一个非 pad token 的位置
            last_token_pos = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_indices, last_token_pos, :]

            yes_logits = last_logits[:, self.yes_token_id]
            no_logits = last_logits[:, self.no_token_id]

            batch_two_class_logits = torch.stack([no_logits, yes_logits], dim=1)
            batch_probs = torch.softmax(batch_two_class_logits, dim=1)[:, 1]
            scores.extend(batch_probs.detach().cpu().tolist())

        return scores

    @torch.no_grad()
    def evaluate_groups(self, model, group_rows: List[Dict]) -> Dict[str, float]:
        total_groups = 0
        top1_correct = 0
        reciprocal_rank_sum = 0.0
        pair_total = 0
        pair_correct = 0

        for row in group_rows:
            query = row["query"]
            candidates = row["candidates"]

            documents = [x["document"] for x in candidates]
            labels = [int(x["label"]) for x in candidates]

            scores = self.score_documents(model, query, documents)
            ranked = sorted(zip(documents, labels, scores), key=lambda x: x[2], reverse=True)

            total_groups += 1

            if ranked[0][1] == 1:
                top1_correct += 1

            first_positive_rank = None
            for rank_idx, (_, label, _) in enumerate(ranked, start=1):
                if label == 1:
                    first_positive_rank = rank_idx
                    break
            if first_positive_rank is not None:
                reciprocal_rank_sum += 1.0 / first_positive_rank

            for label, score in zip(labels, scores):
                pred = 1 if score >= 0.5 else 0
                pair_correct += int(pred == label)
                pair_total += 1

        top1_acc = top1_correct / total_groups if total_groups else 0.0
        mrr = reciprocal_rank_sum / total_groups if total_groups else 0.0
        pair_acc = pair_correct / pair_total if pair_total else 0.0

        return {
            "group_count": float(total_groups),
            "top1_acc": top1_acc,
            "mrr": mrr,
            "pair_acc_at_0.5": pair_acc,
        }


class GroupEvalCallback(TrainerCallback):
    def __init__(self, evaluator: GroupRerankEvaluator, valid_group_rows: List[Dict]):
        self.evaluator = evaluator
        self.valid_group_rows = valid_group_rows

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return

        was_training = model.training
        model.eval()

        group_metrics = self.evaluator.evaluate_groups(model, self.valid_group_rows)
        msg = (
            f"[GroupEval] step={state.global_step}"
            f" | groups={int(group_metrics['group_count'])}"
            f" | top1_acc={group_metrics['top1_acc']:.4f}"
            f" | mrr={group_metrics['mrr']:.4f}"
            f" | pair_acc@0.5={group_metrics['pair_acc_at_0.5']:.4f}"
        )
        print(msg)

        if was_training:
            model.train()


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"可训练参数: {trainable_params:,} | "
        f"总参数: {all_param:,} | "
        f"训练占比: {ratio:.4f}%"
    )


def build_training_args_kwargs(
    output_dir: str,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    seed: int,
):
    sig = inspect.signature(TrainingArguments.__init__)
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        dataloader_num_workers=0,
        load_best_model_at_end=False,
    )

    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="用 QLoRA 微调 Qwen reranker，并按 group top1/MRR 做验证。"
    )
    parser.add_argument("--base_model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--valid_group_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--group_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--instruction",
        default=(
            "Given a patient's colloquial symptom description, judge whether "
            "the medical term and its definition match the patient's meaning."
        ),
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id 仍为空，请检查 tokenizer 配置。")

    yes_token_ids = tokenizer("yes", add_special_tokens=False).input_ids
    no_token_ids = tokenizer("no", add_special_tokens=False).input_ids
    if len(yes_token_ids) != 1 or len(no_token_ids) != 1:
        raise ValueError(
            f"当前 tokenizer 下 yes/no 不是单 token。yes={yes_token_ids}, no={no_token_ids}"
        )
    yes_token_id = yes_token_ids[0]
    no_token_id = no_token_ids[0]

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    print_trainable_parameters(model)

    train_rows = load_jsonl_rows(args.train_file)
    valid_rows = load_jsonl_rows(args.valid_file)
    valid_group_rows = load_jsonl_rows(args.valid_group_file)

    if len(train_rows) == 0:
        raise ValueError(f"训练集为空: {args.train_file}")
    if len(valid_rows) == 0:
        raise ValueError(f"验证集为空: {args.valid_file}")
    if len(valid_group_rows) == 0:
        raise ValueError(f"group 验证集为空: {args.valid_group_file}")

    train_dataset = PairDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        instruction=args.instruction,
        max_length=args.max_length,
    )
    valid_dataset = PairDataset(
        rows=valid_rows,
        tokenizer=tokenizer,
        instruction=args.instruction,
        max_length=args.max_length,
    )
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    steps_per_epoch = math.ceil(len(train_dataset) / args.per_device_train_batch_size)
    update_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = math.ceil(args.num_train_epochs * update_steps_per_epoch)
    warmup_steps = int(max_train_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        **build_training_args_kwargs(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            seed=args.seed,
        )
    )

    evaluator = GroupRerankEvaluator(
        tokenizer=tokenizer,
        instruction=args.instruction,
        max_length=args.max_length,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        batch_size=args.group_eval_batch_size,
    )

    callbacks = [
        PrettyTrainCallback(),
        GroupEvalCallback(evaluator=evaluator, valid_group_rows=valid_group_rows),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            resume_path = find_latest_checkpoint(args.output_dir)
            if resume_path is None:
                raise ValueError(
                    f'你传了 --resume_from_checkpoint latest，但在 {args.output_dir} 下没找到 checkpoint-* 目录。'
                )
        else:
            resume_path = args.resume_from_checkpoint

        print(f"从 checkpoint 恢复训练: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，LoRA adapter 已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()