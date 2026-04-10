
import argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenRerankerWithLoRA:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        max_length: int = 2048,
        batch_size: int = 4,
        instruction: str = (
            "Given a patient's colloquial symptom description, judge whether "
            "the medical term and its definition match the patient's meaning."
        ),
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        ).eval()

        self.model = PeftModel.from_pretrained(base, adapter_path).eval()

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
    def score(self, query: str, documents: list) -> list:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--documents", nargs="+", required=True)
    args = parser.parse_args()

    scorer = QwenRerankerWithLoRA(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
    )
    scores = scorer.score(args.query, args.documents)

    for doc, score in sorted(zip(args.documents, scores), key=lambda x: x[1], reverse=True):
        print(f"{score:.6f}\t{doc}")


if __name__ == "__main__":
    main()
