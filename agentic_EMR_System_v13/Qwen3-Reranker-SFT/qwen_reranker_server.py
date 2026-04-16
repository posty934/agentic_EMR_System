import os
from typing import List, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_name: str
    adapter_path: str


class QwenLoRAReranker:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        max_length: int = 1024,
        batch_size: int = 8,
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

        self.model = self._load_model()

    def _load_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA。这个服务建议部署在 GPU 机器上。")

        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
        except Exception:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
            base = base.to("cuda")

        model = PeftModel.from_pretrained(base, self.adapter_path).eval()
        return model

    def _get_model_device(self):
        return next(self.model.parameters()).device

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _build_inputs(self, pairs: List[tuple]):
        texts = [self._format_pair(query, doc) for query, doc in pairs]

        max_body_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_body_length <= 0:
            raise ValueError("max_length 过小，无法容纳 Qwen reranker 前后缀模板。")

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
    def score(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []

        all_scores: List[float] = []

        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start:start + self.batch_size]
            pairs = [(query, doc) for doc in batch_docs]
            inputs = self._build_inputs(pairs)

            logits = self.model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self.yes_token_id]
            no_logits = logits[:, self.no_token_id]

            batch_logits = torch.stack([no_logits, yes_logits], dim=1)
            batch_log_probs = torch.nn.functional.log_softmax(batch_logits, dim=1)
            batch_scores = batch_log_probs[:, 1].exp().detach().cpu().tolist()
            all_scores.extend(float(x) for x in batch_scores)

        return all_scores


BASE_MODEL = os.getenv(
    "QWEN_BASE_MODEL",
    "/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-Reranker-4B",
)
ADAPTER_PATH = os.getenv(
    "QWEN_ADAPTER_PATH",
    "/root/autodl-tmp/train_outputs/qwen_reranker_4b_lora_v3/checkpoint-200",
)
API_KEY = os.getenv("RERANKER_API_KEY", "").strip()
MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))

print("正在加载远程 Qwen LoRA 重排服务...")
print(f"BASE_MODEL={BASE_MODEL}")
print(f"ADAPTER_PATH={ADAPTER_PATH}")
print(f"MAX_LENGTH={MAX_LENGTH}, BATCH_SIZE={BATCH_SIZE}")

reranker = QwenLoRAReranker(
    base_model=BASE_MODEL,
    adapter_path=ADAPTER_PATH,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
)

app = FastAPI(title="Qwen LoRA Reranker Service")


def verify_api_key(authorization: Optional[str]):
    if not API_KEY:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")

    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_ready=True,
        model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank_api(
    req: RerankRequest,
    authorization: Optional[str] = Header(default=None),
):
    verify_api_key(authorization)

    query = (req.query or "").strip()
    documents = req.documents or []

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空。")
    if not isinstance(documents, list):
        raise HTTPException(status_code=400, detail="documents 必须是 list。")

    scores = reranker.score(query=query, documents=documents)
    return RerankResponse(scores=scores)