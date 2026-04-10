
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="将 LoRA adapter 合并回基座模型。")
    parser.add_argument("--base_model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"合并完成，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

