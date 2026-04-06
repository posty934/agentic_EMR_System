import os
import json
import random
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ==========================================
# 动态路径配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SYMPTOMS_FILE = os.path.join(PROJECT_ROOT, "knowledge", "symptoms.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_contrastive_old.jsonl")
TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_contrastive_old.jsonl")


class ContrastiveSample(BaseModel):
    query: str = Field(description="患者极其口语化的表达")
    pos: List[str] = Field(description="正样本列表（包含1个正确术语），格式为 '术语：定义'")
    neg: List[str] = Field(description="硬负样本列表（2到3个极具迷惑性但错误的术语），格式为 '术语：定义'")


class BatchSynthesisResult(BaseModel):
    samples: List[ContrastiveSample] = Field(description="生成的对比学习样本列表")


def generate_contrastive_dataset():
    print("🚀 启动消化内科 [对比学习] 微调数据合成引擎 (去废话纯净版)...\n")

    llm = ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0.7,
    )

    parser = JsonOutputParser(pydantic_object=BatchSynthesisResult)

    # === 2. 核心 Prompt (已加入极其严厉的去废话指令) ===
    system_prompt = """
你是一个高级医学NLP数据专家，专门负责构建用于【对比学习(Contrastive Learning)】的重排模型训练集。

你的任务是为我提供的【目标医学术语】，生成极具挑战性的训练数据。
我会提供给你一个【全局消化科术语词典】（共45个症状）作为你的负样本备选池。

【数据生成规则】
针对每个传入的目标术语，请生成 2 条不同的患者 query。
每条 query 必须包含：
1. query: 患者真实、口语化的主诉和感受。⚠️【绝对禁令】：严禁以“医生”、“大夫”、“你好”、“请问”等任何称呼或问候语开头！直接切入主题描述症状。绝不能直接说出医学术语。
2. pos: 1个正确的正样本（当前目标术语及定义）。
3. neg: 2-3个【硬负例（Hard Negatives）】。硬负例必须从【全局词典】中挑选，它们必须在字面、发病部位或大白话表达上与query高度相似，但在临床语义上是错误的！

✅ 优秀硬负例参考：
- query: "拉出来的屎像铅笔一样细，感觉肠子被什么东西堵住了似的。"
- pos: ["粪便变细：排出的成形大便直径明显变小，呈铅笔状或细条状，常提示肠道狭窄。"]
- neg: ["便秘：排便次数减少或排便困难，粪便干硬。", "里急后重：下腹部窘迫，有强烈的便意但排便不尽..."]

【输出格式要求】
严格输出 JSON 格式，不要有任何 Markdown 标记或多余解释：
{{
  "samples": [
    {{
      "query": "...",
      "pos": ["术语：定义"],
      "neg": ["错词1：定义1", "错词2：定义2"]
    }}
  ]
}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
【全局消化科术语词典】（请仅从这里挑选 pos 和 neg）：
{all_terms_json}

=======================================

请【只为以下这批目标术语】生成数据（每个术语生成2条不同query）：
{target_terms_json}
""")
    ])

    chain = prompt | llm | parser

    # === 3. 加载术语库 ===
    try:
        with open(SYMPTOMS_FILE, "r", encoding="utf-8") as f:
            digestive_terms = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到症状字典文件: {SYMPTOMS_FILE}")
        return

    all_terms_str = "\n".join([f"- {item['term']}：{item['desc']}" for item in digestive_terms])

    chunk_size = 3
    term_chunks = [digestive_terms[i:i + chunk_size] for i in range(0, len(digestive_terms), chunk_size)]

    all_samples = []

    print(f"📦 共加载 {len(digestive_terms)} 个术语，分为 {len(term_chunks)} 批进行生成...")

    for i, chunk in enumerate(term_chunks):
        print(f"⏳ 正在生成第 {i + 1}/{len(term_chunks)} 批数据...")
        target_terms_json = json.dumps(chunk, ensure_ascii=False)
        try:
            result = chain.invoke({
                "all_terms_json": all_terms_str,
                "target_terms_json": target_terms_json
            })
            samples = result.get("samples", [])
            all_samples.extend(samples)
            print(f"   -> 成功获取 {len(samples)} 个 Contrastive Query 组")
        except Exception as e:
            print(f"❌ 第 {i + 1} 批数据生成出错: {e}")

    if not all_samples:
        print("⚠️ 未生成任何数据，程序退出。")
        return

    # === 4. 保存为对比学习标准的 JSONL 格式 (追加模式) ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    def append_to_jsonl(data_list, filename):
        with open(filename, mode='a', encoding='utf-8') as f:
            for sample in data_list:
                if hasattr(sample, "model_dump"):
                    json_str = json.dumps(sample.model_dump(), ensure_ascii=False)
                else:
                    json_str = json.dumps(sample, ensure_ascii=False)
                f.write(json_str + "\n")

    append_to_jsonl(train_data, TRAIN_OUTPUT_FILE)
    append_to_jsonl(test_data, TEST_OUTPUT_FILE)

    print(f"\n🎉 写入完成！本次共新增 {len(all_samples)} 条对比学习数据。")
    print(f"📁 训练集: {TRAIN_OUTPUT_FILE} (新增 {len(train_data)} 条)")
    print(f"📁 验证集: {TEST_OUTPUT_FILE} (新增 {len(test_data)} 条)")


if __name__ == "__main__":
    # Windows 环境下强制输出 utf-8 防止终端报错
    import sys

    sys.stdout.reconfigure(encoding='utf-8')
    generate_contrastive_dataset()