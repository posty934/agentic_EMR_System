import os
import csv
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


# === 1. 定义大模型输出的数据结构 ===
class SyntheticSample(BaseModel):
    patient_query: str = Field(description="患者的口语化表达")
    standard_term: str = Field(description="对应的标准医学术语")
    label: float = Field(description="标签：1.0表示医学语义一致（正样本），0.0表示不一致（负样本）")
    sample_type: str = Field(description="样本类型：正样本 / 普通负样本 / 困难负样本")


class BatchSynthesisResult(BaseModel):
    samples: List[SyntheticSample] = Field(description="生成的数据集样本列表")


def generate_digestive_dataset():
    print("🚀 启动消化内科微调数据合成引擎...\n")

    # 初始化大模型 (请确保 .env 里的配置是正确的)
    llm = ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0.7,  # 稍微调高温度，让大模型生成的口语更多样化
    )

    parser = JsonOutputParser(pydantic_object=BatchSynthesisResult)

    # === 2. 核心提示词工程 (指导大模型如何造数据) ===
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的医疗NLP数据标注专家。
你的任务是针对给定的【消化内科标准术语】，生成用于微调重排模型（Reranker）的对比学习数据集。

对于给定的每一个标准术语，你必须严格生成以下 7 个样本：
1. 【正样本】(3个，label=1.0)：极度口语化、接地气的患者表达，甚至可以带点方言色彩，但医学本质上完全等同于该标准术语。
2. 【普通负样本】(2个，label=0.0)：同属消化科，但与该术语完全不同的其他症状（用于拉开基础距离）。
3. 【困难负样本】(2个，label=0.0)：字面非常相似，或者容易混淆，但医学病理上**绝对不同**的表达。这是最重要的！(例如：针对"反酸"，困难负样本可以是"打嗝"或"呕吐"，它们相关但不等同)。

必须严格遵守以下JSON格式输出：
{format_instructions}"""),
        ("human", "请为标准术语【{term}】生成 7 个训练样本。")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    # === 3. 准备消化科核心术语表 ===
    # 为了测试，我们先跑 5 个核心词。
    # 等跑通后，你可以往这里加 50 个词，让它跑一晚上，你的 5000 条数据集就有了！
    digestive_terms = [
        "反酸",
        "烧心",
        "腹痛",
        "腹泻",
        "恶心"
    ]

    output_file = "data/digestive_finetune_data.csv"
    os.makedirs("data", exist_ok=True)  # 确保 data 文件夹存在

    # === 4. 批量生成并写入 CSV ===
    print(f"📄 准备将生成的数据写入: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["patient_query", "standard_term", "label", "sample_type"])

        for term in digestive_terms:
            print(f"⏳ 正在呼叫大模型生成【{term}】的训练数据...")
            try:
                result = chain.invoke({"term": term})
                samples = result.get("samples", [])

                for sample in samples:
                    writer.writerow([
                        sample.get("patient_query"),
                        sample.get("standard_term"),
                        sample.get("label"),
                        sample.get("sample_type")
                    ])
                print(f"   ✅ 成功生成 {len(samples)} 条样本。")
            except Exception as e:
                print(f"   ❌ 生成【{term}】时出错: {e}")

    print(f"\n🎉 全部完成！请去检查 {output_file} 看看大模型的工作成果吧！")


if __name__ == "__main__":
    generate_digestive_dataset()