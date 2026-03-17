import os
import csv
import math
import random
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader


def load_dataset_from_csv(csv_path):
    """从 CSV 文件读取训练数据并转化为 InputExample 格式"""
    examples = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据集文件：{csv_path}。请先运行生成脚本！")

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # 按字典格式读取表头
        for row in reader:
            query = row['patient_query']
            term = row['standard_term']
            label = float(row['label'])  # 将字符串 "1.0" 转化为浮点数

            # CrossEncoder 需要的格式：InputExample(texts=[句1, 句2], label=分数)
            examples.append(InputExample(texts=[query, term], label=label))

    return examples


def train_medical_reranker():
    print("🚀 开始初始化【消化内科】专属重排模型微调流程...\n")

    # === 1. 加载预训练基座模型 ===
    model_name = 'hfl/chinese-roberta-wwm-ext'
    print(f"📦 正在加载预训练基座模型: {model_name}")
    model = CrossEncoder(model_name, num_labels=1, max_length=128)

    # === 2. 加载并切分 CSV 数据集 ===
    csv_path = "data/digestive_finetune_data.csv"
    print(f"📊 正在从 {csv_path} 读取微调数据...")
    all_examples = load_dataset_from_csv(csv_path)

    # 打乱数据，防止模型记住顺序
    random.seed(42)
    random.shuffle(all_examples)

    # 按 9:1 划分训练集和验证集
    split_idx = int(len(all_examples) * 0.9)
    train_data = all_examples[:split_idx]
    dev_data = all_examples[split_idx:]

    print(f"✅ 成功加载 {len(all_examples)} 条数据！")
    print(f"   ┣ 🏋️ 训练集: {len(train_data)} 条")
    print(f"   ┗ 📏 验证集: {len(dev_data)} 条\n")

    # === 3. 配置 DataLoader 和 Evaluator ===
    # batch_size 设为 4，绝大多数电脑都能跑得动
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=4)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_data, name='digestive-dev')

    # === 4. 设置训练超参数 ===
    num_epochs = 4
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    # 我们把输出文件夹改个霸气的名字，代表它是你专属炼制的
    output_dir = './finetuned_digestive_reranker'

    print("⚙️ 训练参数设置完毕：")
    print(f" - Epochs: {num_epochs}")
    print(f" - Batch Size: 4")
    print(f" - 模型保存路径: {output_dir}\n")

    # === 5. 启动训练 ===
    print("🔥 开始训练! (观察 Loss 下降是深度学习最爽的时刻)...\n")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=10,  # 每隔10步用验证集评估一次
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True
    )

    print(f"\n🎉 训练圆满完成！你专属的消化内科重排模型已保存至: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    train_medical_reranker()