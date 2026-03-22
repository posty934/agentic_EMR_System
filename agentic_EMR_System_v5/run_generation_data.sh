#!/bin/bash

# ==========================================
# 自动化数据合成脚本 (Contrastive Learning - JSONL版)
# 请在 agentic_EMR_System 根目录下运行此脚本
# ==========================================

# 指向你脚本目录下的 python 文件
SCRIPT_PATH="scripts/generate_finetune_data.py"

# 循环次数：每次约生成90条，12次约1080条
TOTAL_RUNS=10
SLEEP_SECONDS=10

echo "======================================================="
echo "🚀 开始批量生成对比学习训练数据 (.jsonl)"
echo "🎯 目标生成数量：约 1000 条"
echo "🔄 计划运行轮数：$TOTAL_RUNS 轮 (每次间隔 $SLEEP_SECONDS 秒)"
echo "======================================================="

for ((i=1; i<=TOTAL_RUNS; i++))
do
    echo ""
    echo "▶️  [正在执行 第 $i / $TOTAL_RUNS 轮]..."

    # 运行 Python 脚本
    PYTHONIOENCODING=utf-8 python "$SCRIPT_PATH"

    # 检查上一条命令是否执行成功
    if [ $? -ne 0 ]; then
        echo "❌ 第 $i 轮执行失败！脚本已终止，请检查上方的 Python 报错。"
        exit 1
    fi

    # 如果不是最后一轮，则暂停一段时间保护 API
    if [ $i -lt $TOTAL_RUNS ]; then
        echo "⏳ 第 $i 轮顺利完成。休眠 $SLEEP_SECONDS 秒..."
        sleep $SLEEP_SECONDS
    fi
done

echo ""
echo "======================================================="
echo "🎉 全部 $TOTAL_RUNS 轮数据生成任务已圆满结束！"
echo "📊 当前 scripts/data 目录下的数据量："

# 统计 scripts/data 目录下的行数
if [ -f "scripts/data/train_contrastive.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < scripts/data/train_contrastive_old.jsonl 2>/dev/null)
    echo "   - 训练集 (train): $TRAIN_COUNT 条"
fi

if [ -f "scripts/data/test_contrastive.jsonl" ]; then
    TEST_COUNT=$(wc -l < scripts/data/test_contrastive_old.jsonl 2>/dev/null)
    echo "   - 验证集 (test):  $TEST_COUNT 条"
fi
echo "======================================================="


echo ""
read -p "数据生成完毕！请按回车键 (Enter) 关闭此窗口..."