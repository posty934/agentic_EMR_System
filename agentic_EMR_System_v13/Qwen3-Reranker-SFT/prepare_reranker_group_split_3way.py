import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
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


def save_jsonl(rows: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rows(rows: List[Dict]) -> List[Dict]:
    normalized = []
    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        pos_list = row.get("pos") or []
        neg_list = row.get("neg") or []

        if not query:
            continue
        if not isinstance(pos_list, list) or not isinstance(neg_list, list):
            raise ValueError(f"第 {idx} 条样本的 pos/neg 不是 list。")

        pos_docs = [x.strip() for x in pos_list if isinstance(x, str) and x.strip()]
        neg_docs = [x.strip() for x in neg_list if isinstance(x, str) and x.strip()]

        if not pos_docs or not neg_docs:
            continue

        pos_docs = list(dict.fromkeys(pos_docs))
        neg_docs = list(dict.fromkeys(neg_docs))

        # 去掉 pos/neg 重叠
        neg_docs = [x for x in neg_docs if x not in set(pos_docs)]

        if not pos_docs or not neg_docs:
            continue

        normalized.append(
            {
                "query": query,
                "pos": pos_docs,
                "neg": neg_docs,
            }
        )
    return normalized


def split_rows_three_way(
    rows: List[Dict],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if valid_ratio <= 0 or test_ratio <= 0 or (valid_ratio + test_ratio) >= 1:
        raise ValueError("valid_ratio 和 test_ratio 必须 > 0，且两者之和必须 < 1。")

    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    n = len(rows)
    valid_size = max(1, int(round(n * valid_ratio)))
    test_size = max(1, int(round(n * test_ratio)))
    train_size = n - valid_size - test_size

    if train_size <= 0:
        raise ValueError("切分后训练集为空，请调小 valid_ratio / test_ratio。")

    train_rows = rows[:train_size]
    valid_rows = rows[train_size:train_size + valid_size]
    test_rows = rows[train_size + valid_size:]

    return train_rows, valid_rows, test_rows


def build_pair_rows(rows: List[Dict], positive_repeat: int = 1) -> List[Dict]:
    pair_rows = []
    for row in rows:
        query = row["query"]
        pos_docs = row["pos"]
        neg_docs = row["neg"]

        for doc in pos_docs:
            for _ in range(max(1, positive_repeat)):
                pair_rows.append(
                    {
                        "query": query,
                        "document": doc,
                        "label": "yes",
                        "source_type": "positive",
                    }
                )

        for doc in neg_docs:
            pair_rows.append(
                {
                    "query": query,
                    "document": doc,
                    "label": "no",
                    "source_type": "negative",
                }
            )
    return pair_rows


def build_group_rows(
    rows: List[Dict],
    shuffle_candidates: bool = False,
    seed: int = 42,
    split_name: str = "valid",
) -> List[Dict]:
    rng = random.Random(seed)
    group_rows = []

    for i, row in enumerate(rows, start=1):
        query = row["query"]
        pos_docs = row["pos"]
        neg_docs = row["neg"]

        candidates = []
        for doc in pos_docs:
            candidates.append({"document": doc, "label": 1})
        for doc in neg_docs:
            candidates.append({"document": doc, "label": 0})

        if shuffle_candidates:
            rng.shuffle(candidates)

        group_rows.append(
            {
                "query": query,
                "candidates": candidates,
                "meta": {
                    "split": split_name,
                    "group_id": f"{split_name}_{i:05d}",
                    "positive_documents": pos_docs,
                },
            }
        )

    return group_rows


def print_summary(
    all_rows: List[Dict],
    train_rows: List[Dict],
    valid_rows: List[Dict],
    test_rows: List[Dict],
    train_pairs: List[Dict],
    valid_pairs: List[Dict],
    test_pairs: List[Dict],
    valid_groups: List[Dict],
    test_groups: List[Dict],
):
    print("=" * 80)
    print(f"原始 query 数: {len(all_rows)}")
    print(f"训练 query 数: {len(train_rows)}")
    print(f"验证 query 数: {len(valid_rows)}")
    print(f"测试 query 数: {len(test_rows)}")
    print("-" * 80)
    print(f"训练 pair 数: {len(train_pairs)}")
    print(f"验证 pair 数: {len(valid_pairs)}")
    print(f"测试 pair 数: {len(test_pairs)}")
    print("-" * 80)
    print(f"验证 group 数: {len(valid_groups)}")
    print(f"测试 group 数: {len(test_groups)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="按 query 级别做 train/valid/test 三切分，并生成 pair/group 文件。"
    )
    parser.add_argument("--input", required=True, help="输入 contrastive jsonl")
    parser.add_argument("--train_output", required=True, help="输出 train pair jsonl")
    parser.add_argument("--valid_output", required=True, help="输出 valid pair jsonl")
    parser.add_argument("--test_output", required=True, help="输出 test pair jsonl")
    parser.add_argument("--valid_group_output", required=True, help="输出 valid group jsonl")
    parser.add_argument("--test_group_output", required=True, help="输出 test group jsonl")
    parser.add_argument("--valid_ratio", type=float, default=0.076, help="验证集 query 比例")
    parser.add_argument("--test_ratio", type=float, default=0.076, help="测试集 query 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--positive_repeat",
        type=int,
        default=1,
        help="训练集中正样本重复次数。建议先用 1。",
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="是否打乱 valid/test 中每个 group 的候选顺序。",
    )
    args = parser.parse_args()

    raw_rows = load_jsonl(args.input)
    rows = normalize_rows(raw_rows)

    if not rows:
        raise ValueError("清洗后没有可用样本。")

    train_rows, valid_rows, test_rows = split_rows_three_way(
        rows=rows,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_pairs = build_pair_rows(train_rows, positive_repeat=args.positive_repeat)
    valid_pairs = build_pair_rows(valid_rows, positive_repeat=1)
    test_pairs = build_pair_rows(test_rows, positive_repeat=1)

    valid_groups = build_group_rows(
        valid_rows,
        shuffle_candidates=args.shuffle_candidates,
        seed=args.seed,
        split_name="valid",
    )
    test_groups = build_group_rows(
        test_rows,
        shuffle_candidates=args.shuffle_candidates,
        seed=args.seed + 1,
        split_name="test",
    )

    save_jsonl(train_pairs, args.train_output)
    save_jsonl(valid_pairs, args.valid_output)
    save_jsonl(test_pairs, args.test_output)
    save_jsonl(valid_groups, args.valid_group_output)
    save_jsonl(test_groups, args.test_group_output)

    print_summary(
        all_rows=rows,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        test_pairs=test_pairs,
        valid_groups=valid_groups,
        test_groups=test_groups,
    )

    print("输出文件：")
    print(f"  train_pairs : {args.train_output}")
    print(f"  valid_pairs : {args.valid_output}")
    print(f"  test_pairs  : {args.test_output}")
    print(f"  valid_group : {args.valid_group_output}")
    print(f"  test_group  : {args.test_group_output}")


if __name__ == "__main__":
    main()