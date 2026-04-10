import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


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

        normalized.append(
            {
                "query": query,
                "pos": pos_docs,
                "neg": neg_docs,
            }
        )
    return normalized


def split_rows_by_query(rows: List[Dict], valid_ratio: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    valid_size = max(1, int(len(rows) * valid_ratio))
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]
    return train_rows, valid_rows


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


def build_group_rows(rows: List[Dict], shuffle_candidates: bool = False, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    group_rows = []

    for row in rows:
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
                "positive_documents": pos_docs,
                "candidates": candidates,
            }
        )
    return group_rows


def save_jsonl(rows: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="按 query 切分 contrastive 数据，并生成 pair 训练集 + group 验证集。"
    )
    parser.add_argument("--input", required=True, help="输入 contrastive jsonl")
    parser.add_argument("--train_output", required=True, help="输出 train pair jsonl")
    parser.add_argument("--valid_output", required=True, help="输出 valid pair jsonl")
    parser.add_argument("--valid_group_output", required=True, help="输出 valid group jsonl")
    parser.add_argument("--valid_ratio", type=float, default=0.15, help="验证集 query 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--positive_repeat",
        type=int,
        default=1,
        help="正样本重复次数。先用 1，别默认复制到和负样本一样多。",
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="是否打乱验证集每个 group 内候选顺序。",
    )
    args = parser.parse_args()

    raw_rows = load_jsonl(args.input)
    rows = normalize_rows(raw_rows)

    train_rows, valid_rows = split_rows_by_query(
        rows,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    train_pairs = build_pair_rows(train_rows, positive_repeat=args.positive_repeat)
    valid_pairs = build_pair_rows(valid_rows, positive_repeat=1)
    valid_groups = build_group_rows(
        valid_rows,
        shuffle_candidates=args.shuffle_candidates,
        seed=args.seed,
    )

    save_jsonl(train_pairs, args.train_output)
    save_jsonl(valid_pairs, args.valid_output)
    save_jsonl(valid_groups, args.valid_group_output)

    print(f"原始 query 数: {len(rows)}")
    print(f"训练 query 数: {len(train_rows)}")
    print(f"验证 query 数: {len(valid_rows)}")
    print(f"训练 pair 数: {len(train_pairs)}")
    print(f"验证 pair 数: {len(valid_pairs)}")
    print(f"验证 group 数: {len(valid_groups)}")
    print(f"训练文件: {args.train_output}")
    print(f"验证 pair 文件: {args.valid_output}")
    print(f"验证 group 文件: {args.valid_group_output}")


if __name__ == "__main__":
    main()