import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Popularity baseline for GenRec-Agent."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="datasets/processed/train.jsonl",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default="datasets/processed/valid.jsonl",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="datasets/processed/test.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/processed/popularity_results.json",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
    )
    parser.add_argument(
        "--exclude_history",
        action="store_true",
        help="If set, remove items already in the user's history from recommendations.",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_popularity(train_rows: list[dict]) -> list[str]:
    counter = Counter()

    # 用 train 中的 target 统计热门商品。
    # 这里不直接用 interactions.csv，是为了严格避免 valid/test target 泄漏。
    for row in train_rows:
        target = str(row["target"])
        counter[target] += 1

    ranked_items = [item for item, _ in counter.most_common()]
    return ranked_items


def recommend_popular(
    ranked_items: list[str],
    history: Iterable[str],
    k: int,
    exclude_history: bool,
) -> list[str]:
    history_set = set(str(x) for x in history)

    recs = []
    for item in ranked_items:
        if exclude_history and item in history_set:
            continue
        recs.append(item)
        if len(recs) >= k:
            break

    return recs


def recall_at_k(recs: list[str], target: str, k: int) -> float:
    return 1.0 if target in recs[:k] else 0.0


def hitrate_at_k(recs: list[str], target: str, k: int) -> float:
    return 1.0 if target in recs[:k] else 0.0


def ndcg_at_k(recs: list[str], target: str, k: int) -> float:
    topk = recs[:k]
    if target not in topk:
        return 0.0

    rank = topk.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def evaluate_split(
    rows: list[dict],
    ranked_items: list[str],
    ks: list[int],
    exclude_history: bool,
) -> dict:
    metrics = {}

    for k in ks:
        recall_scores = []
        hit_scores = []
        ndcg_scores = []

        for row in rows:
            history = row["history"]
            target = str(row["target"])
            recs = recommend_popular(
                ranked_items=ranked_items,
                history=history,
                k=k,
                exclude_history=exclude_history,
            )

            recall_scores.append(recall_at_k(recs, target, k))
            hit_scores.append(hitrate_at_k(recs, target, k))
            ndcg_scores.append(ndcg_at_k(recs, target, k))

        n = max(len(rows), 1)
        metrics[f"Recall@{k}"] = sum(recall_scores) / n
        metrics[f"HitRate@{k}"] = sum(hit_scores) / n
        metrics[f"NDCG@{k}"] = sum(ndcg_scores) / n

    return metrics


def main():
    args = parse_args()

    print("📥 Loading train / valid / test splits...")
    train_rows = read_jsonl(args.train_path)
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)

    print(f"Train samples: {len(train_rows):,}")
    print(f"Valid samples: {len(valid_rows):,}")
    print(f"Test samples:  {len(test_rows):,}")

    print("🔥 Building popularity ranking from train targets...")
    ranked_items = build_popularity(train_rows)

    if not ranked_items:
        raise ValueError("No ranked items found from train data.")

    print(f"Popular items: {len(ranked_items):,}")
    print(f"Top 10 popular items: {ranked_items[:10]}")

    print("\n📊 Evaluating valid split...")
    valid_metrics = evaluate_split(
        rows=valid_rows,
        ranked_items=ranked_items,
        ks=args.ks,
        exclude_history=args.exclude_history,
    )

    print("📊 Evaluating test split...")
    test_metrics = evaluate_split(
        rows=test_rows,
        ranked_items=ranked_items,
        ks=args.ks,
        exclude_history=args.exclude_history,
    )

    results = {
        "baseline": "popularity",
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "test_path": args.test_path,
        "ks": args.ks,
        "exclude_history": args.exclude_history,
        "num_train_samples": len(train_rows),
        "num_valid_samples": len(valid_rows),
        "num_test_samples": len(test_rows),
        "num_popular_items": len(ranked_items),
        "valid": valid_metrics,
        "test": test_metrics,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n========== Popularity Baseline ==========")
    print("Valid:")
    for k, v in valid_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Test:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSaved results to: {output_path}")
    print("=========================================")


if __name__ == "__main__":
    main()