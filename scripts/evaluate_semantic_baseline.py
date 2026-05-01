import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic-ID neighbor baseline for GenRec-Agent."
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
        "--semantic_ids_path",
        type=str,
        default="datasets/processed/semantic_ids.json",
    )
    parser.add_argument(
        "--sid_to_items_path",
        type=str,
        default="datasets/processed/sid_to_items.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/processed/semantic_baseline_results.json",
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


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_popularity(train_rows: list[dict]) -> tuple[Counter, list[str]]:
    counter = Counter()
    for row in train_rows:
        counter[str(row["target"])] += 1

    ranked_items = [item for item, _ in counter.most_common()]
    return counter, ranked_items


def get_item_sid_key(item_id: str, semantic_ids: dict) -> str | None:
    sid = semantic_ids.get(str(item_id))
    if sid is None:
        return None

    # 当前第一版 semantic_id = [cluster_id]
    if isinstance(sid, list) and len(sid) > 0:
        return str(sid[0])

    return str(sid)


def recommend_semantic_neighbors(
    history: Iterable[str],
    semantic_ids: dict,
    sid_to_items: dict,
    popularity_counter: Counter,
    global_popular_items: list[str],
    k: int,
    exclude_history: bool,
) -> list[str]:
    history = [str(x) for x in history]
    history_set = set(history)

    # 按用户历史中 cluster 出现次数给 cluster 加权
    cluster_counter = Counter()
    for item in history:
        sid_key = get_item_sid_key(item, semantic_ids)
        if sid_key is not None:
            cluster_counter[sid_key] += 1

    candidate_scores = Counter()

    # 同 semantic cluster 召回候选
    for sid_key, cluster_weight in cluster_counter.most_common():
        items = sid_to_items.get(str(sid_key), [])
        for item in items:
            item = str(item)
            if exclude_history and item in history_set:
                continue

            # 分数 = cluster 匹配权重 + 商品训练集流行度小加权
            # 这样既利用 semantic id，又避免同 cluster 内完全随机排序
            pop = popularity_counter.get(item, 0)
            candidate_scores[item] += cluster_weight * 1000 + pop

    ranked_candidates = [
        item for item, _ in candidate_scores.most_common()
    ]

    # 不足 TopK 时，用全局热门商品补齐
    recs = []
    seen = set()

    for item in ranked_candidates:
        if item in seen:
            continue
        if exclude_history and item in history_set:
            continue
        recs.append(item)
        seen.add(item)
        if len(recs) >= k:
            return recs

    for item in global_popular_items:
        item = str(item)
        if item in seen:
            continue
        if exclude_history and item in history_set:
            continue
        recs.append(item)
        seen.add(item)
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
    semantic_ids: dict,
    sid_to_items: dict,
    popularity_counter: Counter,
    global_popular_items: list[str],
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

            recs = recommend_semantic_neighbors(
                history=history,
                semantic_ids=semantic_ids,
                sid_to_items=sid_to_items,
                popularity_counter=popularity_counter,
                global_popular_items=global_popular_items,
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

    print("📥 Loading semantic id files...")
    semantic_ids = load_json(args.semantic_ids_path)
    sid_to_items = load_json(args.sid_to_items_path)

    print(f"Semantic IDs: {len(semantic_ids):,}")
    print(f"Clusters:     {len(sid_to_items):,}")

    print("🔥 Building popularity ranking from train targets...")
    popularity_counter, global_popular_items = build_popularity(train_rows)
    print(f"Popular items: {len(global_popular_items):,}")

    print("\n📊 Evaluating valid split...")
    valid_metrics = evaluate_split(
        rows=valid_rows,
        semantic_ids=semantic_ids,
        sid_to_items=sid_to_items,
        popularity_counter=popularity_counter,
        global_popular_items=global_popular_items,
        ks=args.ks,
        exclude_history=args.exclude_history,
    )

    print("📊 Evaluating test split...")
    test_metrics = evaluate_split(
        rows=test_rows,
        semantic_ids=semantic_ids,
        sid_to_items=sid_to_items,
        popularity_counter=popularity_counter,
        global_popular_items=global_popular_items,
        ks=args.ks,
        exclude_history=args.exclude_history,
    )

    results = {
        "baseline": "semantic_id_neighbor",
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "test_path": args.test_path,
        "semantic_ids_path": args.semantic_ids_path,
        "sid_to_items_path": args.sid_to_items_path,
        "ks": args.ks,
        "exclude_history": args.exclude_history,
        "num_train_samples": len(train_rows),
        "num_valid_samples": len(valid_rows),
        "num_test_samples": len(test_rows),
        "num_semantic_ids": len(semantic_ids),
        "num_clusters": len(sid_to_items),
        "num_popular_items": len(global_popular_items),
        "valid": valid_metrics,
        "test": test_metrics,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n========== Semantic-ID Neighbor Baseline ==========")
    print("Valid:")
    for k, v in valid_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Test:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSaved results to: {output_path}")
    print("===================================================")


if __name__ == "__main__":
    main()