import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recommender.inference import GenRecInference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sampled-candidate ranking for GenRec-Agent."
    )
    parser.add_argument("--train_path", type=str, default="datasets/processed/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="datasets/processed/valid.jsonl")
    parser.add_argument("--test_path", type=str, default="datasets/processed/test.jsonl")
    parser.add_argument("--semantic_ids_path", type=str, default="datasets/processed/semantic_ids.json")
    parser.add_argument("--sid_to_items_path", type=str, default="datasets/processed/sid_to_items.json")
    parser.add_argument("--model_path", type=str, default="models/genrec_gru.pt")

    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--num_negatives", type=int, default=99)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--top_clusters", type=int, default=5)
    parser.add_argument("--exclude_history", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output_json",
        type=str,
        default="reports/sampled_ranking_results.json",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="reports/sampled_ranking_summary.md",
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


def build_item_popularity(train_rows: list[dict]) -> Counter:
    counter = Counter()
    for row in train_rows:
        counter[str(row["target"])] += 1
    return counter


def normalize_counter(counter: Counter) -> dict[str, float]:
    if not counter:
        return {}
    max_value = max(counter.values())
    if max_value <= 0:
        return {}
    return {str(k): float(v) / float(max_value) for k, v in counter.items()}


def get_item_sid_key(item_id: str, semantic_ids: dict) -> str | None:
    sid = semantic_ids.get(str(item_id))
    if sid is None:
        return None
    if isinstance(sid, list) and len(sid) > 0:
        return str(sid[0])
    return str(sid)


def build_all_items(
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
    semantic_ids: dict,
) -> list[str]:
    items = set(str(x) for x in semantic_ids.keys())

    for rows in [train_rows, valid_rows, test_rows]:
        for row in rows:
            items.add(str(row["target"]))
            for h in row.get("history", []):
                items.add(str(h))

    return sorted(items)


def sample_candidate_set(
    target: str,
    history: Iterable[str],
    all_items: list[str],
    num_negatives: int,
    rng: random.Random,
    exclude_history: bool,
) -> list[str]:
    target = str(target)
    history_set = set(str(x) for x in history)

    forbidden = {target}
    if exclude_history:
        forbidden.update(history_set)

    negative_pool = [item for item in all_items if str(item) not in forbidden]

    if len(negative_pool) < num_negatives:
        raise ValueError(
            f"Not enough negative items: requested={num_negatives}, available={len(negative_pool)}"
        )

    negatives = rng.sample(negative_pool, num_negatives)
    candidates = [target] + negatives
    rng.shuffle(candidates)
    return candidates


def recall_at_k(ranked_items: list[str], target: str, k: int) -> float:
    return 1.0 if str(target) in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items: list[str], target: str, k: int) -> float:
    topk = ranked_items[:k]
    target = str(target)

    if target not in topk:
        return 0.0

    rank = topk.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def mrr(ranked_items: list[str], target: str) -> float:
    target = str(target)
    if target not in ranked_items:
        return 0.0

    rank = ranked_items.index(target) + 1
    return 1.0 / rank


def get_gru_cluster_probs(
    engine: GenRecInference,
    history: Iterable[str],
) -> np.ndarray | None:
    if not engine.model_loaded or engine.model is None:
        return None

    history = [str(x) for x in history]
    max_history_len = int(engine.model_args.get("max_history_len", 20))

    cluster_seq = []
    for item in history:
        cid = engine.get_cluster_id(item)
        if cid is not None:
            cluster_seq.append(cid)

    if not cluster_seq:
        return None

    cluster_seq = cluster_seq[-max_history_len:]
    token_seq = [cid + 1 for cid in cluster_seq]

    pad_len = max_history_len - len(token_seq)
    if pad_len > 0:
        token_seq = [0] * pad_len + token_seq

    input_ids = torch.tensor([token_seq], dtype=torch.long, device=engine.device)

    with torch.no_grad():
        logits = engine.model(input_ids)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return probs


def score_popularity_candidates(
    candidates: list[str],
    pop_norm: dict[str, float],
) -> dict[str, float]:
    return {item: pop_norm.get(str(item), 0.0) for item in candidates}


def score_semantic_neighbor_candidates(
    candidates: list[str],
    history: Iterable[str],
    semantic_ids: dict,
    pop_norm: dict[str, float],
) -> dict[str, float]:
    history = [str(x) for x in history]

    cluster_counter = Counter()
    for item in history:
        sid_key = get_item_sid_key(item, semantic_ids)
        if sid_key is not None:
            cluster_counter[sid_key] += 1

    scores = {}

    for item in candidates:
        item = str(item)
        sid_key = get_item_sid_key(item, semantic_ids)

        cluster_score = 0.0
        if sid_key is not None:
            cluster_score = float(cluster_counter.get(sid_key, 0))

        pop_score = pop_norm.get(item, 0.0)

        # Semantic match dominates; popularity breaks ties.
        scores[item] = cluster_score * 10.0 + 0.1 * pop_score

    return scores


def score_genrec_gru_candidates(
    candidates: list[str],
    history: Iterable[str],
    engine: GenRecInference,
    semantic_ids: dict,
    pop_norm: dict[str, float],
) -> dict[str, float]:
    probs = get_gru_cluster_probs(engine, history)

    if probs is None:
        # If model is missing or history has no valid clusters, fallback to semantic score.
        return score_semantic_neighbor_candidates(
            candidates=candidates,
            history=history,
            semantic_ids=semantic_ids,
            pop_norm=pop_norm,
        )

    scores = {}

    for item in candidates:
        item = str(item)
        sid_key = get_item_sid_key(item, semantic_ids)

        cluster_score = 0.0
        if sid_key is not None:
            cid = int(sid_key)
            if 0 <= cid < len(probs):
                cluster_score = float(probs[cid])

        pop_score = pop_norm.get(item, 0.0)

        # GRU cluster probability dominates; popularity breaks ties inside clusters.
        scores[item] = cluster_score * 100.0 + 0.1 * pop_score

    return scores


def rank_candidates(scores: dict[str, float]) -> list[str]:
    return [
        item
        for item, _ in sorted(
            scores.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
    ]


def init_metric_store(ks: list[int]) -> dict[str, list[float]]:
    store = {"MRR": []}
    for k in ks:
        store[f"Recall@{k}"] = []
        store[f"NDCG@{k}"] = []
    return store


def update_metrics(
    store: dict[str, list[float]],
    ranked_items: list[str],
    target: str,
    ks: list[int],
):
    store["MRR"].append(mrr(ranked_items, target))
    for k in ks:
        store[f"Recall@{k}"].append(recall_at_k(ranked_items, target, k))
        store[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, target, k))


def aggregate_metrics(store: dict[str, list[float]]) -> dict[str, float]:
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in store.items()
    }


def evaluate_sampled_ranking(
    rows: list[dict],
    all_items: list[str],
    semantic_ids: dict,
    pop_norm: dict[str, float],
    engine: GenRecInference,
    ks: list[int],
    num_negatives: int,
    exclude_history: bool,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    methods = {
        "popularity": init_metric_store(ks),
        "semantic_neighbor": init_metric_store(ks),
        "genrec_gru": init_metric_store(ks),
    }

    num_skipped = 0

    for idx, row in enumerate(rows):
        history = [str(x) for x in row.get("history", [])]
        target = str(row["target"])

        try:
            candidates = sample_candidate_set(
                target=target,
                history=history,
                all_items=all_items,
                num_negatives=num_negatives,
                rng=rng,
                exclude_history=exclude_history,
            )
        except ValueError:
            num_skipped += 1
            continue

        pop_scores = score_popularity_candidates(
            candidates=candidates,
            pop_norm=pop_norm,
        )
        semantic_scores = score_semantic_neighbor_candidates(
            candidates=candidates,
            history=history,
            semantic_ids=semantic_ids,
            pop_norm=pop_norm,
        )
        genrec_scores = score_genrec_gru_candidates(
            candidates=candidates,
            history=history,
            engine=engine,
            semantic_ids=semantic_ids,
            pop_norm=pop_norm,
        )

        update_metrics(
            methods["popularity"],
            rank_candidates(pop_scores),
            target,
            ks,
        )
        update_metrics(
            methods["semantic_neighbor"],
            rank_candidates(semantic_scores),
            target,
            ks,
        )
        update_metrics(
            methods["genrec_gru"],
            rank_candidates(genrec_scores),
            target,
            ks,
        )

    return {
        "num_rows": len(rows),
        "num_evaluated": len(rows) - num_skipped,
        "num_skipped": num_skipped,
        "metrics": {
            method: aggregate_metrics(store)
            for method, store in methods.items()
        },
    }


def write_markdown_summary(
    output_md: str | Path,
    results: dict,
):
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    metrics = results["metrics"]
    methods = ["popularity", "semantic_neighbor", "genrec_gru"]

    metric_names = []
    for k in results["ks"]:
        metric_names.append(f"Recall@{k}")
        metric_names.append(f"NDCG@{k}")
    metric_names.append("MRR")

    lines = []
    lines.append("# Sampled-Candidate Ranking Evaluation")
    lines.append("")
    lines.append("This report adds a sampled-candidate ranking evaluation for GenRec-Agent.")
    lines.append("")
    lines.append("Important: this evaluation is **not a replacement** for full-catalog ranking. ")
    lines.append("It is a diagnostic benchmark that measures how well each method ranks the target item within a smaller candidate set.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Split: `{results['split']}`")
    lines.append(f"- Num negatives per case: `{results['num_negatives']}`")
    lines.append(f"- Candidate set size: `{results['num_negatives'] + 1}`")
    lines.append(f"- Exclude history: `{results['exclude_history']}`")
    lines.append(f"- Seed: `{results['seed']}`")
    lines.append(f"- Num evaluated: `{results['num_evaluated']}`")
    lines.append(f"- Num skipped: `{results['num_skipped']}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Method | " + " | ".join(metric_names) + " |")
    lines.append("|---" + "|---:" * len(metric_names) + "|")

    for method in methods:
        row = [method]
        for metric_name in metric_names:
            row.append(f"{metrics[method].get(metric_name, 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Full-catalog ranking remains the stricter retrieval setting.")
    lines.append("- Sampled-candidate ranking is used here to inspect candidate-level ranking behavior under a controlled candidate pool.")
    lines.append("- These numbers should not be directly compared with full-catalog Recall/NDCG.")
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    print("📥 Loading splits and metadata...")
    train_rows = read_jsonl(args.train_path)
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)

    rows = valid_rows if args.split == "valid" else test_rows

    semantic_ids = load_json(args.semantic_ids_path)
    sid_to_items = load_json(args.sid_to_items_path)

    print(f"Train samples: {len(train_rows):,}")
    print(f"Valid samples: {len(valid_rows):,}")
    print(f"Test samples:  {len(test_rows):,}")
    print(f"Eval split:    {args.split}")
    print(f"Eval rows:     {len(rows):,}")

    all_items = build_all_items(
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        semantic_ids=semantic_ids,
    )

    print(f"All items: {len(all_items):,}")

    item_popularity = build_item_popularity(train_rows)
    pop_norm = normalize_counter(item_popularity)

    print("📦 Loading GenRecInference engine...")
    engine = GenRecInference(
        model_path=args.model_path,
        train_path=args.train_path,
        semantic_ids_path=args.semantic_ids_path,
        sid_to_items_path=args.sid_to_items_path,
    )

    print(f"Model loaded: {engine.model_loaded}")
    if engine.load_error:
        print(f"Load error: {engine.load_error}")

    print("\n📊 Running sampled-candidate evaluation...")
    sampled_results = evaluate_sampled_ranking(
        rows=rows,
        all_items=all_items,
        semantic_ids=semantic_ids,
        pop_norm=pop_norm,
        engine=engine,
        ks=args.ks,
        num_negatives=args.num_negatives,
        exclude_history=args.exclude_history,
        seed=args.seed,
    )

    results = {
        "evaluation_type": "sampled_candidate_ranking",
        "split": args.split,
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "test_path": args.test_path,
        "semantic_ids_path": args.semantic_ids_path,
        "sid_to_items_path": args.sid_to_items_path,
        "model_path": args.model_path,
        "num_negatives": args.num_negatives,
        "candidate_set_size": args.num_negatives + 1,
        "ks": args.ks,
        "exclude_history": args.exclude_history,
        "seed": args.seed,
        "num_all_items": len(all_items),
        "model_loaded": engine.model_loaded,
        "model_load_error": engine.load_error,
        **sampled_results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_markdown_summary(args.output_md, results)

    print("\n========== Sampled-Candidate Ranking Results ==========")
    print(f"Split: {args.split}")
    print(f"Candidate set size: {args.num_negatives + 1}")
    print(f"Evaluated rows: {results['num_evaluated']}")
    print(f"Skipped rows:   {results['num_skipped']}")

    for method, metrics in results["metrics"].items():
        print(f"\n[{method}]")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    print(f"\nSaved JSON to: {output_json}")
    print(f"Saved MD to:   {args.output_md}")
    print("=======================================================")


if __name__ == "__main__":
    main()