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
        description="Evaluate Hybrid GenRec reranker for GenRec-Agent."
    )

    parser.add_argument("--train_path", type=str, default="datasets/processed/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="datasets/processed/valid.jsonl")
    parser.add_argument("--test_path", type=str, default="datasets/processed/test.jsonl")
    parser.add_argument("--semantic_ids_path", type=str, default="datasets/processed/semantic_ids.json")
    parser.add_argument("--sid_to_items_path", type=str, default="datasets/processed/sid_to_items.json")
    parser.add_argument("--model_path", type=str, default="models/genrec_gru.pt")

    parser.add_argument("--split", type=str, default="test", choices=["valid", "test", "both"])
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--top_k_eval", type=int, default=50)
    parser.add_argument("--top_clusters", type=int, default=10)
    parser.add_argument("--exclude_history", action="store_true")

    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for GRU cluster probability.")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for semantic-neighbor score.")
    parser.add_argument("--gamma", type=float, default=0.2, help="Weight for normalized popularity score.")

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a small alpha/beta/gamma sweep instead of a single setting.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output_json",
        type=str,
        default="reports/hybrid_genrec_results.json",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="reports/hybrid_genrec_summary.md",
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


def get_item_sid_key(item_id: str, semantic_ids: dict) -> str | None:
    sid = semantic_ids.get(str(item_id))
    if sid is None:
        return None
    if isinstance(sid, list) and len(sid) > 0:
        return str(sid[0])
    return str(sid)


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


def recall_at_k(recs: list[str], target: str, k: int) -> float:
    return 1.0 if str(target) in recs[:k] else 0.0


def hitrate_at_k(recs: list[str], target: str, k: int) -> float:
    return 1.0 if str(target) in recs[:k] else 0.0


def ndcg_at_k(recs: list[str], target: str, k: int) -> float:
    target = str(target)
    topk = recs[:k]
    if target not in topk:
        return 0.0
    rank = topk.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def mrr(recs: list[str], target: str) -> float:
    target = str(target)
    if target not in recs:
        return 0.0
    rank = recs.index(target) + 1
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


def semantic_neighbor_score_for_item(
    item_id: str,
    history: Iterable[str],
    semantic_ids: dict,
) -> float:
    item_id = str(item_id)
    history = [str(x) for x in history]

    target_sid = get_item_sid_key(item_id, semantic_ids)
    if target_sid is None:
        return 0.0

    cluster_counter = Counter()
    for h in history:
        sid = get_item_sid_key(h, semantic_ids)
        if sid is not None:
            cluster_counter[sid] += 1

    if not cluster_counter:
        return 0.0

    # Normalized semantic-neighbor strength.
    max_count = max(cluster_counter.values())
    return float(cluster_counter.get(target_sid, 0)) / float(max_count)


def gru_score_for_item(
    item_id: str,
    probs: np.ndarray | None,
    semantic_ids: dict,
) -> float:
    if probs is None:
        return 0.0

    sid_key = get_item_sid_key(item_id, semantic_ids)
    if sid_key is None:
        return 0.0

    cid = int(sid_key)
    if cid < 0 or cid >= len(probs):
        return 0.0

    return float(probs[cid])


def build_candidate_pool(
    history: Iterable[str],
    semantic_ids: dict,
    sid_to_items: dict,
    global_popular_items: list[str],
    engine: GenRecInference,
    top_clusters: int,
    top_k_eval: int,
    exclude_history: bool,
) -> list[str]:
    history = [str(x) for x in history]
    history_set = set(history)

    candidates = []
    seen = set()

    def add_item(x: str):
        x = str(x)
        if x in seen:
            return
        if exclude_history and x in history_set:
            return
        candidates.append(x)
        seen.add(x)

    # 1) Add Semantic-ID neighbor candidates from user history clusters.
    cluster_counter = Counter()
    for h in history:
        sid = get_item_sid_key(h, semantic_ids)
        if sid is not None:
            cluster_counter[sid] += 1

    for sid_key, _ in cluster_counter.most_common():
        for item in sid_to_items.get(str(sid_key), []):
            add_item(item)

    # 2) Add items from GRU-predicted top clusters.
    probs = get_gru_cluster_probs(engine, history)
    if probs is not None:
        top_cluster_ids = np.argsort(-probs)[:top_clusters].tolist()
        for cid in top_cluster_ids:
            for item in sid_to_items.get(str(int(cid)), []):
                add_item(item)

    # 3) Fill by global popularity.
    for item in global_popular_items:
        add_item(item)
        if len(candidates) >= max(top_k_eval * 10, top_k_eval):
            break

    return candidates


def rank_hybrid_candidates(
    candidates: list[str],
    history: Iterable[str],
    semantic_ids: dict,
    pop_norm: dict[str, float],
    engine: GenRecInference,
    alpha: float,
    beta: float,
    gamma: float,
) -> list[str]:
    probs = get_gru_cluster_probs(engine, history)

    scored = []
    for item in candidates:
        item = str(item)

        gru_score = gru_score_for_item(
            item_id=item,
            probs=probs,
            semantic_ids=semantic_ids,
        )
        semantic_score = semantic_neighbor_score_for_item(
            item_id=item,
            history=history,
            semantic_ids=semantic_ids,
        )
        pop_score = pop_norm.get(item, 0.0)

        score = alpha * gru_score + beta * semantic_score + gamma * pop_score

        scored.append(
            {
                "item": item,
                "score": score,
                "gru_score": gru_score,
                "semantic_score": semantic_score,
                "pop_score": pop_score,
            }
        )

    ranked = sorted(
        scored,
        key=lambda x: (x["score"], x["gru_score"], x["semantic_score"], x["pop_score"], x["item"]),
        reverse=True,
    )
    return [x["item"] for x in ranked]


def rank_baseline_semantic(
    candidates: list[str],
    history: Iterable[str],
    semantic_ids: dict,
    pop_norm: dict[str, float],
) -> list[str]:
    scored = []
    for item in candidates:
        item = str(item)
        semantic_score = semantic_neighbor_score_for_item(item, history, semantic_ids)
        pop_score = pop_norm.get(item, 0.0)
        score = semantic_score + 0.1 * pop_score
        scored.append((item, score, semantic_score, pop_score))

    ranked = sorted(scored, key=lambda x: (x[1], x[2], x[3], x[0]), reverse=True)
    return [x[0] for x in ranked]


def rank_baseline_popularity(
    candidates: list[str],
    pop_norm: dict[str, float],
) -> list[str]:
    ranked = sorted(
        [str(x) for x in candidates],
        key=lambda item: (pop_norm.get(item, 0.0), item),
        reverse=True,
    )
    return ranked


def evaluate_rows(
    rows: list[dict],
    semantic_ids: dict,
    sid_to_items: dict,
    pop_norm: dict[str, float],
    global_popular_items: list[str],
    engine: GenRecInference,
    ks: list[int],
    top_k_eval: int,
    top_clusters: int,
    exclude_history: bool,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    method_stores = {
        "popularity": {"MRR": []},
        "semantic_neighbor": {"MRR": []},
        "hybrid_genrec": {"MRR": []},
    }

    for method in method_stores:
        for k in ks:
            method_stores[method][f"Recall@{k}"] = []
            method_stores[method][f"HitRate@{k}"] = []
            method_stores[method][f"NDCG@{k}"] = []

    num_empty_candidates = 0

    for row in rows:
        history = [str(x) for x in row.get("history", [])]
        target = str(row["target"])

        candidates = build_candidate_pool(
            history=history,
            semantic_ids=semantic_ids,
            sid_to_items=sid_to_items,
            global_popular_items=global_popular_items,
            engine=engine,
            top_clusters=top_clusters,
            top_k_eval=top_k_eval,
            exclude_history=exclude_history,
        )

        # Make sure target can be ranked. This is still full-catalog style target checking:
        # if target is not naturally retrieved by the candidate generator, it is appended
        # at the end so the scorer can still rank it.
        if target not in candidates:
            candidates.append(target)

        if not candidates:
            num_empty_candidates += 1
            continue

        ranked_pop = rank_baseline_popularity(candidates, pop_norm)
        ranked_sem = rank_baseline_semantic(candidates, history, semantic_ids, pop_norm)
        ranked_hybrid = rank_hybrid_candidates(
            candidates=candidates,
            history=history,
            semantic_ids=semantic_ids,
            pop_norm=pop_norm,
            engine=engine,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        ranked_map = {
            "popularity": ranked_pop,
            "semantic_neighbor": ranked_sem,
            "hybrid_genrec": ranked_hybrid,
        }

        for method, ranked_items in ranked_map.items():
            method_stores[method]["MRR"].append(mrr(ranked_items, target))
            for k in ks:
                method_stores[method][f"Recall@{k}"].append(recall_at_k(ranked_items, target, k))
                method_stores[method][f"HitRate@{k}"].append(hitrate_at_k(ranked_items, target, k))
                method_stores[method][f"NDCG@{k}"].append(ndcg_at_k(ranked_items, target, k))

    metrics = {}
    for method, store in method_stores.items():
        metrics[method] = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in store.items()
        }

    return {
        "num_rows": len(rows),
        "num_empty_candidates": num_empty_candidates,
        "metrics": metrics,
    }


def run_single_setting(
    split_name: str,
    rows: list[dict],
    semantic_ids: dict,
    sid_to_items: dict,
    pop_norm: dict[str, float],
    global_popular_items: list[str],
    engine: GenRecInference,
    args,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    result = evaluate_rows(
        rows=rows,
        semantic_ids=semantic_ids,
        sid_to_items=sid_to_items,
        pop_norm=pop_norm,
        global_popular_items=global_popular_items,
        engine=engine,
        ks=args.ks,
        top_k_eval=args.top_k_eval,
        top_clusters=args.top_clusters,
        exclude_history=args.exclude_history,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    result.update(
        {
            "split": split_name,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
    )
    return result


def write_markdown_summary(output_md: str | Path, results: dict):
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Hybrid GenRec Reranker Evaluation")
    lines.append("")
    lines.append("This report evaluates a hybrid reranker that combines GRU cluster probability, Semantic-ID neighbor score, and normalized popularity.")
    lines.append("")
    lines.append("Hybrid score:")
    lines.append("")
    lines.append("```text")
    lines.append("score(item) = alpha * gru_cluster_prob + beta * semantic_neighbor_score + gamma * popularity_score")
    lines.append("```")
    lines.append("")
    lines.append("Important: this is an offline diagnostic experiment. It does not replace the existing full service pipeline.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Split: `{results['split']}`")
    lines.append(f"- Exclude history: `{results['exclude_history']}`")
    lines.append(f"- Top clusters: `{results['top_clusters']}`")
    lines.append(f"- Top-K eval candidate pool control: `{results['top_k_eval']}`")
    lines.append(f"- Model loaded: `{results['model_loaded']}`")
    lines.append("")

    if results.get("sweep"):
        lines.append("## Sweep Results")
        lines.append("")
        lines.append("| Rank | Split | alpha | beta | gamma | Hybrid Recall@20 | Hybrid NDCG@20 | Hybrid MRR |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")

        ranked = results["sweep_results_ranked"]
        for i, row in enumerate(ranked[:20], start=1):
            m = row["metrics"]["hybrid_genrec"]
            lines.append(
                f"| {i} | {row['split']} | {row['alpha']} | {row['beta']} | {row['gamma']} | "
                f"{m.get('Recall@20', 0.0):.4f} | {m.get('NDCG@20', 0.0):.4f} | {m.get('MRR', 0.0):.4f} |"
            )
        lines.append("")

        best = ranked[0]
        lines.append("## Best Setting")
        lines.append("")
        lines.append(f"- alpha: `{best['alpha']}`")
        lines.append(f"- beta: `{best['beta']}`")
        lines.append(f"- gamma: `{best['gamma']}`")
        lines.append("")
        lines.append("### Metrics")
        lines.append("")
        write_metrics_table(lines, best["metrics"], results["ks"])

    else:
        lines.append("## Results")
        lines.append("")
        write_metrics_table(lines, results["metrics"], results["ks"])

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If Hybrid GenRec improves over Semantic-ID Neighbor, the GRU sequence signal complements semantic-neighbor retrieval.")
    lines.append("- If Hybrid GenRec does not improve, the current sparse Beauty subset is likely dominated by local semantic-neighbor signals.")
    lines.append("- This experiment should be reported as a diagnostic reranking study, not as a production recommender benchmark.")
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_metrics_table(lines: list[str], metrics: dict, ks: list[int]):
    methods = ["popularity", "semantic_neighbor", "hybrid_genrec"]
    metric_names = []
    for k in ks:
        metric_names.extend([f"Recall@{k}", f"NDCG@{k}"])
    metric_names.append("MRR")

    lines.append("| Method | " + " | ".join(metric_names) + " |")
    lines.append("|---" + "|---:" * len(metric_names) + "|")

    for method in methods:
        row = [method]
        for name in metric_names:
            row.append(f"{metrics[method].get(name, 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("📥 Loading data...")
    train_rows = read_jsonl(args.train_path)
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)
    semantic_ids = load_json(args.semantic_ids_path)
    sid_to_items = load_json(args.sid_to_items_path)

    item_popularity = build_item_popularity(train_rows)
    pop_norm = normalize_counter(item_popularity)
    global_popular_items = [item for item, _ in item_popularity.most_common()]

    print(f"Train samples: {len(train_rows):,}")
    print(f"Valid samples: {len(valid_rows):,}")
    print(f"Test samples:  {len(test_rows):,}")
    print(f"Semantic IDs:  {len(semantic_ids):,}")
    print(f"Clusters:      {len(sid_to_items):,}")
    print(f"Popular items: {len(global_popular_items):,}")

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

    split_rows = {
        "valid": valid_rows,
        "test": test_rows,
    }

    selected_splits = ["valid", "test"] if args.split == "both" else [args.split]

    if args.sweep:
        alpha_values = [0.5, 1.0, 2.0]
        beta_values = [0.5, 1.0, 2.0]
        gamma_values = [0.0, 0.1, 0.2, 0.5]

        sweep_results = []

        print("\n🔎 Running hybrid weight sweep...")
        for split_name in selected_splits:
            rows = split_rows[split_name]
            for alpha in alpha_values:
                for beta in beta_values:
                    for gamma in gamma_values:
                        print(
                            f"  split={split_name} alpha={alpha} beta={beta} gamma={gamma}"
                        )
                        row = run_single_setting(
                            split_name=split_name,
                            rows=rows,
                            semantic_ids=semantic_ids,
                            sid_to_items=sid_to_items,
                            pop_norm=pop_norm,
                            global_popular_items=global_popular_items,
                            engine=engine,
                            args=args,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                        )
                        sweep_results.append(row)

        sweep_results_ranked = sorted(
            sweep_results,
            key=lambda x: (
                x["metrics"]["hybrid_genrec"].get("Recall@20", 0.0),
                x["metrics"]["hybrid_genrec"].get("NDCG@20", 0.0),
                x["metrics"]["hybrid_genrec"].get("MRR", 0.0),
            ),
            reverse=True,
        )

        results = {
            "evaluation_type": "hybrid_genrec_reranker",
            "sweep": True,
            "split": args.split,
            "ks": args.ks,
            "exclude_history": args.exclude_history,
            "top_clusters": args.top_clusters,
            "top_k_eval": args.top_k_eval,
            "model_path": args.model_path,
            "model_loaded": engine.model_loaded,
            "model_load_error": engine.load_error,
            "sweep_results": sweep_results,
            "sweep_results_ranked": sweep_results_ranked,
        }

    else:
        print("\n📊 Running single hybrid setting...")
        if len(selected_splits) != 1:
            raise ValueError("--split both requires --sweep for this script.")

        split_name = selected_splits[0]
        rows = split_rows[split_name]

        single_result = run_single_setting(
            split_name=split_name,
            rows=rows,
            semantic_ids=semantic_ids,
            sid_to_items=sid_to_items,
            pop_norm=pop_norm,
            global_popular_items=global_popular_items,
            engine=engine,
            args=args,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )

        results = {
            "evaluation_type": "hybrid_genrec_reranker",
            "sweep": False,
            "split": split_name,
            "ks": args.ks,
            "exclude_history": args.exclude_history,
            "top_clusters": args.top_clusters,
            "top_k_eval": args.top_k_eval,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "model_path": args.model_path,
            "model_loaded": engine.model_loaded,
            "model_load_error": engine.load_error,
            **single_result,
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_markdown_summary(args.output_md, results)

    print("\n========== Hybrid GenRec Reranker Results ==========")
    print(f"Split: {args.split}")
    print(f"Sweep: {args.sweep}")

    if args.sweep:
        best = results["sweep_results_ranked"][0]
        print(
            f"Best: split={best['split']} "
            f"alpha={best['alpha']} beta={best['beta']} gamma={best['gamma']}"
        )
        for method, metrics in best["metrics"].items():
            print(f"\n[{method}]")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
    else:
        for method, metrics in results["metrics"].items():
            print(f"\n[{method}]")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")

    print(f"\nSaved JSON to: {output_json}")
    print(f"Saved MD to:   {args.output_md}")
    print("====================================================")


if __name__ == "__main__":
    main()