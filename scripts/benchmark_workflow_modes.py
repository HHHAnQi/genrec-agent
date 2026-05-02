import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.workflow import GenRecWorkflow
from schemas.models import RecommendationState


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark GenRec-Agent workflow modes."
    )
    parser.add_argument("--test_path", type=str, default="datasets/processed/test.jsonl")
    parser.add_argument("--max_users", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--llm_reason_top_n", type=int, default=3)
    parser.add_argument("--output_json", type=str, default="reports/workflow_benchmark_results.json")
    parser.add_argument("--output_md", type=str, default="reports/workflow_benchmark_summary.md")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_user_ids(test_path: str | Path, max_users: int) -> list[str]:
    rows = read_jsonl(test_path)

    user_ids = []
    seen = set()

    for row in rows:
        uid = row.get("user_id") or row.get("reviewerID") or row.get("user")
        if uid is None:
            continue

        uid = str(uid)
        if uid in seen:
            continue

        user_ids.append(uid)
        seen.add(uid)

        if len(user_ids) >= max_users:
            break

    if not user_ids:
        # Fallback to the user used in previous manual tests.
        user_ids = ["AE23ZBUF2YVBQPH2NN6F5XSA3QYQ"]

    return user_ids[:max_users]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return float(values_sorted[0])

    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)

    if f == c:
        return float(values_sorted[f])

    return float(values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_len(x) -> int:
    if x is None:
        return 0
    if isinstance(x, list):
        return len(x)
    return 0


def extract_trace_stats(trace: list[dict]) -> dict:
    agent_latency = {}
    agent_success = {}
    agent_fallback = {}

    invalid_ids_count = 0
    query_guardrail_count = 0
    llm_reason_fallback_count = 0
    llm_quality_fallback_count = 0
    llm_batch_call_count = 0
    llm_batch_input_items = 0
    template_reason_items = 0
    semantic_queries = []

    for entry in trace:
        agent_name = entry.get("agent", "unknown")
        metadata = entry.get("metadata") or {}

        agent_latency.setdefault(agent_name, []).append(float(entry.get("latency_ms") or 0.0))
        agent_success.setdefault(agent_name, []).append(bool(entry.get("success")))
        agent_fallback.setdefault(agent_name, []).append(bool(entry.get("fallback_used")))

        invalid_ids_count += safe_len(metadata.get("invalid_ids"))
        query_guardrail_count += int(metadata.get("removed_by_query_guardrail") or 0)
        llm_reason_fallback_count += int(metadata.get("llm_fallback_count") or 0)
        llm_quality_fallback_count += int(metadata.get("llm_quality_fallback_count") or 0)
        llm_batch_call_count += int(metadata.get("llm_batch_call_count") or 0)
        llm_batch_input_items += int(metadata.get("llm_batch_input_items") or 0)
        template_reason_items += int(metadata.get("template_reason_items") or 0)

        if metadata.get("semantic_query"):
            semantic_queries.append(metadata.get("semantic_query"))

    return {
        "agent_latency": agent_latency,
        "agent_success": agent_success,
        "agent_fallback": agent_fallback,
        "invalid_ids_count": invalid_ids_count,
        "query_guardrail_count": query_guardrail_count,
        "llm_reason_fallback_count": llm_reason_fallback_count,
        "llm_quality_fallback_count": llm_quality_fallback_count,
        "llm_batch_call_count": llm_batch_call_count,
        "llm_batch_input_items": llm_batch_input_items,
        "template_reason_items": template_reason_items,
        "semantic_queries": semantic_queries,
    }


async def run_one_case(
    workflow: GenRecWorkflow,
    user_id: str,
    mode_name: str,
    config: dict,
    top_k: int,
    llm_reason_top_n: int,
) -> dict:
    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=top_k,
        mode=config["mode"],
        rerank_mode=config["rerank_mode"],
        marketing_mode=config["marketing_mode"],
        llm_reason_top_n=llm_reason_top_n,
    )

    start = time.perf_counter()

    error = None
    result = None

    try:
        result = await workflow.ainvoke(state)
    except Exception as e:
        error = repr(e)

    latency_ms = (time.perf_counter() - start) * 1000

    if result is None:
        return {
            "mode_name": mode_name,
            "user_id": user_id,
            "success": False,
            "error": error,
            "latency_ms": latency_ms,
            "fallback_used": True,
            "num_final_items": 0,
            "trace": [],
            "trace_stats": {},
        }

    success = error is None and len(result.final_items) > 0
    trace_stats = extract_trace_stats(result.trace)

    return {
        "mode_name": mode_name,
        "user_id": user_id,
        "success": success,
        "error": error,
        "latency_ms": latency_ms,
        "fallback_used": result.fallback_used,
        "num_final_items": len(result.final_items),
        "trace": result.trace,
        "trace_stats": trace_stats,
    }


def aggregate_mode_results(mode_name: str, records: list[dict]) -> dict:
    latencies = [float(r["latency_ms"]) for r in records]
    success_values = [1.0 if r["success"] else 0.0 for r in records]
    fallback_values = [1.0 if r["fallback_used"] else 0.0 for r in records]
    num_items = [int(r["num_final_items"]) for r in records]

    invalid_ids_count = sum(int(r.get("trace_stats", {}).get("invalid_ids_count", 0)) for r in records)
    query_guardrail_count = sum(int(r.get("trace_stats", {}).get("query_guardrail_count", 0)) for r in records)
    llm_reason_fallback_count = sum(int(r.get("trace_stats", {}).get("llm_reason_fallback_count", 0)) for r in records)
    llm_quality_fallback_count = sum(int(r.get("trace_stats", {}).get("llm_quality_fallback_count", 0)) for r in records)
    llm_batch_call_count = sum(int(r.get("trace_stats", {}).get("llm_batch_call_count", 0)) for r in records)
    llm_batch_input_items = sum(int(r.get("trace_stats", {}).get("llm_batch_input_items", 0)) for r in records)
    template_reason_items = sum(int(r.get("trace_stats", {}).get("template_reason_items", 0)) for r in records)

    agent_latency_values = {}

    for r in records:
        trace_stats = r.get("trace_stats", {})
        agent_latency = trace_stats.get("agent_latency", {})
        for agent_name, vals in agent_latency.items():
            agent_latency_values.setdefault(agent_name, []).extend(vals)

    agent_latency_summary = {
        agent_name: {
            "avg_ms": mean(vals),
            "p50_ms": percentile(vals, 50),
            "p95_ms": percentile(vals, 95),
        }
        for agent_name, vals in sorted(agent_latency_values.items())
    }

    semantic_queries = []
    for r in records:
        semantic_queries.extend(r.get("trace_stats", {}).get("semantic_queries", []))

    return {
        "mode_name": mode_name,
        "num_cases": len(records),
        "success_rate": mean(success_values),
        "fallback_rate": mean(fallback_values),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "avg_num_final_items": mean(num_items),
        "invalid_ids_count": invalid_ids_count,
        "query_guardrail_count": query_guardrail_count,
        "llm_reason_fallback_count": llm_reason_fallback_count,
        "llm_quality_fallback_count": llm_quality_fallback_count,
        "llm_batch_call_count": llm_batch_call_count,
        "llm_batch_input_items": llm_batch_input_items,
        "template_reason_items": template_reason_items,
        "agent_latency_summary": agent_latency_summary,
        "semantic_queries_sample": semantic_queries[:5],
    }


def write_markdown_summary(
    output_md: str | Path,
    results: dict,
):
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    summaries = results["summaries"]

    lines = []
    lines.append("# Workflow Benchmark Summary")
    lines.append("")
    lines.append("This report benchmarks GenRec-Agent workflow modes from an AI application engineering perspective.")
    lines.append("")
    lines.append("It focuses on endpoint-level latency, success rate, fallback rate, and agent-level latency breakdown.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Test path: `{results['test_path']}`")
    lines.append(f"- Number of users: `{len(results['user_ids'])}`")
    lines.append(f"- Top-K: `{results['top_k']}`")
    lines.append(f"- LLM reason Top-N: `{results['llm_reason_top_n']}`")
    lines.append("")
    lines.append("## Mode Comparison")
    lines.append("")
    lines.append("| Mode | Success Rate | Any Fallback Flag Rate | Avg Latency ms | P50 ms | P95 ms | Avg Items | invalid_ids | Query Guardrail | LLM Reason Fallback |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for mode_name, summary in summaries.items():
        lines.append(
            f"| {mode_name} | "
            f"{summary['success_rate']:.2f} | "
            f"{summary['fallback_rate']:.2f} | "
            f"{summary['avg_latency_ms']:.1f} | "
            f"{summary['p50_latency_ms']:.1f} | "
            f"{summary['p95_latency_ms']:.1f} | "
            f"{summary['avg_num_final_items']:.1f} | "
            f"{summary['invalid_ids_count']} | "
            f"{summary['query_guardrail_count']} | "
            f"{summary['llm_reason_fallback_count']} |"
        )

    lines.append("")
    lines.append("## Agent-level Latency Breakdown")
    lines.append("")

    for mode_name, summary in summaries.items():
        lines.append(f"### {mode_name}")
        lines.append("")
        lines.append("| Agent | Avg ms | P50 ms | P95 ms |")
        lines.append("|---|---:|---:|---:|")

        for agent_name, agent_summary in summary["agent_latency_summary"].items():
            lines.append(
                f"| {agent_name} | "
                f"{agent_summary['avg_ms']:.1f} | "
                f"{agent_summary['p50_ms']:.1f} | "
                f"{agent_summary['p95_ms']:.1f} |"
            )

        lines.append("")

    lines.append("## LLM Query Recall Samples")
    lines.append("")

    for mode_name, summary in summaries.items():
        queries = summary.get("semantic_queries_sample", [])
        if not queries:
            continue

        lines.append(f"### {mode_name}")
        lines.append("")
        for q in queries:
            lines.append(f"- `{q}`")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `genrec_gru_template` is the fastest non-LLM baseline.")
    lines.append("- LLM-enabled modes increase latency because they call DeepSeek for query generation, candidate reranking, and/or recommendation reasons.")
    lines.append("- `llm_reason_top_n` controls marketing latency by generating LLM reasons only for the top-N items.")
    lines.append("- `invalid_ids_count=0` indicates candidate-constrained LLM reranking did not output out-of-candidate product IDs.")
    lines.append("- `query_guardrail_count` shows how many ambiguous vector-retrieval results were filtered by the query recall guardrail.")
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main_async():
    args = parse_args()

    user_ids = load_user_ids(args.test_path, args.max_users)

    modes = {
        "genrec_gru_template": {
            "mode": "genrec_gru",
            "rerank_mode": "none",
            "marketing_mode": "template",
        },
        "genrec_gru_llm_full": {
            "mode": "genrec_gru",
            "rerank_mode": "llm",
            "marketing_mode": "llm",
        },
        "llm_query_recall_template": {
            "mode": "llm_query_recall",
            "rerank_mode": "none",
            "marketing_mode": "template",
        },
        "llm_query_recall_llm_full": {
            "mode": "llm_query_recall",
            "rerank_mode": "llm",
            "marketing_mode": "llm",
        },
    }

    print("========== Workflow Benchmark ==========")
    print("Users:", user_ids)
    print("Top-K:", args.top_k)
    print("LLM reason Top-N:", args.llm_reason_top_n)
    print("Modes:", list(modes.keys()))
    print("========================================")

    workflow = GenRecWorkflow()

    records_by_mode = {}

    for mode_name, config in modes.items():
        print(f"\n===== Running mode: {mode_name} =====")
        records = []

        for idx, user_id in enumerate(user_ids, start=1):
            print(f"[{mode_name}] {idx}/{len(user_ids)} user_id={user_id}")

            record = await run_one_case(
                workflow=workflow,
                user_id=user_id,
                mode_name=mode_name,
                config=config,
                top_k=args.top_k,
                llm_reason_top_n=args.llm_reason_top_n,
            )

            print(
                f"  success={record['success']} "
                f"fallback={record['fallback_used']} "
                f"latency_ms={record['latency_ms']:.1f} "
                f"items={record['num_final_items']}"
            )

            records.append(record)

        records_by_mode[mode_name] = records

    summaries = {
        mode_name: aggregate_mode_results(mode_name, records)
        for mode_name, records in records_by_mode.items()
    }

    results = {
        "benchmark_type": "workflow_modes",
        "test_path": args.test_path,
        "user_ids": user_ids,
        "top_k": args.top_k,
        "llm_reason_top_n": args.llm_reason_top_n,
        "modes": modes,
        "records_by_mode": records_by_mode,
        "summaries": summaries,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_markdown_summary(args.output_md, results)

    print("\n========== Benchmark Summary ==========")
    for mode_name, summary in summaries.items():
        print(
            f"{mode_name}: "
            f"success_rate={summary['success_rate']:.2f}, "
            f"fallback_rate={summary['fallback_rate']:.2f}, "
            f"avg_latency_ms={summary['avg_latency_ms']:.1f}, "
            f"p95_latency_ms={summary['p95_latency_ms']:.1f}"
        )

    print(f"\nSaved JSON to: {output_json}")
    print(f"Saved MD to:   {args.output_md}")
    print("=======================================")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()