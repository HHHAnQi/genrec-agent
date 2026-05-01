import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark GenRec-Agent FastAPI /recommend endpoint."
    )
    parser.add_argument(
        "--user_sequences_path",
        type=str,
        default="datasets/processed/user_sequences.json",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/recommend",
    )
    parser.add_argument("--num_requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="genrec_gru",
        choices=["genrec_gru", "semantic_neighbor", "popularity"],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="reports/api_benchmark_results.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_user_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        seqs = json.load(f)
    return list(seqs.keys())


async def send_one_request(
    client: httpx.AsyncClient,
    url: str,
    user_id: str,
    top_k: int,
    mode: str,
) -> dict:
    payload = {
        "user_id": user_id,
        "top_k": top_k,
        "mode": mode,
    }

    start = time.perf_counter()

    try:
        resp = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return {
                "success": False,
                "status_code": resp.status_code,
                "latency_ms": elapsed_ms,
                "fallback_used": None,
                "num_items": 0,
                "error": resp.text[:500],
            }

        data = resp.json()

        return {
            "success": True,
            "status_code": resp.status_code,
            "latency_ms": elapsed_ms,
            "server_latency_ms": data.get("latency_ms"),
            "fallback_used": bool(data.get("fallback_used", False)),
            "num_items": len(data.get("items", [])),
            "trace_len": len(data.get("trace", [])),
            "error": None,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "success": False,
            "status_code": None,
            "latency_ms": elapsed_ms,
            "fallback_used": None,
            "num_items": 0,
            "error": repr(e),
        }


async def run_benchmark(args):
    random.seed(args.seed)

    user_ids = load_user_ids(args.user_sequences_path)
    if not user_ids:
        raise ValueError("No user ids found.")

    sampled_users = [
        random.choice(user_ids)
        for _ in range(args.num_requests)
    ]

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        async def bounded_request(uid):
            async with semaphore:
                return await send_one_request(
                    client=client,
                    url=args.url,
                    user_id=uid,
                    top_k=args.top_k,
                    mode=args.mode,
                )

        tasks = [bounded_request(uid) for uid in sampled_users]
        results = await asyncio.gather(*tasks)

    return results


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def summarize(results: list[dict], args) -> dict:
    total = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    latencies = [r["latency_ms"] for r in successes]
    server_latencies = [
        r["server_latency_ms"]
        for r in successes
        if r.get("server_latency_ms") is not None
    ]

    fallback_values = [
        r["fallback_used"]
        for r in successes
        if r.get("fallback_used") is not None
    ]

    item_counts = [r["num_items"] for r in successes]

    summary = {
        "url": args.url,
        "mode": args.mode,
        "top_k": args.top_k,
        "num_requests": total,
        "concurrency": args.concurrency,
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": len(successes) / max(total, 1),
        "fallback_rate": (
            sum(1 for x in fallback_values if x) / max(len(fallback_values), 1)
        ),
        "avg_items_returned": (
            statistics.mean(item_counts) if item_counts else 0.0
        ),
        "client_latency_ms": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "server_latency_ms": {
            "avg": statistics.mean(server_latencies) if server_latencies else 0.0,
            "p50": percentile(server_latencies, 50),
            "p95": percentile(server_latencies, 95),
            "p99": percentile(server_latencies, 99),
            "min": min(server_latencies) if server_latencies else 0.0,
            "max": max(server_latencies) if server_latencies else 0.0,
        },
        "sample_failures": failures[:5],
    }

    return summary


async def main_async():
    args = parse_args()

    print("========== Benchmark Config ==========")
    print(f"url:             {args.url}")
    print(f"mode:            {args.mode}")
    print(f"top_k:           {args.top_k}")
    print(f"num_requests:    {args.num_requests}")
    print(f"concurrency:     {args.concurrency}")
    print("=====================================\n")

    start = time.perf_counter()
    results = await run_benchmark(args)
    total_time = time.perf_counter() - start

    summary = summarize(results, args)
    summary["total_wall_time_sec"] = total_time
    summary["throughput_req_per_sec"] = args.num_requests / max(total_time, 1e-9)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "raw_results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("========== Benchmark Summary ==========")
    print(f"Requests:              {summary['num_requests']}")
    print(f"Concurrency:           {summary['concurrency']}")
    print(f"Success rate:          {summary['success_rate']:.4f}")
    print(f"Fallback rate:         {summary['fallback_rate']:.4f}")
    print(f"Avg items returned:    {summary['avg_items_returned']:.2f}")
    print(f"Throughput req/s:      {summary['throughput_req_per_sec']:.2f}")

    print("\nClient latency ms:")
    for k, v in summary["client_latency_ms"].items():
        print(f"  {k}: {v:.2f}")

    print("\nServer latency ms:")
    for k, v in summary["server_latency_ms"].items():
        print(f"  {k}: {v:.2f}")

    print(f"\nSaved results to: {output_path}")
    print("=======================================")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()