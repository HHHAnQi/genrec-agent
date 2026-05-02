import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.workflow import GenRecWorkflow
from schemas.models import RecommendationState


async def main():
    # Force invalid LLM endpoint to test rerank fallback.
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_API_KEY"] = "fake-key-for-rerank-fallback-test"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9999/v1"
    os.environ["LLM_MODEL"] = "fake-model"
    os.environ["LLM_TIMEOUT_SECONDS"] = "1"

    with open("datasets/processed/user_sequences.json", "r", encoding="utf-8") as f:
        seqs = json.load(f)

    user_id, _ = next(iter(seqs.items()))

    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=10,
        mode="genrec_gru",
        rerank_mode="llm",
        marketing_mode="template",
    )

    workflow = GenRecWorkflow()
    result = await workflow.ainvoke(state)

    rerank_trace = None
    for t in result.trace:
        if t.get("agent") == "LLMRerankAgent":
            rerank_trace = t
            break

    metadata = rerank_trace.get("metadata", {}) if rerank_trace else {}

    passed = (
        result.user_context is not None
        and len(result.final_items) == 10
        and metadata.get("rerank_mode") == "llm"
        and metadata.get("rerank_applied") is False
        and metadata.get("fallback_reason") == "llm_call_failed"
        and rerank_trace.get("fallback_used") is True
    )

    summary = {
        "case": "llm_rerank_fallback",
        "passed": passed,
        "num_final_items": len(result.final_items),
        "fallback_used": result.fallback_used,
        "rerank_metadata": metadata,
        "sample_items": [
            {
                "product_id": item.product_id,
                "title": item.title,
                "source": item.source,
                "reason": item.reason,
            }
            for item in result.final_items[:3]
        ],
        "trace": result.trace,
    }

    output_path = Path("reports/llm_rerank_fallback_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("========== LLM Rerank Fallback Test ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("==============================================")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())