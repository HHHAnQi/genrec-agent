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
    # Force an invalid OpenAI-compatible provider configuration.
    # This should trigger LLM failure and template fallback inside MarketingAgent.
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_API_KEY"] = "fake-key-for-fallback-test"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9999/v1"
    os.environ["LLM_MODEL"] = "fake-model"
    os.environ["LLM_TIMEOUT_SECONDS"] = "1"

    with open("datasets/processed/user_sequences.json", "r", encoding="utf-8") as f:
        seqs = json.load(f)

    user_id, _ = next(iter(seqs.items()))

    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=5,
        mode="genrec_gru",
        marketing_mode="llm",
    )

    workflow = GenRecWorkflow()
    result = await workflow.ainvoke(state)

    marketing_trace = None
    for t in result.trace:
        if t.get("agent") == "MarketingAgent":
            marketing_trace = t
            break

    metadata = marketing_trace.get("metadata", {}) if marketing_trace else {}

    passed = (
        result.user_context is not None
        and len(result.final_items) == 5
        and metadata.get("reason_type") == "llm"
        and metadata.get("llm_fallback_count", 0) > 0
        and metadata.get("llm_success_count", 0) == 0
    )

    summary = {
        "case": "llm_marketing_fallback",
        "passed": passed,
        "num_final_items": len(result.final_items),
        "fallback_used": result.fallback_used,
        "marketing_metadata": metadata,
        "sample_items": [
            {
                "product_id": item.product_id,
                "title": item.title,
                "reason": item.reason,
            }
            for item in result.final_items[:3]
        ],
        "trace": result.trace,
    }

    output_path = Path("reports/llm_marketing_fallback_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("========== LLM Marketing Fallback Test ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=================================================")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())