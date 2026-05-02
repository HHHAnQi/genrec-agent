import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graph.workflow import GenRecWorkflow
from schemas.models import RecommendationState


async def main():
    with open("datasets/processed/user_sequences.json", "r", encoding="utf-8") as f:
        seqs = json.load(f)

    user_id, full_seq = next(iter(seqs.items()))

    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=5,
        mode="genrec_gru",
        marketing_mode="llm",
    )

    workflow = GenRecWorkflow()
    result = await workflow.ainvoke(state)

    print("========== LLM Marketing Test ==========")
    print("user_id:", user_id)
    print("marketing_mode:", result.marketing_mode)
    print("fallback_used:", result.fallback_used)
    print("num_final_items:", len(result.final_items))

    print("\nItems:")
    for item in result.final_items:
        print({
            "product_id": item.product_id,
            "title": item.title,
            "brand": item.brand,
            "reason": item.reason,
        })

    print("\nTrace:")
    print(json.dumps(result.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())