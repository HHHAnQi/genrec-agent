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
    user_id = "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ"

    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=10,
        mode="llm_query_recall",
        rerank_mode="llm",
        marketing_mode="llm",
        llm_reason_top_n=3,
    )

    workflow = GenRecWorkflow()
    result = await workflow.ainvoke(state)

    print("========== LLM Query Recall Full Pipeline Test ==========")
    print("user_id:", result.user_id)
    print("mode:", result.mode)
    print("rerank_mode:", result.rerank_mode)
    print("marketing_mode:", result.marketing_mode)
    print("llm_reason_top_n:", result.llm_reason_top_n)
    print("fallback_used:", result.fallback_used)
    print("num_final_items:", len(result.final_items))

    print("\nItems:")
    for idx, item in enumerate(result.final_items, start=1):
        print(
            {
                "rank": idx,
                "product_id": item.product_id,
                "score": round(float(item.score), 4),
                "source": item.source,
                "brand": item.brand,
                "category": item.category,
                "title": item.title,
                "reason": item.reason,
            }
        )

    print("\nTrace:")
    print(json.dumps(result.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())