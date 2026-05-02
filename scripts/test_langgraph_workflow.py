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
        top_k=10,
        mode="genrec_gru",
        # marketing_mode="llm",
    )

    print("========== Input ==========")
    print("request_id:", state.request_id)
    print("user_id:", user_id)
    print("full_sequence:", full_seq)
    print("===========================\n")

    workflow = GenRecWorkflow()
    result = await workflow.ainvoke(state)

    print("========== Final Recommendations ==========")
    for idx, item in enumerate(result.final_items, start=1):
        print(
            f"{idx}. {item.product_id} | "
            f"score={item.score:.4f} | "
            f"stock={item.stock} | "
            f"brand={item.brand} | "
            f"title={item.title}"
        )

    print("\n========== State Summary ==========")
    print("request_id:", result.request_id)
    print("user_id:", result.user_id)
    print("top_k:", result.top_k)
    print("mode:", result.mode)
    print("fallback_used:", result.fallback_used)
    print("user_context exists:", result.user_context is not None)
    print("num_candidates:", len(result.candidates))
    print("num_filtered:", len(result.filtered_candidates))
    print("num_final_items:", len(result.final_items))

    print("\n========== Trace ==========")
    print(json.dumps(result.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())