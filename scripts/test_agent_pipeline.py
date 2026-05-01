import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.filter import FilterAgent
from agents.generative_rec import GenerativeRecAgent
from agents.user_profile import UserProfileAgent
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
    )

    print("========== Input ==========")
    print("request_id:", state.request_id)
    print("user_id:", user_id)
    print("full_sequence:", full_seq)
    print("===========================\n")

    user_profile_agent = UserProfileAgent()
    generative_rec_agent = GenerativeRecAgent()
    filter_agent = FilterAgent(min_stock=1, remove_history=True)

    # 1. UserProfileAgent
    profile_resp = await user_profile_agent.run(state)
    print("UserProfileAgent:", profile_resp.success, profile_resp.metadata)
    if not profile_resp.success:
        print("error:", profile_resp.error)
        return

    state.user_context = profile_resp.data
    state.trace.append(
        {
            "agent": "UserProfileAgent",
            "success": profile_resp.success,
            "latency_ms": profile_resp.latency_ms,
            "metadata": profile_resp.metadata,
        }
    )

    # 2. GenerativeRecAgent
    rec_resp = await generative_rec_agent.run(state)
    print("GenerativeRecAgent:", rec_resp.success, rec_resp.metadata)
    if not rec_resp.success:
        print("error:", rec_resp.error)
        return

    state.candidates = rec_resp.data
    state.fallback_used = state.fallback_used or rec_resp.fallback_used
    state.trace.append(
        {
            "agent": "GenerativeRecAgent",
            "success": rec_resp.success,
            "latency_ms": rec_resp.latency_ms,
            "fallback_used": rec_resp.fallback_used,
            "metadata": rec_resp.metadata,
        }
    )

    # 3. FilterAgent
    filter_resp = await filter_agent.run(state)
    print("FilterAgent:", filter_resp.success, filter_resp.metadata)
    if not filter_resp.success:
        print("error:", filter_resp.error)
        return

    state.filtered_candidates = filter_resp.data
    state.final_items = filter_resp.data
    state.trace.append(
        {
            "agent": "FilterAgent",
            "success": filter_resp.success,
            "latency_ms": filter_resp.latency_ms,
            "metadata": filter_resp.metadata,
        }
    )

    print("\n========== Final Recommendations ==========")
    for idx, item in enumerate(state.final_items, start=1):
        print(
            f"{idx}. {item.product_id} | "
            f"score={item.score:.4f} | "
            f"stock={item.stock} | "
            f"brand={item.brand} | "
            f"title={item.title}"
        )

    print("\n========== Trace ==========")
    print(json.dumps(state.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())