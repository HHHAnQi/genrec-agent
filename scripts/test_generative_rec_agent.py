import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json

from agents.generative_rec import GenerativeRecAgent
from schemas.models import RecommendationState, UserContext


async def main():
    with open("datasets/processed/user_sequences.json", "r", encoding="utf-8") as f:
        seqs = json.load(f)

    user_id, seq = next(iter(seqs.items()))
    history = seq[:-1]

    state = RecommendationState(
        request_id="test-001",
        user_id=user_id,
        top_k=10,
        mode="genrec_gru",
        user_context=UserContext(
            user_id=user_id,
            recent_clicks=history,
        ),
    )

    agent = GenerativeRecAgent()
    response = await agent.run(state)

    print("user_id:", user_id)
    print("history:", history)
    print("success:", response.success)
    print("fallback_used:", response.fallback_used)
    print("latency_ms:", response.latency_ms)
    print("metadata:", response.metadata)

    print("\nCandidates:")
    for item in response.data:
        print(item.model_dump())


if __name__ == "__main__":
    asyncio.run(main())