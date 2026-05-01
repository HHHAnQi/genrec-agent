import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.generative_rec import GenerativeRecAgent
from agents.user_profile import UserProfileAgent
from schemas.models import RecommendationState


def save_json(path: str, data: dict):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def test_normal_agent(user_id: str):
    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=10,
        mode="genrec_gru",
    )

    profile_agent = UserProfileAgent()
    profile_resp = await profile_agent.run(state)

    if not profile_resp.success:
        return {
            "case": "normal_agent",
            "passed": False,
            "error": profile_resp.error,
        }

    state.user_context = profile_resp.data

    rec_agent = GenerativeRecAgent(
        model_path="models/genrec_gru.pt",
    )
    rec_resp = await rec_agent.run(state)

    passed = (
        rec_resp.success
        and not rec_resp.fallback_used
        and rec_resp.metadata.get("used_mode") == "genrec_gru"
        and rec_resp.metadata.get("model_loaded") is True
    )

    return {
        "case": "normal_agent",
        "passed": passed,
        "success": rec_resp.success,
        "fallback_used": rec_resp.fallback_used,
        "metadata": rec_resp.metadata,
        "num_items": len(rec_resp.data or []),
        "latency_ms": rec_resp.latency_ms,
    }


async def test_missing_model_fallback(user_id: str):
    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=user_id,
        top_k=10,
        mode="genrec_gru",
    )

    profile_agent = UserProfileAgent()
    profile_resp = await profile_agent.run(state)

    if not profile_resp.success:
        return {
            "case": "missing_model_fallback",
            "passed": False,
            "error": profile_resp.error,
        }

    state.user_context = profile_resp.data

    rec_agent = GenerativeRecAgent(
        model_path="models/not_exist_genrec_gru.pt",
    )
    rec_resp = await rec_agent.run(state)

    passed = (
        rec_resp.success
        and rec_resp.fallback_used
        and rec_resp.metadata.get("used_mode") == "semantic_neighbor_fallback"
        and rec_resp.metadata.get("model_loaded") is False
    )

    return {
        "case": "missing_model_fallback",
        "passed": passed,
        "success": rec_resp.success,
        "fallback_used": rec_resp.fallback_used,
        "metadata": rec_resp.metadata,
        "num_items": len(rec_resp.data or []),
        "latency_ms": rec_resp.latency_ms,
    }


async def test_unknown_user_api(
    url: str = "http://127.0.0.1:8000/recommend",
):
    payload = {
        "user_id": "UNKNOWN_USER_FOR_FALLBACK_TEST",
        "top_k": 10,
        "mode": "genrec_gru",
    }

    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        resp = await client.post(url, json=payload)

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text[:500]}

    passed = resp.status_code == 404

    return {
        "case": "unknown_user_api",
        "passed": passed,
        "status_code": resp.status_code,
        "body": body,
    }


async def main():
    with open("datasets/processed/user_sequences.json", "r", encoding="utf-8") as f:
        seqs = json.load(f)

    user_id = next(iter(seqs.keys()))

    print("Using test user:", user_id)

    results = []
    results.append(await test_normal_agent(user_id))
    results.append(await test_missing_model_fallback(user_id))
    results.append(await test_unknown_user_api())

    summary = {
        "all_passed": all(x.get("passed", False) for x in results),
        "num_cases": len(results),
        "num_passed": sum(1 for x in results if x.get("passed", False)),
        "results": results,
    }

    print("\n========== Fallback Test Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===========================================")

    save_json("reports/fallback_test_results.json", summary)
    print("\nSaved to: reports/fallback_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())