from agents.base import BaseAgent
from schemas.models import AgentResponse, CandidateItem, RecommendationState
from services.llm_client import LLMClient


class LLMRerankAgent(BaseAgent):
    """
    Candidate-constrained LLM rerank agent.

    Important:
    - LLM does NOT generate new product IDs.
    - LLM can only reorder candidates produced by GenerativeRecAgent / FilterAgent.
    - If LLM fails or returns invalid IDs, fallback to the original ranking.
    """

    def __init__(self):
        super().__init__(
            name="LLMRerankAgent",
            timeout_seconds=10.0,
            max_retries=1,
        )
        self.llm_client = LLMClient()

    async def _run(self, state: RecommendationState) -> AgentResponse:
        if not state.filtered_candidates:
            return AgentResponse(
                success=False,
                error="No filtered candidates found in RecommendationState.",
            )

        rerank_mode = getattr(state, "rerank_mode", "none")

        if rerank_mode != "llm":
            return AgentResponse(
                success=True,
                data=state.filtered_candidates,
                fallback_used=False,
                metadata={
                    "rerank_mode": rerank_mode,
                    "rerank_applied": False,
                    "reason": "rerank_mode is not llm.",
                    "input_items": len(state.filtered_candidates),
                    "output_items": len(state.filtered_candidates),
                },
            )

        user_context = state.user_context

        top_brands = []
        top_categories = []

        if user_context is not None:
            if user_context.brand_pref:
                top_brands = [
                    x[0]
                    for x in sorted(
                        user_context.brand_pref.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                ]

            if user_context.category_pref:
                top_categories = [
                    x[0]
                    for x in sorted(
                        user_context.category_pref.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                ]

        original_items = state.filtered_candidates
        original_ids = [str(x.product_id) for x in original_items]
        original_map = {str(x.product_id): x for x in original_items}

        system_prompt = (
            "You are an LLM reranking agent for an e-commerce recommendation system. "
            "You will receive a user profile and a list of candidate products. "
            "Your task is to rerank the candidates according to user preference. "
            "You must only choose product_ids from the provided candidates. "
            "Do not invent new product_ids. "
            "Return valid JSON only with keys: reranked_product_ids, summary, risk."
        )

        user_payload = {
            "user_profile": {
                "top_categories": top_categories,
                "top_brands": top_brands,
            },
            "candidates": [
                {
                    "product_id": item.product_id,
                    "title": item.title,
                    "brand": item.brand,
                    "category": item.category,
                    "score": item.score,
                    "stock": item.stock,
                    "source": item.source,
                }
                for item in original_items
            ],
            "task": "Rerank the candidate products. Only return product_ids from the candidate list.",
        }

        result = self.llm_client.generate_json(
            system_prompt=system_prompt,
            user_payload=user_payload,
        )

        if not result.success:
            return AgentResponse(
                success=True,
                data=original_items,
                fallback_used=True,
                metadata={
                    "rerank_mode": "llm",
                    "rerank_applied": False,
                    "fallback_reason": "llm_call_failed",
                    "llm_error": result.error,
                    "llm_provider": result.provider,
                    "llm_model": result.model,
                    "llm_latency_ms": result.latency_ms,
                    "input_items": len(original_items),
                    "output_items": len(original_items),
                },
            )

        raw_ids = result.data.get("reranked_product_ids", [])

        if not isinstance(raw_ids, list):
            return AgentResponse(
                success=True,
                data=original_items,
                fallback_used=True,
                metadata={
                    "rerank_mode": "llm",
                    "rerank_applied": False,
                    "fallback_reason": "invalid_output_type",
                    "llm_provider": result.provider,
                    "llm_model": result.model,
                    "llm_latency_ms": result.latency_ms,
                    "input_items": len(original_items),
                    "output_items": len(original_items),
                },
            )

        valid_ids = []
        invalid_ids = []
        seen = set()

        for pid in raw_ids:
            pid = str(pid)
            if pid in original_map and pid not in seen:
                valid_ids.append(pid)
                seen.add(pid)
            elif pid not in original_map:
                invalid_ids.append(pid)

        # Append missing candidates in original order to avoid losing recall.
        for pid in original_ids:
            if pid not in seen:
                valid_ids.append(pid)
                seen.add(pid)

        if not valid_ids:
            return AgentResponse(
                success=True,
                data=original_items,
                fallback_used=True,
                metadata={
                    "rerank_mode": "llm",
                    "rerank_applied": False,
                    "fallback_reason": "no_valid_ids",
                    "invalid_ids": invalid_ids[:10],
                    "llm_provider": result.provider,
                    "llm_model": result.model,
                    "llm_latency_ms": result.latency_ms,
                    "input_items": len(original_items),
                    "output_items": len(original_items),
                },
            )

        reranked_items: list[CandidateItem] = []

        for rank, pid in enumerate(valid_ids):
            old_item = original_map[pid]
            new_score = 1.0 / (rank + 1)

            reranked_items.append(
                old_item.model_copy(
                    update={
                        "score": new_score,
                        "source": f"{old_item.source}+llm_rerank",
                    }
                )
            )

        return AgentResponse(
            success=True,
            data=reranked_items,
            fallback_used=False,
            metadata={
                "rerank_mode": "llm",
                "rerank_applied": True,
                "llm_provider": result.provider,
                "llm_model": result.model,
                "llm_latency_ms": result.latency_ms,
                "invalid_ids": invalid_ids[:10],
                "input_items": len(original_items),
                "output_items": len(reranked_items),
                "summary": result.data.get("summary"),
                "risk": result.data.get("risk"),
            },
        )