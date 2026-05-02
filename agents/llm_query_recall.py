import json
from typing import Any

from agents.base import BaseAgent
from agents.generative_rec import GenerativeRecAgent
from schemas.models import AgentResponse, CandidateItem, RecommendationState
from services.llm_client import LLMClient
from services.vector_retriever import VectorRetriever

import pandas as pd

class LLMQueryRecallAgent(BaseAgent):
    """
    LLM-guided semantic query recall agent.

    The LLM does NOT generate product IDs directly.
    It generates a semantic query / intent summary, and VectorRetriever maps
    that query to real products from the local catalog.
    """

    def __init__(
        self,
        fallback_agent: GenerativeRecAgent | None = None,
        retrieval_multiplier: int = 5,
        items_path: str = "datasets/processed/items.csv",
    ):
        super().__init__(
            name="LLMQueryRecallAgent",
            timeout_seconds=20.0,
            max_retries=1,
        )

        self.llm_client = LLMClient()
        self.vector_retriever = VectorRetriever()
        self.fallback_agent = fallback_agent or GenerativeRecAgent()
        self.retrieval_multiplier = retrieval_multiplier

        self.items_df = pd.read_csv(items_path)
        self.item_info = {
            str(row["product_id"]): row.to_dict()
            for _, row in self.items_df.iterrows()
        }

    async def _run(self, state: RecommendationState) -> AgentResponse:
        if state.user_context is None:
            return AgentResponse(
                success=False,
                error="Missing user_context in RecommendationState.",
            )

        intent_result = self._generate_semantic_query(state)

        if not intent_result["success"]:
            return await self._fallback_to_genrec(
                state=state,
                fallback_reason="llm_query_generation_failed",
                error=intent_result.get("error"),
                extra_metadata=intent_result,
            )

        semantic_query = intent_result.get("semantic_query", "").strip()
        if not self._is_query_valid(semantic_query):
            return await self._fallback_to_genrec(
                state=state,
                fallback_reason="invalid_or_empty_semantic_query",
                error=f"Invalid semantic query: {semantic_query!r}",
                extra_metadata=intent_result,
            )

        history = (
            state.user_context.recent_clicks
            or state.user_context.recent_purchases
            or []
        )

        retrieve_top_k = max(state.top_k, state.top_k * self.retrieval_multiplier)

        vector_results, retrieval_meta = self.vector_retriever.retrieve(
            query=semantic_query,
            top_k=retrieve_top_k,
            exclude_product_ids=history,
        )

        if not retrieval_meta.get("success") or not vector_results:
            return await self._fallback_to_genrec(
                state=state,
                fallback_reason="vector_retrieval_failed",
                error=retrieval_meta.get("error") or "No vector retrieval results.",
                extra_metadata={
                    **intent_result,
                    "retrieval": retrieval_meta,
                },
            )

        candidates = []
        removed_by_query_guardrail = []

        for item in vector_results:
            candidate = CandidateItem(
                product_id=item.product_id,
                score=float(item.score),
                source="llm_query_recall",
                title=item.title,
                category=item.category,
                brand=item.brand,
                price=item.price,
                stock=item.stock,
            )

            valid, reason = self._is_retrieved_item_valid_for_query(
                semantic_query=semantic_query,
                item=candidate,
            )

            if not valid:
                removed_by_query_guardrail.append(
                    {
                        "product_id": candidate.product_id,
                        "title": candidate.title,
                        "reason": reason,
                    }
                )
                continue

            candidates.append(candidate)

            if len(candidates) >= state.top_k:
                break

        return AgentResponse(
            success=True,
            data=candidates,
            fallback_used=False,
            metadata={
                "requested_mode": state.mode,
                "used_mode": "llm_query_recall",
                "llm_role": "semantic_query_generation",
                "llm_decides_products": False,
                "llm_candidate_scope": "vector_retrieval_from_local_catalog",
                "semantic_query": semantic_query,
                "intent_summary": intent_result.get("intent_summary"),
                "preferred_attributes": intent_result.get("preferred_attributes", []),
                "risk": intent_result.get("risk"),
                "llm_provider": intent_result.get("llm_provider"),
                "llm_model": intent_result.get("llm_model"),
                "llm_latency_ms": intent_result.get("llm_latency_ms"),
                "retrieval_backend": retrieval_meta.get("retrieval_backend"),
                "embedding_model": retrieval_meta.get("embedding_model"),
                "retrieval_latency_ms": retrieval_meta.get("retrieval_latency_ms"),
                "retrieved_item_count": len(vector_results),
                "num_returned": len(candidates),
                "fallback_reason": None,
                "removed_by_query_guardrail": len(removed_by_query_guardrail),
                "query_guardrail_sample": removed_by_query_guardrail[:3],   
            },
        )

    def _generate_semantic_query(self, state: RecommendationState) -> dict[str, Any]:
        user_context = state.user_context
        assert user_context is not None

        top_categories = sorted(
            user_context.category_pref.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        top_brands = sorted(
            user_context.brand_pref.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        system_prompt = (
            "You are a semantic query generation agent for an e-commerce recommendation system. "
            "Your job is to infer a specific product-search query from the user's recent behavior. "
            "Return valid JSON only. "
            "Do not generate product IDs. "
            "Do not invent unavailable product facts. "
            "Do not mention prices, discounts, medical claims, or guarantees. "
            "The semantic_query must be specific enough for vector retrieval. "
            "Avoid overly generic queries such as 'beauty products', 'skincare cosmetics', or 'all beauty'. "
            "Prefer concrete product types, use cases, and attributes, such as body wash, hand care, facial cleanser, moisturizer, hydrating, portable, refreshing, or daily personal care. "
            "The JSON object must contain exactly these keys: "
            "intent_summary, semantic_query, preferred_attributes, risk."
        )

        recent_ids = (
            user_context.recent_clicks[-10:]
            or user_context.recent_purchases[-10:]
            or []
        )

        recent_items = []
        for pid in recent_ids:
            info = self.item_info.get(str(pid), {})
            recent_items.append(
                {
                    "product_id": str(pid),
                    "title": self._safe_str(info.get("title")),
                    "brand": self._safe_str(info.get("brand")),
                    "category": self._safe_str(info.get("category")),
                }
            )
            
        user_payload = {
            "task_type": "llm_guided_semantic_query_recall",
            "user_id": state.user_id,
            "recent_items": recent_items,
            "top_categories": top_categories,
            "top_brands": top_brands,
            "required_output_schema": {
                "intent_summary": "string",
                "semantic_query": "string",
                "preferred_attributes": ["string"],
                "risk": "low | medium | high",
            },
            "constraints": [
                "semantic_query should be concise and searchable.",
                "semantic_query should describe product types and attributes.",
                "semantic_query must not contain product IDs.",
                "semantic_query must not contain price or discount claims.",
                "Use recent item titles and brands to infer concrete product types.",
                "Avoid generic category-only queries.",
                "Focus on 1-2 dominant product types instead of listing many unrelated beauty areas.",
                "Avoid broad mixed queries such as 'facial care hair care lip balm beauty accessories'.",
            ],
        }

        result = self.llm_client.generate_json(
            system_prompt=system_prompt,
            user_payload=user_payload,
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "llm_provider": result.provider,
                "llm_model": result.model,
                "llm_latency_ms": result.latency_ms,
            }

        data = result.data or {}

        intent_summary = self._safe_str(data.get("intent_summary"))
        semantic_query = self._safe_str(data.get("semantic_query"))
        preferred_attributes = data.get("preferred_attributes", [])
        risk = self._safe_str(data.get("risk"), default="unknown")

        if not isinstance(preferred_attributes, list):
            preferred_attributes = []

        preferred_attributes = [
            self._safe_str(x)
            for x in preferred_attributes
            if self._safe_str(x)
        ][:8]

        return {
            "success": True,
            "error": None,
            "intent_summary": intent_summary,
            "semantic_query": semantic_query,
            "preferred_attributes": preferred_attributes,
            "risk": risk,
            "raw_llm_data": data,
            "llm_provider": result.provider,
            "llm_model": result.model,
            "llm_latency_ms": result.latency_ms,
        }

    async def _fallback_to_genrec(
        self,
        state: RecommendationState,
        fallback_reason: str,
        error: str | None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        # Preserve original requested mode for metadata, but use genrec_gru for fallback.
        fallback_state = state.model_copy(update={"mode": "genrec_gru"})

        response = await self.fallback_agent.run(fallback_state)

        metadata = {
            "requested_mode": state.mode,
            "used_mode": "genrec_gru_fallback",
            "llm_role": "semantic_query_generation",
            "llm_decides_products": False,
            "fallback_reason": fallback_reason,
            "fallback_error": error,
            "fallback_agent_success": response.success,
            "fallback_agent_metadata": response.metadata,
        }

        if extra_metadata:
            metadata["llm_query_recall_metadata"] = extra_metadata

        return AgentResponse(
            success=response.success,
            data=response.data,
            error=response.error,
            fallback_used=True,
            metadata=metadata,
        )

    @staticmethod
    def _safe_str(value, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    def _is_query_valid(self, query: str) -> bool:
        query = self._safe_str(query)
        if not query:
            return False

        words = query.split()
        if len(words) < 3:
            return False

        if len(query) > 240:
            return False

        generic_queries = {
            "beauty products",
            "all beauty",
            "skincare cosmetics",
            "beauty skincare cosmetics",
            "personal care products",
        }

        query_lower = query.lower().strip()

        generic_queries = {
            "beauty products",
            "all beauty",
            "skincare cosmetics",
            "beauty skincare cosmetics",
            "personal care products",
        }

        if query_lower in generic_queries:
            return False

        broad_groups = [
            ["facial", "hair", "lip", "accessories"],
            ["skincare", "makeup", "hair", "tools"],
        ]

        query_lower = query.lower().strip()

        for group in broad_groups:
            hit_count = sum(1 for term in group if term in query_lower)
            if hit_count >= 4:
                return False

        blocked_terms = [
            "product_id",
            "asin",
            "cheap",
            "cheapest",
            "discount",
            "sale",
            "free shipping",
            "guaranteed",
            "cure",
            "medical",
            "doctor recommended",
        ]

        query_lower = query.lower()
        for term in blocked_terms:
            if term in query_lower:
                return False

        return True

    def _is_retrieved_item_valid_for_query(
        self,
        semantic_query: str,
        item: CandidateItem,
    ) -> tuple[bool, str | None]:
        query_lower = semantic_query.lower()
        title_lower = (item.title or "").lower()

        # If query is about skincare sheet masks, avoid protective/travel mask ambiguity.
        if "sheet mask" in query_lower or "facial care" in query_lower:
            blocked_title_terms = [
                "anti-fog",
                "clear face mask",
                "mask bracket",
                "travel pillowcase",
                "ear plugs",
                "international travel",
                "restaurant",
                "school",
                "nail shop",
            ]
            for term in blocked_title_terms:
                if term in title_lower:
                    return False, f"blocked_ambiguous_mask_item:{term}"

        return True, None