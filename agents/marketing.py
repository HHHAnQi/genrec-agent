from agents.base import BaseAgent
from schemas.models import AgentResponse, CandidateItem, RecommendationState
from services.llm_client import LLMClient


class MarketingAgent(BaseAgent):
    """
    Marketing/reason generation agent.

    Modes:
    - template: deterministic template-based reason
    - llm: batch LLM-generated structured reasons with template fallback

    LLM does not decide product IDs. It only explains already selected items.
    """

    def __init__(self):
        super().__init__(
            name="MarketingAgent",
            timeout_seconds=30.0,
            max_retries=1,
        )
        self.llm_client = LLMClient()

    async def _run(self, state: RecommendationState) -> AgentResponse:
        if not state.filtered_candidates:
            return AgentResponse(
                success=False,
                error="No filtered candidates found in RecommendationState.",
            )

        user_context = state.user_context
        top_brand, top_category, top_brands, top_categories = self._extract_user_preferences(
            user_context
        )

        marketing_mode = getattr(state, "marketing_mode", "template")

        if marketing_mode == "llm":
            return self._run_llm_batch(
                state=state,
                top_brand=top_brand,
                top_category=top_category,
                top_brands=top_brands,
                top_categories=top_categories,
            )

        final_items = []
        for item in state.filtered_candidates:
            reason = self._build_template_reason(
                item=item,
                top_brand=top_brand,
                top_category=top_category,
            )
            final_items.append(item.model_copy(update={"reason": reason}))

        return AgentResponse(
            success=True,
            data=final_items,
            fallback_used=False,
            metadata={
                "input_items": len(state.filtered_candidates),
                "output_items": len(final_items),
                "reason_type": "template",
                "llm_reason_mode": "none",
                "llm_success_count": 0,
                "llm_fallback_count": 0,
                "llm_errors_sample": [],
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            },
        )

    def _run_llm_batch(
        self,
        state: RecommendationState,
        top_brand: str | None,
        top_category: str | None,
        top_brands: list[str],
        top_categories: list[str],
    ) -> AgentResponse:
        items = state.filtered_candidates

        system_prompt = (
            "You are a recommendation explanation agent for an e-commerce system. "
            "Generate concise recommendation reasons for multiple candidate products. "
            "You must return valid JSON only. "
            "The JSON object must contain exactly these keys: reasons, style, risk. "
            "The value of reasons must be a list of objects. "
            "Each object must contain product_id and reason. "
            "Only use product_ids from the given items. "
            "Do not invent product facts. "
            "Do not mention unavailable information. "
            "Do not return scores. "
            "Do not include markdown."
        )

        user_payload = {
            "task_type": "batch_marketing_reason",
            "user_profile": {
                "top_categories": top_categories,
                "top_brands": top_brands,
            },
            "items": [
                {
                    "product_id": item.product_id,
                    "title": item.title,
                    "brand": item.brand,
                    "category": item.category,
                    "price": item.price,
                    "stock": item.stock,
                }
                for item in items
            ],
            "required_output_schema": {
                "reasons": [
                    {
                        "product_id": "string",
                        "reason": "string",
                    }
                ],
                "style": "concise",
                "risk": "low",
            },
        }

        result = self.llm_client.generate_json(
            system_prompt=system_prompt,
            user_payload=user_payload,
        )

        if not result.success:
            return self._fallback_all_to_template(
                items=items,
                top_brand=top_brand,
                top_category=top_category,
                error=result.error or "LLM batch call failed.",
                result=result,
            )

        reason_map, invalid_reason_items = self._parse_batch_reasons(
            result_data=result.data,
            valid_product_ids={str(item.product_id) for item in items},
        )

        final_items: list[CandidateItem] = []
        fallback_count = 0

        for item in items:
            pid = str(item.product_id)
            reason = reason_map.get(pid)

            if not reason:
                fallback_count += 1
                reason = self._build_template_reason(
                    item=item,
                    top_brand=top_brand,
                    top_category=top_category,
                )

            final_items.append(item.model_copy(update={"reason": reason}))

        success_count = len(items) - fallback_count
        fallback_used = fallback_count > 0

        return AgentResponse(
            success=True,
            data=final_items,
            fallback_used=fallback_used,
            metadata={
                "input_items": len(items),
                "output_items": len(final_items),
                "reason_type": "llm",
                "llm_reason_mode": "batch",
                "llm_success_count": success_count,
                "llm_fallback_count": fallback_count,
                "llm_errors_sample": invalid_reason_items[:3],
                "llm_provider": result.provider,
                "llm_model": result.model,
                "llm_latency_ms": result.latency_ms,
                "llm_batch_call_count": 1,
            },
        )

    def _fallback_all_to_template(
        self,
        items: list[CandidateItem],
        top_brand: str | None,
        top_category: str | None,
        error: str,
        result,
    ) -> AgentResponse:
        final_items = []

        for item in items:
            reason = self._build_template_reason(
                item=item,
                top_brand=top_brand,
                top_category=top_category,
            )
            final_items.append(item.model_copy(update={"reason": reason}))

        return AgentResponse(
            success=True,
            data=final_items,
            fallback_used=True,
            metadata={
                "input_items": len(items),
                "output_items": len(final_items),
                "reason_type": "llm",
                "llm_reason_mode": "batch",
                "llm_success_count": 0,
                "llm_fallback_count": len(items),
                "llm_errors_sample": [error],
                "llm_provider": result.provider,
                "llm_model": result.model,
                "llm_latency_ms": result.latency_ms,
                "llm_batch_call_count": 1,
            },
        )

    def _parse_batch_reasons(
        self,
        result_data: dict,
        valid_product_ids: set[str],
    ) -> tuple[dict[str, str], list[str]]:
        raw_reasons = result_data.get("reasons")
        reason_map: dict[str, str] = {}
        errors: list[str] = []

        if not isinstance(raw_reasons, list):
            return {}, ["Invalid LLM output: reasons is not a list."]

        for idx, x in enumerate(raw_reasons):
            if not isinstance(x, dict):
                errors.append(f"Reason item {idx} is not a dict.")
                continue

            product_id = str(x.get("product_id", ""))
            reason = x.get("reason")

            if product_id not in valid_product_ids:
                errors.append(f"Invalid product_id from LLM: {product_id}")
                continue

            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"Invalid reason for product_id: {product_id}")
                continue

            reason_map[product_id] = reason.strip()

        return reason_map, errors

    def _extract_user_preferences(self, user_context):
        top_brand = None
        top_category = None
        top_brands = []
        top_categories = []

        if user_context is not None:
            if user_context.brand_pref:
                sorted_brands = sorted(
                    user_context.brand_pref.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_brand = sorted_brands[0][0]
                top_brands = [x[0] for x in sorted_brands[:5]]

            if user_context.category_pref:
                sorted_categories = sorted(
                    user_context.category_pref.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_category = sorted_categories[0][0]
                top_categories = [x[0] for x in sorted_categories[:5]]

        return top_brand, top_category, top_brands, top_categories

    def _build_template_reason(
        self,
        item: CandidateItem,
        top_brand: str | None,
        top_category: str | None,
    ) -> str:
        title = item.title or "this item"
        brand = item.brand or "this brand"
        category = item.category or "beauty products"

        if top_brand and item.brand and str(item.brand).lower() == str(top_brand).lower():
            return (
                f"Recommended because you recently interacted with products from {brand}, "
                f"and {title} matches that brand preference."
            )

        if top_category and item.category and str(item.category).lower() == str(top_category).lower():
            return (
                f"Recommended based on your recent interest in {category}; "
                f"{title} may fit your personal-care routine."
            )

        if item.stock is not None and item.stock > 0:
            return (
                f"Recommended as an available {category} item from {brand}, "
                f"ranked by your recent behavior sequence."
            )

        return (
            "Recommended by the generative recommendation model based on your recent behavior sequence."
        )