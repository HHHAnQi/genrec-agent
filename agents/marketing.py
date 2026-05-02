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

        llm_reason_top_n = int(getattr(state, "llm_reason_top_n", len(items)))
        llm_reason_top_n = max(0, min(llm_reason_top_n, len(items)))

        llm_items = items[:llm_reason_top_n]
        template_items = items[llm_reason_top_n:]

        # If llm_reason_top_n = 0, disable LLM reason generation.
        if not llm_items:
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
                fallback_used=False,
                metadata={
                    "input_items": len(items),
                    "output_items": len(final_items),
                    "reason_type": "template",
                    "llm_reason_mode": "top_n_disabled",
                    "llm_reason_top_n": llm_reason_top_n,
                    "llm_batch_input_items": 0,
                    "template_reason_items": len(template_items),
                    "llm_success_count": 0,
                    "llm_fallback_count": 0,
                    "llm_errors_sample": [],
                    "llm_invalid_reason_count": 0,
                    "llm_quality_fallback_count": 0,
                    "llm_provider": self.llm_client.provider,
                    "llm_model": self.llm_client.model,
                    "llm_latency_ms": 0.0,
                    "llm_batch_call_count": 0,
                },
            )

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
                for item in llm_items
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
            valid_product_ids={str(item.product_id) for item in llm_items},
        )

        final_items: list[CandidateItem] = []
        fallback_count = 0
        quality_errors: list[str] = []
        template_reason_count = 0

        llm_item_ids = {str(item.product_id) for item in llm_items}

        for item in items:
            pid = str(item.product_id)

            # Items outside top_n use template reasons directly.
            if pid not in llm_item_ids:
                template_reason_count += 1
                reason = self._build_template_reason(
                    item=item,
                    top_brand=top_brand,
                    top_category=top_category,
                )
                final_items.append(item.model_copy(update={"reason": reason}))
                continue

            reason = reason_map.get(pid)

            if not reason:
                fallback_count += 1
                quality_errors.append(f"{pid}:missing_reason")
                reason = self._build_template_reason(
                    item=item,
                    top_brand=top_brand,
                    top_category=top_category,
                )
            else:
                valid, error = self._is_reason_valid(reason, item)
                if not valid:
                    fallback_count += 1
                    quality_errors.append(f"{pid}:{error}")
                    reason = self._build_template_reason(
                        item=item,
                        top_brand=top_brand,
                        top_category=top_category,
                    )

            final_items.append(item.model_copy(update={"reason": reason}))

        success_count = len(llm_items) - fallback_count
        fallback_used = fallback_count > 0
        all_errors = invalid_reason_items + quality_errors

        return AgentResponse(
            success=True,
            data=final_items,
            fallback_used=fallback_used,
            metadata={
                "input_items": len(items),
                "output_items": len(final_items),
                "reason_type": "llm",
                "llm_reason_mode": "batch_top_n",
                "llm_reason_top_n": llm_reason_top_n,
                "llm_batch_input_items": len(llm_items),
                "template_reason_items": template_reason_count,
                "llm_success_count": success_count,
                "llm_fallback_count": fallback_count,
                "llm_errors_sample": all_errors[:5],
                "llm_invalid_reason_count": len(invalid_reason_items),
                "llm_quality_fallback_count": len(quality_errors),
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

    
    def _is_reason_valid(
        self,
        reason: str,
        item: CandidateItem,
    ) -> tuple[bool, str | None]:
        if not isinstance(reason, str) or not reason.strip():
            return False, "empty_reason"

        reason = reason.strip()
        reason_lower = reason.lower()
        words = reason.split()

        if len(words) < 6:
            return False, "reason_too_short"

        if len(reason) > 260:
            return False, "reason_too_long"

        blocked_terms = [
            "cheapest",
            "best price",
            "guaranteed",
            "guarantee",
            "free shipping",
            "discount",
            "cure",
            "medical",
            "treat disease",
            "doctor recommended",
            "clinically proven",
        ]

        for term in blocked_terms:
            if term in reason_lower:
                return False, f"blocked_term:{term}"

        # If price is missing or simulated, avoid unsupported price/discount claims.
        price_invalid = item.price is None or item.price < 0
        if price_invalid:
            unsupported_price_terms = [
                "$",
                "price",
                "cheap",
                "cheaper",
                "affordable",
                "discount",
                "sale",
                "deal",
                "low cost",
            ]
            for term in unsupported_price_terms:
                if term in reason_lower:
                    return False, f"unsupported_price_claim:{term}"

        if not self._mentions_item_signal(reason_lower, item):
            return False, "not_item_specific"

        return True, None

    def _mentions_item_signal(
        self,
        reason_lower: str,
        item: CandidateItem,
    ) -> bool:
        signals = []

        if item.brand:
            signals.append(str(item.brand).lower())

        if item.category:
            signals.append(str(item.category).lower())

        if item.title:
            title_tokens = [
                token.strip(" ,.-_/()[]{}")
                for token in str(item.title).lower().split()
            ]
            title_tokens = [
                token
                for token in title_tokens
                if len(token) >= 4
                and token not in {
                    "with",
                    "from",
                    "this",
                    "that",
                    "pack",
                    "count",
                    "new",
                    "the",
                    "and",
                    "for",
                }
            ]
            signals.extend(title_tokens[:8])

        for signal in signals:
            if signal and signal in reason_lower:
                return True

        return False