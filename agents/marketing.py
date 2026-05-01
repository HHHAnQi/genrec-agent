from agents.base import BaseAgent
from schemas.models import AgentResponse, CandidateItem, RecommendationState


class MarketingAgent(BaseAgent):
    """
    Lightweight template-based marketing/reason generation agent.

    It does not call an external LLM API, so it is stable and reproducible.
    Later, this can be replaced by an LLM-based copywriting agent.
    """

    def __init__(self):
        super().__init__(
            name="MarketingAgent",
            timeout_seconds=5.0,
            max_retries=1,
        )

    async def _run(self, state: RecommendationState) -> AgentResponse:
        if not state.filtered_candidates:
            return AgentResponse(
                success=False,
                error="No filtered candidates found in RecommendationState.",
            )

        user_context = state.user_context

        top_brand = None
        top_category = None

        if user_context is not None:
            if user_context.brand_pref:
                top_brand = max(user_context.brand_pref.items(), key=lambda x: x[1])[0]
            if user_context.category_pref:
                top_category = max(user_context.category_pref.items(), key=lambda x: x[1])[0]

        final_items: list[CandidateItem] = []

        for item in state.filtered_candidates:
            reason = self._build_reason(
                item=item,
                top_brand=top_brand,
                top_category=top_category,
            )

            updated_item = item.model_copy(update={"reason": reason})
            final_items.append(updated_item)

        return AgentResponse(
            success=True,
            data=final_items,
            metadata={
                "input_items": len(state.filtered_candidates),
                "output_items": len(final_items),
                "reason_type": "template",
            },
        )

    def _build_reason(
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
            f"Recommended by the generative recommendation model based on your recent behavior sequence."
        )