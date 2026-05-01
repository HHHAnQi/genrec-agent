from agents.base import BaseAgent
from schemas.models import AgentResponse, CandidateItem, RecommendationState


class FilterAgent(BaseAgent):
    def __init__(
        self,
        min_stock: int = 1,
        remove_history: bool = True,
    ):
        super().__init__(
            name="FilterAgent",
            timeout_seconds=5.0,
            max_retries=1,
        )
        self.min_stock = min_stock
        self.remove_history = remove_history

    async def _run(self, state: RecommendationState) -> AgentResponse:
        if not state.candidates:
            return AgentResponse(
                success=False,
                error="No candidates found in RecommendationState.",
            )

        history = set()
        if state.user_context is not None:
            history.update(str(x) for x in state.user_context.recent_clicks)
            history.update(str(x) for x in state.user_context.recent_purchases)

        filtered: list[CandidateItem] = []
        removed_by_stock = 0
        removed_by_history = 0
        seen = set()
        removed_duplicate = 0

        for item in state.candidates:
            pid = str(item.product_id)

            if pid in seen:
                removed_duplicate += 1
                continue
            seen.add(pid)

            if self.remove_history and pid in history:
                removed_by_history += 1
                continue

            if item.stock is not None and item.stock < self.min_stock:
                removed_by_stock += 1
                continue

            filtered.append(item)

            if len(filtered) >= state.top_k:
                break

        return AgentResponse(
            success=True,
            data=filtered,
            metadata={
                "input_candidates": len(state.candidates),
                "output_candidates": len(filtered),
                "removed_by_stock": removed_by_stock,
                "removed_by_history": removed_by_history,
                "removed_duplicate": removed_duplicate,
                "min_stock": self.min_stock,
            },
        )