import pandas as pd

from agents.base import BaseAgent
from recommender.inference import GenRecInference
from schemas.models import AgentResponse, CandidateItem, RecommendationState


class GenerativeRecAgent(BaseAgent):
    def __init__(
        self,
        items_path: str = "datasets/processed/items.csv",
        model_path: str = "models/genrec_gru.pt",
        train_path: str = "datasets/processed/train.jsonl",
        semantic_ids_path: str = "datasets/processed/semantic_ids.json",
        sid_to_items_path: str = "datasets/processed/sid_to_items.json",
        top_clusters: int = 5,
    ):
        super().__init__(
            name="GenerativeRecAgent",
            timeout_seconds=10.0,
            max_retries=1,
        )

        self.engine = GenRecInference(
            model_path=model_path,
            train_path=train_path,
            semantic_ids_path=semantic_ids_path,
            sid_to_items_path=sid_to_items_path,
        )
        self.top_clusters = top_clusters

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

        history = (
            state.user_context.recent_clicks
            or state.user_context.recent_purchases
        )

        result = self.engine.recommend(
            history=history,
            top_k=state.top_k,
            mode=state.mode,
            top_clusters=self.top_clusters,
            exclude_history=True,
        )

        candidates = []

        for rank, product_id in enumerate(result["items"]):
            info = self.item_info.get(str(product_id), {})

            score = 1.0 / (rank + 1)

            candidates.append(
                CandidateItem(
                    product_id=str(product_id),
                    score=score,
                    source=result["used_mode"],
                    title=info.get("title"),
                    category=info.get("category"),
                    brand=info.get("brand"),
                    price=info.get("price"),
                    stock=info.get("stock"),
                )
            )

        return AgentResponse(
            success=True,
            data=candidates,
            fallback_used=result["used_mode"] != result["requested_mode"],
            metadata={
                "requested_mode": result["requested_mode"],
                "used_mode": result["used_mode"],
                "model_loaded": result["model_loaded"],
                "load_error": result["load_error"],
                "num_returned": len(candidates),
            },
        )