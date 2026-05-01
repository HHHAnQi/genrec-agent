import json
from collections import Counter
from pathlib import Path

import pandas as pd

from agents.base import BaseAgent
from schemas.models import AgentResponse, RecommendationState, UserContext


class UserProfileAgent(BaseAgent):
    def __init__(
        self,
        user_sequences_path: str = "datasets/processed/user_sequences.json",
        items_path: str = "datasets/processed/items.csv",
    ):
        super().__init__(
            name="UserProfileAgent",
            timeout_seconds=5.0,
            max_retries=1,
        )

        self.user_sequences_path = Path(user_sequences_path)
        self.items_path = Path(items_path)

        with open(self.user_sequences_path, "r", encoding="utf-8") as f:
            self.user_sequences = json.load(f)

        self.items_df = pd.read_csv(self.items_path)
        self.item_info = {
            str(row["product_id"]): row.to_dict()
            for _, row in self.items_df.iterrows()
        }

    async def _run(self, state: RecommendationState) -> AgentResponse:
        user_id = str(state.user_id)

        if user_id not in self.user_sequences:
            return AgentResponse(
                success=False,
                error=f"User not found in user_sequences: {user_id}",
                fallback_used=True,
                metadata={"known_users": len(self.user_sequences)},
            )

        seq = [str(x) for x in self.user_sequences[user_id]]

        # 推荐时一般不要把最后一个 target 当作历史。这里用 seq[:-1] 更贴近离线测试。
        recent_items = seq[:-1] if len(seq) > 1 else seq

        category_counter = Counter()
        brand_counter = Counter()

        for pid in recent_items:
            info = self.item_info.get(pid, {})
            category = str(info.get("category", "unknown"))
            brand = str(info.get("brand", "unknown"))

            if category and category.lower() != "nan":
                category_counter[category] += 1
            if brand and brand.lower() != "nan":
                brand_counter[brand] += 1

        total = max(len(recent_items), 1)

        category_pref = {
            k: round(v / total, 4)
            for k, v in category_counter.most_common(10)
        }
        brand_pref = {
            k: round(v / total, 4)
            for k, v in brand_counter.most_common(10)
        }

        context = UserContext(
            user_id=user_id,
            recent_clicks=recent_items,
            recent_purchases=recent_items,
            category_pref=category_pref,
            brand_pref=brand_pref,
        )

        return AgentResponse(
            success=True,
            data=context,
            metadata={
                "history_len": len(recent_items),
                "top_categories": category_counter.most_common(3),
                "top_brands": brand_counter.most_common(3),
            },
        )