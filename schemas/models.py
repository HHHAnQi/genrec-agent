from typing import Any, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 10
    mode: str = "genrec_gru"


class UserContext(BaseModel):
    user_id: str
    recent_clicks: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    category_pref: dict[str, float] = Field(default_factory=dict)
    brand_pref: dict[str, float] = Field(default_factory=dict)


class CandidateItem(BaseModel):
    product_id: str
    score: float = 0.0
    source: str = "genrec_gru"
    category: Optional[str] = None
    brand: Optional[str] = None
    title: Optional[str] = None
    price: Optional[float] = None
    stock: Optional[int] = None
    reason: Optional[str] = None


class AgentResponse(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    fallback_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationState(BaseModel):
    request_id: str
    user_id: str
    top_k: int = 10
    mode: str = "genrec_gru"

    user_context: Optional[UserContext] = None
    candidates: list[CandidateItem] = Field(default_factory=list)
    filtered_candidates: list[CandidateItem] = Field(default_factory=list)
    final_items: list[CandidateItem] = Field(default_factory=list)

    fallback_used: bool = False
    trace: list[dict[str, Any]] = Field(default_factory=list)