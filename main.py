import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph.workflow import GenRecWorkflow
from schemas.models import RecommendationState


app = FastAPI(
    title="GenRec-Agent",
    description="A LangGraph-based multi-agent generative recommendation system.",
    version="0.1.0",
)

workflow = GenRecWorkflow()


class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User ID from processed Amazon Beauty dataset.")
    top_k: int = Field(10, ge=1, le=50)
    mode: str = Field(
        "genrec_gru",
        description="Recommendation mode: genrec_gru, semantic_neighbor, or popularity.",
    )
    marketing_mode: str = Field(
        "template",
        description="Marketing reason mode: template or llm.",
    )
    rerank_mode: str = Field(
        "none",
        description="Candidate rerank mode: none or llm.",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "GenRec-Agent",
        "workflow": "LangGraph",
    }


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    start = time.perf_counter()

    state = RecommendationState(
        request_id=str(uuid4()),
        user_id=request.user_id,
        top_k=request.top_k,
        mode=request.mode,
        marketing_mode=request.marketing_mode,
        rerank_mode=request.rerank_mode,
    )

    result = await workflow.ainvoke(state)

    if result.user_context is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "User not found or user profile construction failed.",
                "user_id": request.user_id,
                "trace": result.trace,
            },
        )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "request_id": result.request_id,
        "user_id": result.user_id,
        "top_k": result.top_k,
        "mode": result.mode,
        "marketing_mode": result.marketing_mode,
        "rerank_mode": result.rerank_mode,
        "fallback_used": result.fallback_used,
        "latency_ms": latency_ms,
        "items": [item.model_dump() for item in result.final_items],
        "trace": result.trace,
    }