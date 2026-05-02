from langgraph.graph import END, StateGraph

from agents.filter import FilterAgent
from agents.generative_rec import GenerativeRecAgent
from agents.marketing import MarketingAgent
from agents.user_profile import UserProfileAgent
from schemas.models import RecommendationState

from agents.llm_rerank import LLMRerankAgent


class GenRecWorkflow:
    """
    LangGraph workflow for GenRec-Agent.

    Flow:
    UserProfileAgent
      -> GenerativeRecAgent
      -> FilterAgent
      -> MarketingAgent
      -> END
    """

    def __init__(self):
        self.user_profile_agent = UserProfileAgent()
        self.generative_rec_agent = GenerativeRecAgent()
        self.filter_agent = FilterAgent(min_stock=1, remove_history=True)
        self.marketing_agent = MarketingAgent()

        self.graph = self._build_graph()

        self.rerank_agent = LLMRerankAgent()

    def _build_graph(self):
        builder = StateGraph(RecommendationState)

        builder.add_node("user_profile", self.user_profile_node)
        builder.add_node("generative_rec", self.generative_rec_node)
        builder.add_node("filter", self.filter_node)
        builder.add_node("rerank", self.rerank_node)
        builder.add_node("marketing", self.marketing_node)

        builder.set_entry_point("user_profile")

        builder.add_edge("user_profile", "generative_rec")
        builder.add_edge("generative_rec", "filter")
        builder.add_conditional_edges(
            "filter",
            self.route_after_filter,
            {
                "rerank": "rerank",
                "marketing": "marketing",
            },
        )

        builder.add_edge("rerank", "marketing")
        builder.add_edge("marketing", END)

        return builder.compile()

    async def user_profile_node(
        self,
        state: RecommendationState,
    ) -> RecommendationState:
        response = await self.user_profile_agent.run(state)

        state.trace.append(
            {
                "agent": "UserProfileAgent",
                "success": response.success,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used,
                "metadata": response.metadata,
                "error": response.error,
            }
        )

        if not response.success:
            state.fallback_used = True
            return state

        state.user_context = response.data
        state.fallback_used = state.fallback_used or response.fallback_used
        return state

    async def generative_rec_node(
        self,
        state: RecommendationState,
    ) -> RecommendationState:
        if state.user_context is None:
            state.trace.append(
                {
                    "agent": "GenerativeRecAgent",
                    "success": False,
                    "latency_ms": 0.0,
                    "fallback_used": True,
                    "metadata": {},
                    "error": "Skipped because user_context is missing.",
                }
            )
            state.fallback_used = True
            return state

        response = await self.generative_rec_agent.run(state)

        state.trace.append(
            {
                "agent": "GenerativeRecAgent",
                "success": response.success,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used,
                "metadata": response.metadata,
                "error": response.error,
            }
        )

        if not response.success:
            state.fallback_used = True
            return state

        state.candidates = response.data
        state.fallback_used = state.fallback_used or response.fallback_used
        return state

    async def filter_node(
        self,
        state: RecommendationState,
    ) -> RecommendationState:
        if not state.candidates:
            state.trace.append(
                {
                    "agent": "FilterAgent",
                    "success": False,
                    "latency_ms": 0.0,
                    "fallback_used": True,
                    "metadata": {},
                    "error": "Skipped because candidates are missing.",
                }
            )
            state.fallback_used = True
            return state

        response = await self.filter_agent.run(state)

        state.trace.append(
            {
                "agent": "FilterAgent",
                "success": response.success,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used,
                "metadata": response.metadata,
                "error": response.error,
            }
        )

        if not response.success:
            state.fallback_used = True
            return state

        state.filtered_candidates = response.data
        state.final_items = response.data
        return state

    async def rerank_node(
        self,
        state: RecommendationState,
    ) -> RecommendationState:
        if not state.filtered_candidates:
            state.trace.append(
                {
                    "agent": "LLMRerankAgent",
                    "success": False,
                    "latency_ms": 0.0,
                    "fallback_used": True,
                    "metadata": {},
                    "error": "Skipped because filtered_candidates are missing.",
                }
            )
            state.fallback_used = True
            return state

        response = await self.rerank_agent.run(state)

        state.trace.append(
            {
                "agent": "LLMRerankAgent",
                "success": response.success,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used,
                "metadata": response.metadata,
                "error": response.error,
            }
        )

        if not response.success:
            state.fallback_used = True
            return state

        state.filtered_candidates = response.data
        state.final_items = response.data
        state.fallback_used = state.fallback_used or response.fallback_used
        return state

    async def marketing_node(
        self,
        state: RecommendationState,
    ) -> RecommendationState:
        if not state.filtered_candidates:
            state.trace.append(
                {
                    "agent": "MarketingAgent",
                    "success": False,
                    "latency_ms": 0.0,
                    "fallback_used": True,
                    "metadata": {},
                    "error": "Skipped because filtered_candidates are missing.",
                }
            )
            state.fallback_used = True
            return state

        response = await self.marketing_agent.run(state)

        state.trace.append(
            {
                "agent": "MarketingAgent",
                "success": response.success,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used,
                "metadata": response.metadata,
                "error": response.error,
            }
        )

        if not response.success:
            state.fallback_used = True
            return state

        state.final_items = response.data
        state.fallback_used = state.fallback_used or response.fallback_used
        return state

    async def ainvoke(self, state: RecommendationState) -> RecommendationState:
        result = await self.graph.ainvoke(state)

        if isinstance(result, RecommendationState):
            return result

        if isinstance(result, dict):
            return RecommendationState(**result)

        raise TypeError(f"Unexpected workflow result type: {type(result)}")

    def route_after_filter(self, state: RecommendationState) -> str:
        if getattr(state, "rerank_mode", "none") == "llm":
            return "rerank"
        return "marketing"