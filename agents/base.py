import asyncio
import time
from abc import ABC, abstractmethod

from schemas.models import AgentResponse, RecommendationState


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 1,
    ):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def run(self, state: RecommendationState) -> AgentResponse:
        start = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._run(state),
                    timeout=self.timeout_seconds,
                )

                response.latency_ms = (time.perf_counter() - start) * 1000
                response.metadata["agent_name"] = self.name
                response.metadata["attempt"] = attempt + 1

                return response

            except Exception as e:
                last_error = e

        return AgentResponse(
            success=False,
            data=None,
            error=repr(last_error),
            latency_ms=(time.perf_counter() - start) * 1000,
            fallback_used=True,
            metadata={"agent_name": self.name},
        )

    @abstractmethod
    async def _run(self, state: RecommendationState) -> AgentResponse:
        raise NotImplementedError