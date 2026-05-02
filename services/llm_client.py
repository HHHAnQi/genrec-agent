import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class LLMResult:
    success: bool
    data: dict[str, Any]
    error: Optional[str] = None
    latency_ms: float = 0.0
    provider: str = "mock"
    model: str = "mock"


class LLMClient:
    """
    OpenAI-compatible LLM client with mock fallback.

    Supported providers:
    - mock: deterministic local fake LLM, no API key required
    - openai: OpenAI-compatible chat completions endpoint
    - deepseek: DeepSeek OpenAI-compatible API
    - ollama: local OpenAI-compatible server, e.g. Qwen2.5 via Ollama

    Designed for Agent usage:
    - JSON-only output
    - timeout control
    - structured error
    - mock / API / local-provider abstraction
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.timeout_seconds = float(
            timeout_seconds or os.getenv("LLM_TIMEOUT_SECONDS", "10")
        )

        if self.provider == "deepseek":
            default_base_url = "https://api.deepseek.com"
            default_model = "deepseek-chat"
        elif self.provider == "ollama":
            default_base_url = "http://127.0.0.1:11434/v1"
            default_model = "qwen2.5:3b"
        elif self.provider == "openai":
            default_base_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"
        else:
            default_base_url = "https://api.openai.com/v1"
            default_model = "mock"

        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL", default_base_url)
        ).rstrip("/")

        if self.provider == "mock":
            self.model = "mock"
        else:
            self.model = model or os.getenv("LLM_MODEL", default_model)

        # If provider requires API key but key is missing, fallback to mock.
        # Ollama local server can use a dummy key.
        if not self.api_key and self.provider not in {"mock", "ollama"}:
            self.provider = "mock"
            self.model = "mock"
            self.base_url = "https://api.openai.com/v1"

    def generate_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> LLMResult:
        start = time.perf_counter()

        if self.provider == "mock":
            result = self._mock_generate(user_payload)
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result

        if self.provider in {"openai", "deepseek", "ollama"}:
            return self._openai_compatible_generate(
                system_prompt=system_prompt,
                user_payload=user_payload,
                start=start,
            )

        return LLMResult(
            success=False,
            data={},
            error=f"Unsupported LLM provider: {self.provider}",
            latency_ms=(time.perf_counter() - start) * 1000,
            provider=self.provider,
            model=self.model,
        )

    def _mock_generate(self, user_payload: dict[str, Any]) -> LLMResult:
        """
        Mock provider for local reproducibility.

        Supported mock tasks:
        1. Batch marketing reason generation:
           user_payload contains "items"

        2. Candidate reranking:
           user_payload contains "candidates"

        3. Single-item marketing reason:
           user_payload contains "item"
        """

        # 1. Mock batch marketing mode
        if "items" in user_payload:
            items = user_payload.get("items", [])
            user_profile = user_payload.get("user_profile", {})

            top_categories = user_profile.get("top_categories") or []
            top_brands = user_profile.get("top_brands") or []

            reasons = []

            for item in items:
                product_id = str(item.get("product_id"))
                title = item.get("title") or "this item"
                brand = item.get("brand") or "this brand"
                category = item.get("category") or "beauty products"

                if top_brands and brand in top_brands:
                    reason = (
                        f"Because you recently showed interest in {brand}, "
                        f"{title} may be a good match for your preferences."
                    )
                elif top_categories:
                    reason = (
                        f"Based on your recent interest in {top_categories[0]}, "
                        f"{title} from {brand} may fit your personal-care routine."
                    )
                else:
                    reason = (
                        f"{title} is recommended based on your recent behavior sequence "
                        f"and its relevance within {category}."
                    )

                reasons.append(
                    {
                        "product_id": product_id,
                        "reason": reason,
                    }
                )

            return LLMResult(
                success=True,
                data={
                    "reasons": reasons,
                    "style": "concise",
                    "risk": "low",
                },
                error=None,
                provider="mock",
                model="mock",
            )

        # 2. Mock rerank mode
        if "candidates" in user_payload:
            candidates = user_payload.get("candidates", [])
            user_profile = user_payload.get("user_profile", {})

            top_brands = set(user_profile.get("top_brands") or [])
            top_categories = set(user_profile.get("top_categories") or [])

            def score_candidate(x: dict[str, Any]) -> float:
                score = float(x.get("score") or 0.0)
                brand = x.get("brand")
                category = x.get("category")

                bonus = 0.0
                if brand in top_brands:
                    bonus += 0.2
                if category in top_categories:
                    bonus += 0.1

                return score + bonus

            reranked = sorted(
                candidates,
                key=score_candidate,
                reverse=True,
            )

            reranked_product_ids = [
                str(x["product_id"])
                for x in reranked
                if x.get("product_id") is not None
            ]

            return LLMResult(
                success=True,
                data={
                    "reranked_product_ids": reranked_product_ids,
                    "summary": "Mock rerank based on candidate score and user profile alignment.",
                    "risk": "low",
                },
                error=None,
                provider="mock",
                model="mock",
            )

        # 3. Mock single-item marketing mode
        item = user_payload.get("item", {})
        user_profile = user_payload.get("user_profile", {})

        title = item.get("title") or "this item"
        brand = item.get("brand") or "this brand"
        category = item.get("category") or "beauty products"

        top_categories = user_profile.get("top_categories") or []
        top_brands = user_profile.get("top_brands") or []

        if top_brands and brand in top_brands:
            reason = (
                f"Because you recently showed interest in {brand}, "
                f"{title} may be a good match for your preferences."
            )
        elif top_categories:
            reason = (
                f"Based on your recent interest in {top_categories[0]}, "
                f"{title} from {brand} may fit your personal-care routine."
            )
        else:
            reason = (
                f"{title} is recommended based on your recent behavior sequence "
                f"and its relevance within {category}."
            )

        return LLMResult(
            success=True,
            data={
                "reason": reason,
                "style": "concise",
                "risk": "low",
            },
            error=None,
            provider="mock",
            model="mock",
        )

    def _openai_compatible_generate(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        start: float,
    ) -> LLMResult:
        url = f"{self.base_url}/chat/completions"

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]

        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        # Some OpenAI-compatible providers support JSON mode.
        # If unsupported, retry once without response_format.
        if os.getenv("LLM_USE_RESPONSE_FORMAT", "true").lower() == "true":
            request_body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                response = client.post(url, headers=headers, json=request_body)

            latency_ms = (time.perf_counter() - start) * 1000

            # Retry without JSON mode if the provider rejects response_format.
            if response.status_code >= 400 and "response_format" in request_body:
                request_body.pop("response_format", None)
                with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                    response = client.post(url, headers=headers, json=request_body)
                latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code >= 400:
                return LLMResult(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}: {response.text[:500]}",
                    latency_ms=latency_ms,
                    provider=self.provider,
                    model=self.model,
                )

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            parsed = self._parse_json_content(content)

            return LLMResult(
                success=True,
                data=parsed,
                error=None,
                latency_ms=latency_ms,
                provider=self.provider,
                model=self.model,
            )

        except Exception as e:
            return LLMResult(
                success=False,
                data={},
                error=repr(e),
                latency_ms=(time.perf_counter() - start) * 1000,
                provider=self.provider,
                model=self.model,
            )

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        content = content.strip()

        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()

        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError("LLM output is not a JSON object.")

        return parsed