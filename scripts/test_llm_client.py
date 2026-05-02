import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.llm_client import LLMClient


def main():
    client = LLMClient()

    result = client.generate_json(
        system_prompt=(
            "You are a recommendation explanation agent. "
            "Your task is to generate ONE concise recommendation reason. "
            "Return valid JSON only. "
            "The JSON object must contain exactly these keys: "
            "reason, style, risk. "
            "The value of reason must be a natural-language string. "
            "Do not return score. Do not return product IDs. "
            "Do not include markdown."
        ),
        user_payload={
            "task_type": "marketing_reason",
            "user_profile": {
                "top_categories": ["All Beauty"],
                "top_brands": ["Hylunia"],
            },
            "item": {
                "title": "Hylunia Hydrate Body Wash",
                "brand": "Hylunia",
                "category": "All Beauty",
            },
            "required_output_schema": {
                "reason": "string",
                "style": "concise",
                "risk": "low",
            },
        },
    )

    print("success:", result.success)
    print("provider:", result.provider)
    print("model:", result.model)
    print("latency_ms:", result.latency_ms)
    print("error:", result.error)
    print("data:", result.data)


if __name__ == "__main__":
    main()