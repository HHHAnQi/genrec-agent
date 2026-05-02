# DeepSeek LLM Agent Results

## Configuration

- LLM Provider: DeepSeek API
- Model: deepseek-chat
- LLM Client: OpenAI-compatible `/chat/completions`
- Rerank mode: llm
- Marketing mode: llm
- Marketing reason mode: batch

## Validated Capabilities

| Capability | Result |
|---|---|
| DeepSeek API connection | Passed |
| LLMClient JSON parsing | Passed |
| LLMRerankAgent candidate-constrained reranking | Passed |
| Invalid product ID check | Passed, invalid_ids=[] |
| Batch LLMMarketingAgent | Passed |
| Marketing fallback | Passed |
| Rerank fallback | Passed |
| Full LangGraph workflow | Passed |

## Observed Latency

| Component | Latency |
|---|---:|
| LLMRerankAgent | ~2.7 - 2.8s |
| Batch LLMMarketingAgent, Top5 | ~2.5s |
| Batch LLMMarketingAgent, Top10 | ~4.7s |

## Key Engineering Design

The LLM does not directly generate product IDs from the full item pool. Instead, the GRU GenRec model first generates candidate products, and the LLMRerankAgent only reranks within the candidate set. The system validates that all LLM-returned product IDs belong to the original candidate list, preventing product hallucination.

The MarketingAgent uses batch LLM generation to produce recommendation reasons for all final items in a single API call. If the LLM output is invalid or incomplete, the system falls back to template-based reasons for affected items.