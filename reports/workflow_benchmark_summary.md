# Workflow Benchmark Summary

This report benchmarks GenRec-Agent workflow modes from an AI application engineering perspective.

It focuses on endpoint-level latency, success rate, fallback rate, and agent-level latency breakdown.

## Configuration

- Test path: `datasets/processed/test.jsonl`
- Number of users: `5`
- Top-K: `10`
- LLM reason Top-N: `3`

## Mode Comparison

| Mode | Success Rate | Any Fallback Flag Rate | Avg Latency ms | P50 ms | P95 ms | Avg Items | invalid_ids | Query Guardrail | LLM Reason Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| genrec_gru_template | 1.00 | 0.00 | 7.7 | 3.3 | 20.8 | 10.0 | 0 | 0 | 0 |
| genrec_gru_llm_full | 1.00 | 0.80 | 5381.3 | 4968.0 | 7184.8 | 10.0 | 0 | 0 | 6 |
| llm_query_recall_template | 1.00 | 0.00 | 1876.9 | 1845.3 | 2071.5 | 9.8 | 0 | 2 | 0 |
| llm_query_recall_llm_full | 1.00 | 0.20 | 7269.3 | 7476.2 | 8113.6 | 10.0 | 0 | 1 | 1 |

## Agent-level Latency Breakdown

### genrec_gru_template

| Agent | Avg ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| FilterAgent | 0.1 | 0.1 | 0.2 |
| GenerativeRecAgent | 4.0 | 0.9 | 13.5 |
| MarketingAgent | 0.1 | 0.1 | 0.1 |
| UserProfileAgent | 0.2 | 0.1 | 0.5 |

### genrec_gru_llm_full

| Agent | Avg ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| FilterAgent | 0.1 | 0.1 | 0.2 |
| GenerativeRecAgent | 1.4 | 1.6 | 1.6 |
| LLMRerankAgent | 3092.9 | 2473.9 | 5146.1 |
| MarketingAgent | 2280.7 | 2305.2 | 2561.4 |
| UserProfileAgent | 0.2 | 0.2 | 0.3 |

### llm_query_recall_template

| Agent | Avg ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| FilterAgent | 0.1 | 0.1 | 0.1 |
| LLMQueryRecallAgent | 1873.3 | 1842.1 | 2068.0 |
| MarketingAgent | 0.1 | 0.1 | 0.2 |
| UserProfileAgent | 0.2 | 0.1 | 0.2 |

### llm_query_recall_llm_full

| Agent | Avg ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| FilterAgent | 0.1 | 0.1 | 0.1 |
| LLMQueryRecallAgent | 1775.4 | 1790.9 | 1843.1 |
| LLMRerankAgent | 3037.7 | 3061.9 | 3814.6 |
| MarketingAgent | 2449.6 | 2558.1 | 2559.7 |
| UserProfileAgent | 0.2 | 0.2 | 0.3 |

## LLM Query Recall Samples

### llm_query_recall_template

- `facial care kit with sheet masks and hair towel wrap`
- `anti-aging facial serum oil moisturizer`
- `personal care grooming tools oral care nail care skincare`
- `faux mink false eyelashes natural look volume bulk pack`
- `beauty tools and accessories for makeup and nail care`

### llm_query_recall_llm_full

- `facial sheet mask cleanser skincare kit`
- `anti-aging facial serum with hyaluronic acid and retinol`
- `personal care grooming tools oral care nail care skincare`
- `faux mink false eyelashes natural look volume`
- `beauty accessories makeup tools nail care`

## Interpretation

- `genrec_gru_template` is the fastest non-LLM baseline.
- LLM-enabled modes increase latency because they call DeepSeek for query generation, candidate reranking, and/or recommendation reasons.
- `llm_reason_top_n` controls marketing latency by generating LLM reasons only for the top-N items.
- `invalid_ids_count=0` indicates candidate-constrained LLM reranking did not output out-of-candidate product IDs.
- `query_guardrail_count` shows how many ambiguous vector-retrieval results were filtered by the query recall guardrail.
