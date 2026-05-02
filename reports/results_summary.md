# GenRec-Agent Results Summary

This report summarizes the main offline recommendation results, API benchmark results, fallback validation, and DeepSeek LLM-enhanced agent results for GenRec-Agent.

---
## v0.3.0 LLM-guided Semantic Query Recall

### Motivation

Previous versions used LLMs mainly for candidate reranking and recommendation explanation. In v0.3.0, we add LLM-guided candidate recall: the LLM generates a semantic query from user behavior, and the system maps it to real catalog products through local vector retrieval.

### Design

The LLM does not directly generate product IDs. It only generates a semantic query and intent summary. Product candidates are retrieved from the local item embedding index and then processed by the existing filtering, reranking, and marketing pipeline.

### Validated Pipeline

```text
UserProfileAgent
→ LLMQueryRecallAgent
→ FilterAgent
→ LLMRerankAgent
→ MarketingAgent
```

# Key Trace Signals
```table
Field	Expected Meaning
mode=llm_query_recall	Use LLM-guided semantic query recall
llm_decides_products=false	LLM does not generate product IDs
llm_candidate_scope=vector_retrieval_from_local_catalog	Candidates come from local catalog retrieval
retrieval_backend=numpy_cosine	Uses local embedding cosine retrieval
invalid_ids=[]	LLM reranker did not output out-of-candidate IDs
llm_reason_top_n=3	Only Top-3 items use LLM-generated reasons
template_reason_items=7	Remaining items use template reasons
```

# Interpretation

v0.3.0 upgrades the system from LLM-enhanced recommendation post-processing to LLM-guided candidate recall. This provides a safer alternative to direct LLM product generation: the LLM contributes semantic intent, while final products are still grounded in the local catalog and downstream business filters.

## 1. Project Versions

| Version | Description |
|---|---|
| `v0.1.0` | Core Multi-Agent GenRec system with Semantic ID, GRU GenRec, FastAPI serving, benchmark and fallback validation |
| `v0.2.0` | DeepSeek LLM-enhanced GenRec-Agent with LLMRerankAgent, Batch LLMMarketingAgent, OpenAI-compatible LLMClient and LLM fallback tests |

---

## 2. Dataset Summary

The project uses the Amazon Reviews 2023 All Beauty subset.

| Item | Value |
|---|---:|
| Users | 845 |
| Items | 1,442 |
| Interactions | 6,628 |
| Train samples | 3,248 |
| Valid samples | 845 |
| Test samples | 845 |

Processing configuration:

```text
min_rating = 4
min_user_interactions = 4
min_item_interactions = 3
semantic_id_clusters = 128
3. Offline Recommendation Results
Method	Test Recall@10	Test Recall@20	Test Recall@50	Test NDCG@20	Test NDCG@50
Popularity	0.0012	0.0047	0.0284	0.0015	0.0061
Semantic-ID Neighbor	0.0237	0.0355	0.0746	0.0124	0.0202
GRU GenRec	0.0308	0.0438	0.0698	0.0166	0.0217
Interpretation

Semantic-ID Neighbor significantly outperforms the popularity baseline, showing that text-derived Semantic IDs capture useful product-neighborhood signals.

GRU GenRec further improves Recall@10, Recall@20 and NDCG@20 compared with Semantic-ID Neighbor, indicating that the model learns sequential preference patterns beyond simple semantic-neighbor expansion.

4. Core API Benchmark

Benchmark configuration:

num_requests = 100
concurrency = 10
mode = genrec_gru
top_k = 10
rerank_mode = none
marketing_mode = template
Metric	Value
Success Rate	100%
Fallback Rate	0%
Avg Returned Items	9.98 / 10
Throughput	595.83 req/s
Client Avg Latency	14.06 ms
Client P95 Latency	18.46 ms
Client P99 Latency	23.62 ms
Server Avg Latency	10.88 ms
Server P95 Latency	12.62 ms
Server P99 Latency	13.24 ms

This benchmark measures the non-LLM local recommendation path. Real LLM latency depends on provider, network, model response time and TopK size.

5. Core Fallback Validation
Case	Expected Behavior	Result
Normal model path	Use GRU GenRec	Passed
Missing model path	Fallback to Semantic-ID Neighbor	Passed
Unknown user_id	Return structured 404 trace	Passed

Example missing-model fallback:

requested_mode = genrec_gru
used_mode = semantic_neighbor_fallback
fallback_used = true
num_returned = 10
6. DeepSeek LLM Agent Results
6.1 Validated Capabilities
Capability	Result
DeepSeek API connection	Passed
OpenAI-compatible LLMClient	Passed
JSON structured output parsing	Passed
LLMRerankAgent candidate-constrained reranking	Passed
Invalid product ID validation	Passed, invalid_ids=[]
Batch LLMMarketingAgent	Passed
LLMMarketing fallback	Passed
LLMRerank fallback	Passed
Full LangGraph LLM workflow	Passed
6.2 LLMRerankAgent Result

The LLMRerankAgent uses DeepSeek API to rerank candidate products produced by GRU GenRec.

Key design:

GenRec candidates
→ DeepSeek LLMRerankAgent
→ reranked_product_ids
→ validate product IDs against original candidates
→ final candidate order

Observed trace:

rerank_mode = llm
rerank_applied = true
llm_provider = deepseek
llm_model = deepseek-chat
invalid_ids = []
input_items = 10
output_items = 10

Observed latency:

Component	Latency
LLMRerankAgent	~2.7 - 2.8s

The LLM did not create any out-of-candidate product IDs in the validated run.

6.3 Batch LLMMarketingAgent Result

The MarketingAgent was optimized from per-item LLM generation to batch LLM generation.

Previous design:

TopK items → K DeepSeek API calls

Optimized design:

TopK items → 1 DeepSeek API call → product_id-reason mapping

Observed trace for Top5:

reason_type = llm
llm_reason_mode = batch
llm_batch_call_count = 1
llm_success_count = 5
llm_fallback_count = 0
llm_provider = deepseek
llm_model = deepseek-chat

Observed trace for Top10:

reason_type = llm
llm_reason_mode = batch
llm_batch_call_count = 1
llm_success_count = 10
llm_fallback_count = 0
llm_provider = deepseek
llm_model = deepseek-chat

Observed latency:

Component	Latency
Batch LLMMarketingAgent, Top5	~2.5s
Batch LLMMarketingAgent, Top10	~4.7s

Compared with the earlier per-item DeepSeek version, Top5 marketing latency decreased from about 5.4s to about 2.5s while reducing the number of LLM marketing API calls from 5 to 1.

7. LLM Fallback Validation
7.1 LLMMarketing Fallback

When the LLM provider endpoint is invalid, MarketingAgent falls back to template reasons.

Observed result:

case = llm_marketing_fallback
passed = true
fallback_used = true
llm_success_count = 0
llm_fallback_count = 5
final_items = 5
7.2 LLMRerank Fallback

When the LLM provider endpoint is invalid, LLMRerankAgent falls back to the original GenRec ranking.

Observed result:

case = llm_rerank_fallback
passed = true
fallback_used = true
rerank_applied = false
fallback_reason = llm_call_failed
final_items = 10

This ensures that LLM failures do not break the recommendation service.

8. Full LLM-Enhanced Pipeline

Full pipeline configuration:

mode = genrec_gru
rerank_mode = llm
marketing_mode = llm
llm_provider = deepseek
llm_model = deepseek-chat

Workflow:

UserProfileAgent
→ GenerativeRecAgent
→ FilterAgent
→ LLMRerankAgent
→ Batch LLMMarketingAgent
→ Recommendation Response

Observed result:

Field	Value
Final items	10
LLMRerankAgent	Passed
Batch LLMMarketingAgent	Passed
Global fallback_used	False
Invalid product IDs	[]
9. Engineering Takeaways
9.1 LLM is not used for full-catalog generation

The LLM does not directly generate product IDs from the full item pool. GRU GenRec first produces candidate products, and the LLM only reranks within those candidates.

This avoids:

hallucinated product IDs
nonexistent items
uncontrolled recommendation outputs
9.2 Batch generation reduces LLM latency and cost

Per-item LLM reason generation is expensive and slow. Batch LLMMarketingAgent reduces API calls from N calls per request to 1 call per request.

9.3 Fallback is part of the design

If the LLM fails, the system falls back to:

Failed Component	Fallback
LLMRerankAgent	Original GenRec ranking
LLMMarketingAgent	Template recommendation reasons
9.4 Trace is first-class

Each Agent records:

agent name
success flag
latency
fallback status
provider/model metadata
LLM error sample
invalid product IDs

This makes the pipeline easier to debug, benchmark and explain in interviews.

10. Limitations
The dataset is a small processed subset of Amazon Reviews 2023 All Beauty.
Inventory is simulated because Amazon Reviews does not provide real stock.
Current Semantic ID uses one-level KMeans clustering instead of full TIGER / RQ-VAE style Semantic IDs.
LLM latency depends on provider, network and response length.
Batch LLM reason quality may vary and should be further controlled with stronger style prompts or output-quality checks.
No online A/B test or real user feedback loop is included.
11. Future Work
Add multi-level Semantic ID generation, inspired by TIGER-style generative retrieval.
Replace GRU with Transformer/T5-style semantic ID generation.
Add local Qwen2.5 provider through Ollama or vLLM.
Add LLM output quality evaluation for recommendation reasons.
Add category / keyword guardrails for noisy product items.
Add LLM rerank A/B evaluation against non-LLM ranking.
Evaluate on larger Amazon categories or multi-category datasets.
Add online feedback simulation with Thompson Sampling.
12. Resume-Ready Summary

GenRec-Agent is a FastAPI + LangGraph based multi-agent generative recommendation system. It combines Semantic-ID based GRU GenRec with DeepSeek-powered LLM agents for candidate-constrained reranking and batch recommendation reason generation. The system supports API serving, trace logging, latency monitoring, fallback validation, mock / DeepSeek provider switching, and versioned GitHub releases.