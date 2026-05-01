# GenRec-Agent Results Summary

## 1. Dataset

- Dataset: Amazon Reviews 2023 - All Beauty
- Positive feedback: rating >= 4
- User filtering: at least 4 positive interactions
- Item filtering: at least 3 positive interactions before final user filtering
- Users: 845
- Items: 1,442
- Interactions: 6,628
- Train samples: 3,248
- Valid samples: 845
- Test samples: 845

## 2. Recommendation Performance

| Method | Test Recall@10 | Test Recall@20 | Test Recall@50 | Test NDCG@20 | Test NDCG@50 |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0012 | 0.0047 | 0.0284 | 0.0015 | 0.0061 |
| Semantic-ID Neighbor | 0.0237 | 0.0355 | 0.0746 | 0.0124 | 0.0202 |
| GRU GenRec | 0.0308 | 0.0438 | 0.0698 | 0.0166 | 0.0217 |

## 3. Interpretation

Semantic-ID Neighbor significantly improves over the Popularity baseline, showing that text-derived Semantic IDs capture useful product similarity.

GRU GenRec further improves Recall@20 and NDCG@20 by modeling user behavior sequences over semantic item IDs. Semantic-ID Neighbor has stronger broad recall at Recall@50, while GRU GenRec performs better on earlier ranking metrics.

## 4. API Benchmark

Configuration:

- Endpoint: /recommend
- Mode: genrec_gru
- Requests: 100
- Concurrency: 10
- top_k: 10

| Metric | Value |
|---|---:|
| Success Rate | 100% |
| Fallback Rate | 0% |
| Avg Returned Items | 9.98 / 10 |
| Throughput | 595.83 req/s |
| Client Avg Latency | 14.06 ms |
| Client P95 Latency | 18.46 ms |
| Client P99 Latency | 23.62 ms |
| Server Avg Latency | 10.88 ms |
| Server P95 Latency | 12.62 ms |
| Server P99 Latency | 13.24 ms |

## 5. Fallback Validation

| Case | Expected Behavior | Result |
|---|---|---|
| Normal model path | Use GRU GenRec | Passed |
| Missing model path | Fallback to Semantic-ID Neighbor | Passed |
| Unknown user_id | Return structured 404 trace | Passed |

## 6. System Chain

FastAPI /recommend
→ LangGraph Workflow
→ UserProfileAgent
→ GenerativeRecAgent
→ FilterAgent
→ MarketingAgent
→ Recommendation Response