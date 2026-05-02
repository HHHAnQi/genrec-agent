# Hybrid GenRec Reranker Evaluation

This report evaluates a hybrid reranker that combines GRU cluster probability, Semantic-ID neighbor score, and normalized popularity.

Hybrid score:

```text
score(item) = alpha * gru_cluster_prob + beta * semantic_neighbor_score + gamma * popularity_score
```

Important: this is an offline diagnostic experiment. It does not replace the existing full service pipeline.

## Configuration

- Split: `test`
- Exclude history: `True`
- Top clusters: `10`
- Top-K eval candidate pool control: `50`
- Model loaded: `True`

## Results

| Method | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0012 | 0.0006 | 0.0012 | 0.0006 | 0.0047 | 0.0014 | 0.0355 | 0.0074 | 0.0044 |
| semantic_neighbor | 0.0568 | 0.0270 | 0.1290 | 0.0501 | 0.2426 | 0.0784 | 0.3231 | 0.0947 | 0.0397 |
| hybrid_genrec | 0.0556 | 0.0284 | 0.1337 | 0.0532 | 0.2473 | 0.0816 | 0.3302 | 0.0983 | 0.0424 |

## Interpretation

- If Hybrid GenRec improves over Semantic-ID Neighbor, the GRU sequence signal complements semantic-neighbor retrieval.
- If Hybrid GenRec does not improve, the current sparse Beauty subset is likely dominated by local semantic-neighbor signals.
- This experiment should be reported as a diagnostic reranking study, not as a production recommender benchmark.
