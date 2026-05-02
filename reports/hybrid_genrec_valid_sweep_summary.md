# Hybrid GenRec Reranker Evaluation

This report evaluates a hybrid reranker that combines GRU cluster probability, Semantic-ID neighbor score, and normalized popularity.

Hybrid score:

```text
score(item) = alpha * gru_cluster_prob + beta * semantic_neighbor_score + gamma * popularity_score
```

Important: this is an offline diagnostic experiment. It does not replace the existing full service pipeline.

## Configuration

- Split: `valid`
- Exclude history: `True`
- Top clusters: `10`
- Top-K eval candidate pool control: `50`
- Model loaded: `True`

## Sweep Results

| Rank | Split | alpha | beta | gamma | Hybrid Recall@20 | Hybrid NDCG@20 | Hybrid MRR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | valid | 1.0 | 0.5 | 0.2 | 0.2521 | 0.0834 | 0.0424 |
| 2 | valid | 0.5 | 1.0 | 0.1 | 0.2509 | 0.0830 | 0.0423 |
| 3 | valid | 0.5 | 2.0 | 0.1 | 0.2509 | 0.0830 | 0.0423 |
| 4 | valid | 1.0 | 2.0 | 0.2 | 0.2509 | 0.0830 | 0.0423 |
| 5 | valid | 0.5 | 0.5 | 0.1 | 0.2509 | 0.0830 | 0.0423 |
| 6 | valid | 1.0 | 1.0 | 0.2 | 0.2509 | 0.0830 | 0.0423 |
| 7 | valid | 2.0 | 1.0 | 0.5 | 0.2497 | 0.0828 | 0.0423 |
| 8 | valid | 1.0 | 0.5 | 0.1 | 0.2497 | 0.0823 | 0.0417 |
| 9 | valid | 2.0 | 1.0 | 0.2 | 0.2497 | 0.0823 | 0.0417 |
| 10 | valid | 2.0 | 1.0 | 0.1 | 0.2497 | 0.0821 | 0.0414 |
| 11 | valid | 2.0 | 2.0 | 0.5 | 0.2485 | 0.0824 | 0.0422 |
| 12 | valid | 1.0 | 1.0 | 0.1 | 0.2485 | 0.0820 | 0.0416 |
| 13 | valid | 1.0 | 2.0 | 0.1 | 0.2485 | 0.0820 | 0.0416 |
| 14 | valid | 2.0 | 2.0 | 0.2 | 0.2485 | 0.0820 | 0.0416 |
| 15 | valid | 2.0 | 2.0 | 0.1 | 0.2485 | 0.0817 | 0.0413 |
| 16 | valid | 2.0 | 0.5 | 0.1 | 0.2485 | 0.0814 | 0.0409 |
| 17 | valid | 2.0 | 0.5 | 0.0 | 0.2485 | 0.0810 | 0.0407 |
| 18 | valid | 2.0 | 0.5 | 0.2 | 0.2473 | 0.0818 | 0.0417 |
| 19 | valid | 2.0 | 0.5 | 0.5 | 0.2473 | 0.0806 | 0.0403 |
| 20 | valid | 1.0 | 2.0 | 0.5 | 0.2462 | 0.0820 | 0.0423 |

## Best Setting

- alpha: `1.0`
- beta: `0.5`
- gamma: `0.2`

### Metrics

| Method | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0047 | 0.0028 | 0.0059 | 0.0032 | 0.0154 | 0.0056 | 0.0533 | 0.0128 | 0.0070 |
| semantic_neighbor | 0.0592 | 0.0306 | 0.1325 | 0.0539 | 0.2450 | 0.0821 | 0.2982 | 0.0929 | 0.0429 |
| hybrid_genrec | 0.0533 | 0.0277 | 0.1373 | 0.0546 | 0.2521 | 0.0834 | 0.3030 | 0.0937 | 0.0424 |

## Interpretation

- If Hybrid GenRec improves over Semantic-ID Neighbor, the GRU sequence signal complements semantic-neighbor retrieval.
- If Hybrid GenRec does not improve, the current sparse Beauty subset is likely dominated by local semantic-neighbor signals.
- This experiment should be reported as a diagnostic reranking study, not as a production recommender benchmark.
