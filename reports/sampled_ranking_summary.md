# Sampled-Candidate Ranking Evaluation

This report adds a sampled-candidate ranking evaluation for GenRec-Agent.

Important: this evaluation is **not a replacement** for full-catalog ranking. 
It is a diagnostic benchmark that measures how well each method ranks the target item within a smaller candidate set.

## Configuration

- Split: `test`
- Num negatives per case: `99`
- Candidate set size: `100`
- Exclude history: `True`
- Seed: `42`
- Num evaluated: `845`
- Num skipped: `0`

## Results

| Method | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0473 | 0.0250 | 0.0911 | 0.0390 | 0.1669 | 0.0577 | 0.0433 |
| semantic_neighbor | 0.3408 | 0.2457 | 0.3929 | 0.2628 | 0.4746 | 0.2834 | 0.2372 |
| genrec_gru | 0.2556 | 0.1765 | 0.3101 | 0.1938 | 0.4118 | 0.2194 | 0.1768 |

## Interpretation

- Full-catalog ranking remains the stricter retrieval setting.
- Sampled-candidate ranking is used here to inspect candidate-level ranking behavior under a controlled candidate pool.
- These numbers should not be directly compared with full-catalog Recall/NDCG.
