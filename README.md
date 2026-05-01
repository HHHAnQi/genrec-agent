# GenRec-Agent: 基于 LangGraph 多 Agent 编排的生成式电商推荐系统

GenRec-Agent 是一个面向电商推荐场景的多 Agent 生成式推荐系统。项目基于 **FastAPI + LangGraph** 构建状态图工作流，集成用户画像、生成式推荐、业务过滤、推荐理由生成与服务化接口，并在 Amazon Reviews 2023 Beauty 子集上完成推荐效果评估、接口压测与 fallback 验证。

本项目的目标不是复现完整工业级推荐系统，而是构建一个**可本地运行、可评估、可观测、可降级**的多 Agent 推荐系统原型。

---

## 1. Project Highlights

- **多 Agent 编排**：基于 LangGraph 构建 `UserProfileAgent → GenerativeRecAgent → FilterAgent → MarketingAgent` 状态图工作流。
- **生成式推荐核心**：将商品文本信息编码为 Semantic ID，并训练轻量 GRU GenRec 模型预测用户下一步偏好的语义商品簇。
- **真实电商数据**：基于 Amazon Reviews 2023 All Beauty 子集构建用户行为序列和商品文本表示。
- **完整评估闭环**：对比 Popularity、Semantic-ID Neighbor、GRU GenRec 三类推荐策略。
- **服务化接口**：通过 FastAPI `/recommend` 接口返回商品、库存、品牌、推荐理由和 Agent trace。
- **可观测与降级**：记录 trace、latency、fallback 等运行元数据；验证模型缺失时自动降级到 Semantic-ID neighbor 召回。

---

## 2. System Architecture

```text
FastAPI /recommend
        ↓
LangGraph Workflow
        ↓
UserProfileAgent
        ↓
GenerativeRecAgent
        ↓
FilterAgent
        ↓
MarketingAgent
        ↓
Recommendation Response
````

### Agent Responsibilities

| Agent                | Responsibility                                      |
| -------------------- | --------------------------------------------------- |
| `UserProfileAgent`   | 根据 `user_id` 读取用户历史行为，构建结构化用户画像                     |
| `GenerativeRecAgent` | 调用 GRU GenRec / Semantic-ID / Popularity 推理模块生成候选商品 |
| `FilterAgent`        | 根据库存、历史交互、重复商品等规则过滤候选集                              |
| `MarketingAgent`     | 基于模板生成推荐理由，后续可扩展为 LLM 文案生成                          |
| `LangGraph Workflow` | 管理状态流转、Agent 调用、trace 与 fallback 状态                 |

---

## 3. Dataset

项目使用 **Amazon Reviews 2023 - All Beauty** 子集。

### 数据处理策略

* 正反馈定义：`rating >= 4`
* 用户过滤：每个用户至少 4 条正反馈行为
* 商品过滤：每个商品初始至少 3 条交互
* 行为序列按时间排序
* 使用 leave-one-out 构造 train / valid / test

### Processed Dataset Statistics

| Item          | Value |
| ------------- | ----: |
| Users         |   845 |
| Items         | 1,442 |
| Interactions  | 6,628 |
| Train samples | 3,248 |
| Valid samples |   845 |
| Test samples  |   845 |

---

## 4. Semantic ID Construction

每个商品使用以下字段构造商品文本：

```text
title + category + brand + description
```

然后使用 SentenceTransformer 编码，并通过 KMeans 聚类生成 Semantic ID：

```text
item_text → embedding → KMeans cluster → semantic_id
```

当前版本采用轻量化 Semantic ID：

```text
semantic_id = [cluster_id]
n_clusters = 128
```

这是一种工程简化版 Tiger-style Semantic ID 方案，适合本地可复现和快速实验。

---

## 5. Recommendation Methods

项目实现了三类推荐策略：

### 5.1 Popularity Baseline

基于训练集 target 商品出现频率进行热门推荐。

### 5.2 Semantic-ID Neighbor Baseline

基于用户历史商品的 Semantic ID cluster，召回同 cluster 下的商品，并结合训练集流行度排序。

### 5.3 GRU GenRec

输入用户历史商品对应的 Semantic ID 序列，训练 GRU 预测下一个 Semantic ID cluster：

```text
history semantic clusters → GRU → next semantic cluster → candidate items
```

模型输出 cluster 后，再从 `sid_to_items.json` 中映射回候选商品，并结合商品流行度进行排序。

---

## 6. Recommendation Results

| Method               | Test Recall@10 | Test Recall@20 | Test Recall@50 | Test NDCG@20 | Test NDCG@50 |
| -------------------- | -------------: | -------------: | -------------: | -----------: | -----------: |
| Popularity           |         0.0012 |         0.0047 |         0.0284 |       0.0015 |       0.0061 |
| Semantic-ID Neighbor |         0.0237 |         0.0355 |         0.0746 |       0.0124 |       0.0202 |
| GRU GenRec           |         0.0308 |         0.0438 |         0.0698 |       0.0166 |       0.0217 |

### Interpretation

Semantic-ID Neighbor 显著优于 Popularity，说明基于商品文本构建的 Semantic ID 能捕捉有效的商品语义邻域。GRU GenRec 进一步建模用户行为序列，在 Recall@20 和 NDCG@20 上超过无训练的 Semantic-ID Neighbor baseline，说明模型不仅在做相似商品扩散，还学习到了用户序列偏好。

---

## 7. API Benchmark

对 `/recommend` 接口进行本地压测：

```text
num_requests = 100
concurrency = 10
mode = genrec_gru
top_k = 10
```

| Metric             |        Value |
| ------------------ | -----------: |
| Success Rate       |         100% |
| Fallback Rate      |           0% |
| Avg Returned Items |    9.98 / 10 |
| Throughput         | 595.83 req/s |
| Client Avg Latency |     14.06 ms |
| Client P95 Latency |     18.46 ms |
| Client P99 Latency |     23.62 ms |
| Server Avg Latency |     10.88 ms |
| Server P95 Latency |     12.62 ms |
| Server P99 Latency |     13.24 ms |

---

## 8. Fallback Validation

项目验证了三种关键情况：

| Case               | Expected Behavior          | Result |
| ------------------ | -------------------------- | ------ |
| Normal model path  | 使用 GRU GenRec              | Passed |
| Missing model path | 自动降级到 Semantic-ID Neighbor | Passed |
| Unknown user_id    | 返回结构化 404 trace            | Passed |

示例：

```text
Missing model:
requested_mode = genrec_gru
used_mode = semantic_neighbor_fallback
fallback_used = true
num_returned = 10
```

---

## 9. Project Structure

```text
genrec-agent/
├── agents/
│   ├── base.py
│   ├── user_profile.py
│   ├── generative_rec.py
│   ├── filter.py
│   └── marketing.py
├── graph/
│   └── workflow.py
├── recommender/
│   ├── __init__.py
│   └── inference.py
├── schemas/
│   └── models.py
├── scripts/
│   ├── prepare_amazon_beauty.py
│   ├── build_item_texts.py
│   ├── build_semantic_id.py
│   ├── build_splits.py
│   ├── evaluate_popularity.py
│   ├── evaluate_semantic_baseline.py
│   ├── train_genrec_gru.py
│   ├── test_langgraph_workflow.py
│   ├── benchmark_api.py
│   └── test_fallback.py
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── models/
│   └── genrec_gru.pt
├── reports/
│   ├── api_benchmark_results.json
│   └── fallback_test_results.json
├── main.py
└── README.md
```

---

## 10. Installation

```bash
pip install pandas numpy scikit-learn sentence-transformers torch
pip install fastapi uvicorn langgraph httpx
```

---

## 11. Data Preparation

### Step 1: Prepare Amazon Beauty Data

```bash
python scripts/prepare_amazon_beauty.py \
  --min_rating 4 \
  --min_user_interactions 4 \
  --min_item_interactions 3
```

### Step 2: Build Item Texts

```bash
python scripts/build_item_texts.py
```

### Step 3: Build Semantic IDs

```bash
python scripts/build_semantic_id.py \
  --n_clusters 128
```

### Step 4: Build Train / Valid / Test Splits

```bash
python scripts/build_splits.py
```

---

## 12. Offline Evaluation

### Popularity Baseline

```bash
python scripts/evaluate_popularity.py --exclude_history
```

### Semantic-ID Neighbor Baseline

```bash
python scripts/evaluate_semantic_baseline.py --exclude_history
```

### Train GRU GenRec

```bash
python scripts/train_genrec_gru.py \
  --exclude_history \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --top_clusters 5
```

---

## 13. Run FastAPI Service

```bash
uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Recommendation request:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ",
    "top_k": 10,
    "mode": "genrec_gru"
  }'
```

---

## 14. Example API Response

```json
{
  "request_id": "ca43ecfb-4c00-424c-b9de-a1a854062e4d",
  "user_id": "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ",
  "top_k": 10,
  "mode": "genrec_gru",
  "fallback_used": false,
  "latency_ms": 3.5,
  "items": [
    {
      "product_id": "B082VKPJV5",
      "score": 1.0,
      "source": "genrec_gru",
      "category": "All Beauty",
      "brand": "WORKMAN'S FRIEND",
      "title": "Workman's Friend Ultimate Hand Care Bundle",
      "price": -1.0,
      "stock": 90,
      "reason": "Recommended based on your recent interest in All Beauty; Workman's Friend Ultimate Hand Care Bundle may fit your personal-care routine."
    }
  ],
  "trace": [
    {
      "agent": "UserProfileAgent",
      "success": true,
      "latency_ms": 0.13,
      "fallback_used": false
    },
    {
      "agent": "GenerativeRecAgent",
      "success": true,
      "latency_ms": 0.78,
      "fallback_used": false
    },
    {
      "agent": "FilterAgent",
      "success": true,
      "latency_ms": 0.06,
      "fallback_used": false
    },
    {
      "agent": "MarketingAgent",
      "success": true,
      "latency_ms": 0.08,
      "fallback_used": false
    }
  ]
}
```

---

## 15. API Benchmark

Start the service first:

```bash
uvicorn main:app --reload
```

Run benchmark:

```bash
python scripts/benchmark_api.py \
  --num_requests 100 \
  --concurrency 10 \
  --top_k 10 \
  --mode genrec_gru
```

---

## 16. Fallback Test

Start the service first:

```bash
uvicorn main:app --reload
```

Run fallback validation:

```bash
python scripts/test_fallback.py
```

Expected behavior:

```text
normal_agent: passed
missing_model_fallback: passed
unknown_user_api: passed
```

---

## 17. Design Notes

### Why LangGraph?

This project is not a free-form multi-agent chat system. It is a stateful recommendation workflow with deterministic stages:

```text
profile construction → generative recommendation → business filtering → reason generation
```

LangGraph is used to explicitly model the state transitions and preserve traceability across Agents.

### Why Semantic ID?

Direct item ID prediction is sparse for small-scale recommendation data. Semantic ID reduces the prediction space from item-level classes to semantic clusters, making the GRU GenRec model easier to train and more interpretable.

### Why Template MarketingAgent?

The current MarketingAgent uses template-based reason generation to ensure reproducibility and avoid external LLM dependency. It can be replaced by an LLM-based copywriting agent in future versions.

---

## 18. Limitations

* Current Semantic ID uses KMeans rather than full RQ-VAE or Tiger’s original Semantic ID pipeline.
* Dataset is a small processed subset of Amazon Reviews 2023 All Beauty, not a full-scale industrial dataset.
* MarketingAgent currently uses templates instead of LLM-generated copy.
* Inventory is simulated because Amazon Reviews does not provide real stock information.
* Current GenRec model predicts one-level semantic clusters; multi-level Semantic ID generation can be added later.

---

## 19. Future Work

* Add multi-level Semantic ID generation.
* Replace GRU with Transformer/T5-style semantic ID generation.
* Add LLM-based MarketingAgent with safety and style control.
* Add A2A adapter for external Agent interoperability.
* Evaluate on larger Amazon categories or multi-category data.
* Add online feedback simulation with Thompson Sampling.

---

## 20. Resume Summary

GenRec-Agent is a FastAPI + LangGraph based multi-agent recommendation system. It integrates Semantic-ID based generative recommendation into a structured Agent workflow and provides full evaluation, API serving, trace logging, latency benchmark, and fallback validation.

```
