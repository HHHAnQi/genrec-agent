# GenRec-Agent: DeepSeek LLM-Enhanced Multi-Agent Generative Recommendation System

GenRec-Agent 是一个面向电商推荐场景的多 Agent 生成式推荐系统。项目基于 **FastAPI + LangGraph** 构建状态图工作流，集成用户画像、生成式推荐、业务过滤、LLM 候选重排、推荐理由生成与服务化接口，并在 Amazon Reviews 2023 Beauty 子集上完成推荐效果评估、接口压测、fallback 验证与真实 DeepSeek API 接入。

本项目的目标不是复现完整工业级推荐系统，而是构建一个**可本地运行、可评估、可观测、可降级、支持真实 LLM Agent 增强**的推荐系统原型。

---

## Version History

| Version | Description |
|---|---|
| `v0.1.0` | Core Multi-Agent GenRec system: Semantic ID, GRU GenRec, FastAPI, LangGraph workflow, benchmark and fallback validation |
| `v0.2.0` | DeepSeek LLM-enhanced GenRec-Agent: LLMRerankAgent, Batch LLMMarketingAgent, OpenAI-compatible LLMClient, mock / DeepSeek provider, LLM fallback and trace |

---

## 1. Project Highlights

- **多 Agent 编排**：基于 LangGraph 构建 `UserProfileAgent → GenerativeRecAgent → FilterAgent → LLMRerankAgent → MarketingAgent` 状态图工作流。
- **生成式推荐核心**：将商品文本信息编码为 Semantic ID，并训练轻量 GRU GenRec 模型预测用户下一步偏好的语义商品簇。
- **真实 LLM 接入**：通过 OpenAI-compatible `LLMClient` 接入 DeepSeek API，并保留 `mock` 本地模式。
- **候选集内 LLM 重排**：`LLMRerankAgent` 只在 GenRec 候选商品内进行受约束重排，不允许 LLM 生成不存在的商品 ID。
- **Batch LLM 推荐理由生成**：`Batch LLMMarketingAgent` 一次性为 TopK 商品生成推荐理由，将 LLM 文案生成从 N 次 API 调用优化为 1 次 API 调用。
- **真实电商数据**：基于 Amazon Reviews 2023 All Beauty 子集构建用户行为序列和商品文本表示。
- **完整评估闭环**：对比 Popularity、Semantic-ID Neighbor、GRU GenRec 三类推荐策略。
- **服务化接口**：通过 FastAPI `/recommend` 接口返回商品、库存、品牌、推荐理由和 Agent trace。
- **可观测与降级**：记录 trace、latency、fallback、provider、model、invalid_ids 等运行元数据；验证模型缺失和 LLM API 异常时的 fallback 行为。
- **安全配置**：真实 API key 仅放在本地 `.env`，仓库只提交 `.env.example`。

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
LLMRerankAgent
        ↓
MarketingAgent
        ↓
Recommendation Response
````

### Agent Responsibilities

| Agent                | Responsibility                                       |
| -------------------- | ---------------------------------------------------- |
| `UserProfileAgent`   | 根据 `user_id` 读取用户历史行为，构建结构化用户画像                      |
| `GenerativeRecAgent` | 调用 GRU GenRec / Semantic-ID / Popularity 推理模块生成候选商品  |
| `FilterAgent`        | 根据库存、历史交互、重复商品等规则过滤候选集                               |
| `LLMRerankAgent`     | 使用 LLM 在候选集内进行受约束重排，并校验返回 product_id 是否来自候选集         |
| `MarketingAgent`     | 支持 template / LLM 两种推荐理由生成模式；LLM 模式下使用 batch JSON 输出 |
| `LangGraph Workflow` | 管理状态流转、Agent 调用、trace 与 fallback 状态                  |

---

## 3. LLM Agent Design

### 3.1 OpenAI-Compatible LLMClient

项目实现了统一的 `LLMClient`，支持以下 provider：

| Provider   | Purpose                                            |
| ---------- | -------------------------------------------------- |
| `mock`     | 本地可复现的假 LLM，不需要 API key，适合 demo / CI / fallback 测试 |
| `deepseek` | DeepSeek API，真实 LLM 调用主线                           |
| `openai`   | OpenAI-compatible API                              |
| `ollama`   | 本地 OpenAI-compatible server，例如本地 Qwen2.5           |

所有 provider 共用同一套 Agent 逻辑：

```text
LLMClient
→ JSON structured output
→ timeout control
→ error capture
→ fallback handled by Agent
→ trace metadata
```

### 3.2 LLMRerankAgent

`LLMRerankAgent` 不直接从全商品库生成推荐结果，而是在 GenRec 产生的候选集内进行受约束重排：

```text
GenRec candidates
→ LLMRerankAgent
→ reranked_product_ids
→ product_id validation
→ final candidate order
```

核心约束：

```text
LLM can only return product IDs from the given candidate list.
Invalid product IDs are filtered and recorded in trace.
If LLM call fails, fallback to original GenRec ranking.
```

### 3.3 Batch LLMMarketingAgent

`MarketingAgent` 支持两种模式：

| Mode       | Behavior                                          |
| ---------- | ------------------------------------------------- |
| `template` | 使用模板生成推荐理由，速度快、可复现                                |
| `llm`      | 调用 LLM 生成推荐理由，支持 batch JSON 输出和 template fallback |

Batch LLM reason generation：

```text
TopK items
→ 1 DeepSeek API call
→ {"reasons": [{"product_id": "...", "reason": "..."}]}
→ validate product_id-reason mapping
→ fallback missing / invalid reasons to template
```

相比逐商品调用，batch 模式将 LLM marketing 调用从：

```text
N calls per request
```

优化为：

```text
1 call per request
```

---

## 4. Dataset

项目使用 **Amazon Reviews 2023 - All Beauty** 子集。

### Data Processing Strategy

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

## 5. Semantic ID Construction

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

这是一种工程简化版 Tiger-style Semantic ID 方案，适合本地可复现和快速实验。完整 TIGER / RQ-VAE / 多级 Semantic ID 可作为后续 `v0.3.0` 扩展方向。

---

## 6. Recommendation Methods

项目实现了三类推荐策略：

### 6.1 Popularity Baseline

基于训练集 target 商品出现频率进行热门推荐。

### 6.2 Semantic-ID Neighbor Baseline

基于用户历史商品的 Semantic ID cluster，召回同 cluster 下的商品，并结合训练集流行度排序。

### 6.3 GRU GenRec

输入用户历史商品对应的 Semantic ID 序列，训练 GRU 预测下一个 Semantic ID cluster：

```text
history semantic clusters → GRU → next semantic cluster → candidate items
```

模型输出 cluster 后，再从 `sid_to_items.json` 中映射回候选商品，并结合商品流行度进行排序。

---

## 7. Recommendation Results

| Method               | Test Recall@10 | Test Recall@20 | Test Recall@50 | Test NDCG@20 | Test NDCG@50 |
| -------------------- | -------------: | -------------: | -------------: | -----------: | -----------: |
| Popularity           |         0.0012 |         0.0047 |         0.0284 |       0.0015 |       0.0061 |
| Semantic-ID Neighbor |         0.0237 |         0.0355 |         0.0746 |       0.0124 |       0.0202 |
| GRU GenRec           |         0.0308 |         0.0438 |         0.0698 |       0.0166 |       0.0217 |

### Interpretation

Semantic-ID Neighbor 显著优于 Popularity，说明基于商品文本构建的 Semantic ID 能捕捉有效的商品语义邻域。GRU GenRec 进一步建模用户行为序列，在 Recall@20 和 NDCG@20 上超过无训练的 Semantic-ID Neighbor baseline，说明模型不仅在做相似商品扩散，还学习到了用户序列偏好。

---

## 8. DeepSeek LLM Agent Results

### Validated Capabilities

| Capability                                     | Result                   |
| ---------------------------------------------- | ------------------------ |
| DeepSeek API connection                        | Passed                   |
| LLMClient JSON parsing                         | Passed                   |
| LLMRerankAgent candidate-constrained reranking | Passed                   |
| Invalid product ID check                       | Passed, `invalid_ids=[]` |
| Batch LLMMarketingAgent                        | Passed                   |
| LLMMarketing fallback                          | Passed                   |
| LLMRerank fallback                             | Passed                   |
| Full LangGraph workflow                        | Passed                   |

### Observed Latency

| Component                      | Observed Latency |
| ------------------------------ | ---------------: |
| LLMRerankAgent                 |      ~2.7 - 2.8s |
| Batch LLMMarketingAgent, Top5  |            ~2.5s |
| Batch LLMMarketingAgent, Top10 |            ~4.7s |

### Engineering Notes

* DeepSeek API is used through the OpenAI-compatible `/chat/completions` interface.
* LLMRerankAgent does not create new product IDs.
* Batch LLMMarketingAgent reduces marketing generation from N LLM calls to 1 LLM call.
* LLM failures are handled with fallback to original ranking or template reason generation.
* Agent trace records provider, model, latency, fallback status, and invalid product IDs.

---

## 9. API Benchmark

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

> Note: This benchmark is for the non-LLM local path. Real LLM latency depends on provider, network, model and TopK size.

---

## 10. Fallback Validation

项目验证了多种关键 fallback 情况：

| Case                     | Expected Behavior          | Result |
| ------------------------ | -------------------------- | ------ |
| Normal model path        | 使用 GRU GenRec              | Passed |
| Missing model path       | 自动降级到 Semantic-ID Neighbor | Passed |
| Unknown user_id          | 返回结构化 404 trace            | Passed |
| LLMMarketing API failure | fallback 到 template reason | Passed |
| LLMRerank API failure    | fallback 到原始 GenRec 排序     | Passed |

Example:

```text
Missing model:
requested_mode = genrec_gru
used_mode = semantic_neighbor_fallback
fallback_used = true
num_returned = 10
```

LLM failure example:

```text
LLMRerankAgent:
rerank_mode = llm
rerank_applied = false
fallback_reason = llm_call_failed
fallback_used = true
```

---

## 11. Project Structure

```text
genrec-agent/
├── agents/
│   ├── base.py
│   ├── user_profile.py
│   ├── generative_rec.py
│   ├── filter.py
│   ├── llm_rerank.py
│   └── marketing.py
├── graph/
│   └── workflow.py
├── recommender/
│   ├── __init__.py
│   └── inference.py
├── schemas/
│   └── models.py
├── services/
│   ├── __init__.py
│   └── llm_client.py
├── scripts/
│   ├── prepare_amazon_beauty.py
│   ├── build_item_texts.py
│   ├── build_semantic_id.py
│   ├── build_splits.py
│   ├── evaluate_popularity.py
│   ├── evaluate_semantic_baseline.py
│   ├── train_genrec_gru.py
│   ├── test_langgraph_workflow.py
│   ├── test_llm_client.py
│   ├── test_llm_marketing.py
│   ├── test_llm_marketing_fallback.py
│   ├── test_llm_rerank.py
│   ├── test_llm_rerank_fallback.py
│   ├── test_llm_full_pipeline.py
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
│   ├── fallback_test_results.json
│   ├── llm_deepseek_results_summary.md
│   ├── llm_marketing_fallback_results.json
│   └── llm_rerank_fallback_results.json
├── .env.example
├── main.py
├── requirements.txt
└── README.md
```

---

## 12. Installation

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install pandas numpy scikit-learn sentence-transformers torch
pip install fastapi uvicorn langgraph httpx python-dotenv
```

---

## 13. LLM Configuration

### 13.1 Mock Mode

`mock` mode requires no API key and is suitable for local reproducibility.

```env
LLM_PROVIDER=mock
LLM_TIMEOUT_SECONDS=20
LLM_USE_RESPONSE_FORMAT=true
```

### 13.2 DeepSeek API Mode

Create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_TIMEOUT_SECONDS=20
LLM_USE_RESPONSE_FORMAT=true
```

Security note:

```text
Do not commit .env.
Only .env.example should be committed.
```

---

## 14. Data Preparation

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

## 15. Offline Evaluation

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

## 16. Run FastAPI Service

```bash
uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 16.1 Default Recommendation Request

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ",
    "top_k": 10,
    "mode": "genrec_gru"
  }'
```

### 16.2 LLM-Enhanced Recommendation Request

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ",
    "top_k": 10,
    "mode": "genrec_gru",
    "rerank_mode": "llm",
    "marketing_mode": "llm"
  }'
```

---

## 17. Example API Response

```json
{
  "request_id": "dcac4f1b-5599-4fb4-aa5f-6e07ea38ee5a",
  "user_id": "AE23ZBUF2YVBQPH2NN6F5XSA3QYQ",
  "top_k": 10,
  "mode": "genrec_gru",
  "marketing_mode": "llm",
  "rerank_mode": "llm",
  "fallback_used": false,
  "latency_ms": 16.03,
  "items": [
    {
      "product_id": "B082VKPJV5",
      "score": 1.0,
      "source": "genrec_gru+llm_rerank",
      "category": "All Beauty",
      "brand": "WORKMAN'S FRIEND",
      "title": "Workman's Friend Ultimate Hand Care Bundle",
      "price": -1.0,
      "stock": 90,
      "reason": "Top-rated hand care bundle for dry, working hands."
    }
  ],
  "trace": [
    {
      "agent": "UserProfileAgent",
      "success": true,
      "latency_ms": 0.12,
      "fallback_used": false
    },
    {
      "agent": "GenerativeRecAgent",
      "success": true,
      "latency_ms": 0.60,
      "fallback_used": false
    },
    {
      "agent": "FilterAgent",
      "success": true,
      "latency_ms": 0.06,
      "fallback_used": false
    },
    {
      "agent": "LLMRerankAgent",
      "success": true,
      "latency_ms": 2817.55,
      "fallback_used": false,
      "metadata": {
        "rerank_mode": "llm",
        "rerank_applied": true,
        "llm_provider": "deepseek",
        "llm_model": "deepseek-chat",
        "invalid_ids": []
      }
    },
    {
      "agent": "MarketingAgent",
      "success": true,
      "latency_ms": 4689.83,
      "fallback_used": false,
      "metadata": {
        "reason_type": "llm",
        "llm_reason_mode": "batch",
        "llm_success_count": 10,
        "llm_fallback_count": 0,
        "llm_provider": "deepseek",
        "llm_model": "deepseek-chat",
        "llm_batch_call_count": 1
      }
    }
  ]
}
```

---

## 18. LLM Tests

### LLM Client Test

```bash
python scripts/test_llm_client.py
```

### LLM Marketing Test

```bash
python scripts/test_llm_marketing.py
```

### LLM Rerank Test

```bash
python scripts/test_llm_rerank.py
```

### Full LLM Pipeline Test

```bash
python scripts/test_llm_full_pipeline.py
```

Expected trace:

```text
LLMRerankAgent:
llm_provider = deepseek
llm_model = deepseek-chat
rerank_applied = true
invalid_ids = []

MarketingAgent:
reason_type = llm
llm_reason_mode = batch
llm_batch_call_count = 1
llm_success_count = 10
```

---

## 19. Fallback Tests

Start the service first:

```bash
uvicorn main:app --reload
```

Run fallback validation:

```bash
python scripts/test_fallback.py
```

LLM fallback tests:

```bash
python scripts/test_llm_marketing_fallback.py
python scripts/test_llm_rerank_fallback.py
```

Expected behavior:

```text
normal_agent: passed
missing_model_fallback: passed
unknown_user_api: passed
llm_marketing_fallback: passed
llm_rerank_fallback: passed
```

---

## 20. API Benchmark

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

## 21. Design Notes

### Why LangGraph?

This project is not a free-form multi-agent chat system. It is a stateful recommendation workflow with deterministic stages:

```text
profile construction → generative recommendation → filtering → LLM reranking → reason generation
```

LangGraph is used to explicitly model the state transitions and preserve traceability across Agents.

### Why Semantic ID?

Direct item ID prediction is sparse for small-scale recommendation data. Semantic ID reduces the prediction space from item-level classes to semantic clusters, making the GRU GenRec model easier to train and more interpretable.

### Why Candidate-Constrained LLM Reranking?

Letting LLM directly generate product IDs can cause hallucinated or nonexistent items. This project uses GenRec to generate candidates first, then lets the LLM only rerank within the candidate set. Returned product IDs are validated against the candidate list.

### Why Batch LLMMarketingAgent?

Per-item LLM reason generation causes high latency and cost. Batch generation reduces API calls from N calls per request to 1 call per request while still validating the product_id-reason mapping.

### Why Keep Mock Mode?

Mock mode is not a real LLM. It is kept for local reproducibility, CI-style testing, demo stability, and fallback validation when API keys are unavailable.

---

## 22. Limitations

* Current Semantic ID uses KMeans rather than full RQ-VAE or TIGER’s original Semantic ID pipeline.
* Dataset is a small processed subset of Amazon Reviews 2023 All Beauty, not a full-scale industrial dataset.
* Inventory is simulated because Amazon Reviews does not provide real stock information.
* Current GenRec model predicts one-level semantic clusters.
* DeepSeek API latency depends on network and model response time.
* Batch LLM reason quality may vary across items and should be controlled with stronger style prompts or post-filtering in production.

---

## 23. Future Work

* Add multi-level Semantic ID generation.
* Replace GRU with Transformer/T5-style semantic ID generation.
* Add local Qwen2.5 provider through Ollama or vLLM.
* Add LLM output quality evaluation for recommendation reasons.
* Add LLM rerank A/B evaluation against non-LLM ranking.
* Add category / keyword guardrail for noisy items.
* Evaluate on larger Amazon categories or multi-category data.
* Add online feedback simulation with Thompson Sampling.

---

## 24. Resume Summary

GenRec-Agent is a FastAPI + LangGraph based multi-agent generative recommendation system. It combines Semantic-ID based GRU GenRec with DeepSeek-powered LLM agents for candidate-constrained reranking and batch recommendation reason generation. The system supports API serving, trace logging, latency monitoring, fallback validation, and mock / DeepSeek provider switching.

````
