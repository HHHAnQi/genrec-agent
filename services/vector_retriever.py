from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class VectorSearchResult:
    product_id: str
    score: float
    title: str
    category: str
    brand: str
    price: float
    stock: int
    source: str = "llm_query_recall"


class VectorRetriever:
    """
    Lightweight numpy-based vector retriever for GenRec-Agent.

    It maps an LLM-generated semantic query to real products by:
    1. encoding the query with SentenceTransformer;
    2. computing cosine similarity against precomputed item embeddings;
    3. returning product IDs and item metadata from the local product catalog.

    The LLM does not generate product IDs directly. It only generates a semantic query.
    Product IDs always come from the local item catalog.
    """

    def __init__(
        self,
        item_texts_path: str = "datasets/processed/item_texts.csv",
        items_path: str = "datasets/processed/items.csv",
        item_embeddings_path: str = "datasets/processed/item_embeddings.npy",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ):
        self.item_texts_path = Path(item_texts_path)
        self.items_path = Path(items_path)
        self.item_embeddings_path = Path(item_embeddings_path)
        self.embedding_model_name = embedding_model_name
        self.normalize_embeddings = normalize_embeddings

        self.item_texts: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None
        self.embeddings: np.ndarray | None = None
        self.product_ids: list[str] = []
        self.item_meta: dict[str, dict] = {}

        self.model = None
        self.loaded = False
        self.load_error: str | None = None

        self._load()

    def _load(self):
        try:
            self.item_texts = pd.read_csv(self.item_texts_path)
            self.items = pd.read_csv(self.items_path)
            embeddings = np.load(self.item_embeddings_path).astype("float32")
            embeddings = np.nan_to_num(
                embeddings,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype("float32")

            if "product_id" not in self.item_texts.columns:
                raise ValueError(f"`product_id` column missing in {self.item_texts_path}")

            if len(self.item_texts) != embeddings.shape[0]:
                raise ValueError(
                    "Row mismatch between item_texts and embeddings: "
                    f"item_texts={len(self.item_texts)}, embeddings={embeddings.shape[0]}"
                )

            self.product_ids = [str(x) for x in self.item_texts["product_id"].tolist()]
            self.embeddings = embeddings

            if self.normalize_embeddings:
                self.embeddings = self._l2_normalize(self.embeddings)

            self.item_meta = self._build_item_meta(self.items)

            # Lazy-ish model initialization; still done at load time so failures are visible early.
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.embedding_model_name)

            self.loaded = True
            self.load_error = None

        except Exception as e:
            self.loaded = False
            self.load_error = repr(e)
            self.item_texts = None
            self.items = None
            self.embeddings = None
            self.product_ids = []
            self.item_meta = {}
            self.model = None

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return x / norm

    @staticmethod
    def _safe_str(value, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return str(value)

    @staticmethod
    def _safe_float(value, default: float = -1.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, float) and np.isnan(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            if value is None:
                return default
            if isinstance(value, float) and np.isnan(value):
                return default
            return int(value)
        except Exception:
            return default

    def _build_item_meta(self, items_df: pd.DataFrame) -> dict[str, dict]:
        required = {"product_id", "title", "category", "brand", "price", "stock"}
        missing = required - set(items_df.columns)
        if missing:
            raise ValueError(f"Missing columns in items.csv: {sorted(missing)}")

        meta = {}
        for row in items_df.to_dict(orient="records"):
            pid = str(row["product_id"])
            meta[pid] = {
                "product_id": pid,
                "title": self._safe_str(row.get("title")),
                "category": self._safe_str(row.get("category"), default="All Beauty"),
                "brand": self._safe_str(row.get("brand")),
                "price": self._safe_float(row.get("price"), default=-1.0),
                "stock": self._safe_int(row.get("stock"), default=0),
            }
        return meta

    def encode_query(self, query: str) -> np.ndarray:
        if not self.loaded or self.model is None:
            raise RuntimeError(f"VectorRetriever is not loaded: {self.load_error}")

        query = str(query).strip()
        if not query:
            raise ValueError("Empty query cannot be encoded.")

        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        ).astype("float32")[0]

        vec = np.nan_to_num(
            vec,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype("float32")

        if not self.normalize_embeddings:
            vec = vec / max(float(np.linalg.norm(vec)), 1e-12)

        return vec

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        exclude_product_ids: Iterable[str] | None = None,
        min_score: float | None = None,
    ) -> tuple[list[VectorSearchResult], dict]:
        """
        Retrieve real catalog items for a semantic query.

        Returns:
            results: ranked VectorSearchResult list
            metadata: trace-friendly metadata
        """
        start = time.perf_counter()

        if not self.loaded or self.embeddings is None:
            return [], {
                "success": False,
                "error": self.load_error or "VectorRetriever is not loaded.",
                "query": query,
                "top_k": top_k,
                "retrieval_latency_ms": 0.0,
                "num_results": 0,
            }

        query = str(query).strip()
        if not query:
            return [], {
                "success": False,
                "error": "empty_query",
                "query": query,
                "top_k": top_k,
                "retrieval_latency_ms": 0.0,
                "num_results": 0,
            }

        top_k = max(1, int(top_k))
        exclude_set = set(str(x) for x in (exclude_product_ids or []))

        try:
            q = self.encode_query(query)
            q = np.nan_to_num(
                q,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype("float32")

            q_norm = float(np.linalg.norm(q))
            if q_norm <= 1e-12:
                return [], {
                    "success": False,
                    "error": "zero_norm_query_embedding",
                    "query": query,
                    "top_k": top_k,
                    "retrieval_backend": "numpy_cosine",
                    "embedding_model": self.embedding_model_name,
                    "num_index_items": len(self.product_ids),
                    "num_results": 0,
                    "excluded_items": len(exclude_set),
                    "retrieval_latency_ms": (time.perf_counter() - start) * 1000,
                }

            q = (q / q_norm).astype("float32")

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                scores = np.matmul(self.embeddings.astype("float32"), q)

            scores = np.nan_to_num(
                scores,
                nan=-1e9,
                posinf=-1e9,
                neginf=-1e9,
            ).astype("float32")

            # Retrieve more than top_k first, because some items may be excluded.
            candidate_limit = min(len(scores), max(top_k * 5, top_k + len(exclude_set) + 10))
            candidate_indices = np.argsort(-scores)[:candidate_limit]

            results: list[VectorSearchResult] = []
            seen = set()

            for idx in candidate_indices:
                pid = self.product_ids[int(idx)]
                if pid in seen:
                    continue
                if pid in exclude_set:
                    continue

                score = float(scores[int(idx)])
                if min_score is not None and score < min_score:
                    continue

                meta = self.item_meta.get(pid)
                if meta is None:
                    continue

                results.append(
                    VectorSearchResult(
                        product_id=pid,
                        score=score,
                        title=meta["title"],
                        category=meta["category"],
                        brand=meta["brand"],
                        price=meta["price"],
                        stock=meta["stock"],
                    )
                )
                seen.add(pid)

                if len(results) >= top_k:
                    break

            latency_ms = (time.perf_counter() - start) * 1000

            return results, {
                "success": True,
                "error": None,
                "query": query,
                "top_k": top_k,
                "retrieval_backend": "numpy_cosine",
                "embedding_model": self.embedding_model_name,
                "num_index_items": len(self.product_ids),
                "num_results": len(results),
                "excluded_items": len(exclude_set),
                "retrieval_latency_ms": latency_ms,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return [], {
                "success": False,
                "error": repr(e),
                "query": query,
                "top_k": top_k,
                "retrieval_backend": "numpy_cosine",
                "embedding_model": self.embedding_model_name,
                "num_index_items": len(self.product_ids),
                "num_results": 0,
                "excluded_items": len(exclude_set),
                "retrieval_latency_ms": latency_ms,
            }