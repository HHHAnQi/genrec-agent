import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn


class GRUGenRec(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.num_clusters = num_clusters

        self.embedding = nn.Embedding(
            num_embeddings=num_clusters + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_clusters)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        _, hidden = self.gru(emb)
        last_hidden = hidden[-1]
        logits = self.classifier(self.dropout(last_hidden))
        return logits


class GenRecInference:
    """
    Inference module for GenRec-Agent.

    Supported modes:
    - popularity: global popularity recommendation
    - semantic_neighbor: same Semantic-ID cluster neighbor recommendation
    - genrec_gru: GRU predicts next semantic cluster, then maps clusters back to items
    """

    def __init__(
        self,
        model_path: str = "models/genrec_gru.pt",
        train_path: str = "datasets/processed/train.jsonl",
        semantic_ids_path: str = "datasets/processed/semantic_ids.json",
        sid_to_items_path: str = "datasets/processed/sid_to_items.json",
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.train_path = Path(train_path)
        self.semantic_ids_path = Path(semantic_ids_path)
        self.sid_to_items_path = Path(sid_to_items_path)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.semantic_ids = self._load_json(self.semantic_ids_path)
        self.sid_to_items = self._load_json(self.sid_to_items_path)

        self.item_popularity, self.global_popular_items = self._build_item_popularity(
            self.train_path
        )

        self.model = None
        self.model_args = None
        self.model_loaded = False
        self.load_error = None

        self._try_load_model()

    @staticmethod
    def _load_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _build_item_popularity(self, train_path: Path) -> tuple[Counter, list[str]]:
        rows = self._read_jsonl(train_path)
        counter = Counter()

        for row in rows:
            counter[str(row["target"])] += 1

        ranked_items = [item for item, _ in counter.most_common()]
        return counter, ranked_items

    def _try_load_model(self):
        if not self.model_path.exists():
            self.load_error = f"Model file not found: {self.model_path}"
            self.model_loaded = False
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            args = checkpoint.get("args", {})

            self.model_args = args

            self.model = GRUGenRec(
                num_clusters=int(args.get("num_clusters", 128)),
                embedding_dim=int(args.get("embedding_dim", 64)),
                hidden_dim=int(args.get("hidden_dim", 64)),
                num_layers=int(args.get("num_layers", 1)),
                dropout=float(args.get("dropout", 0.1)),
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.model_loaded = True
            self.load_error = None

        except Exception as e:
            self.model = None
            self.model_loaded = False
            self.load_error = repr(e)

    def get_cluster_id(self, item_id: str) -> int | None:
        sid = self.semantic_ids.get(str(item_id))
        if sid is None:
            return None

        if isinstance(sid, list) and len(sid) > 0:
            return int(sid[0])

        try:
            return int(sid)
        except Exception:
            return None

    def _recommend_popularity(
        self,
        history: Iterable[str],
        top_k: int,
        exclude_history: bool = True,
    ) -> list[str]:
        history_set = set(str(x) for x in history)
        recs = []

        for item in self.global_popular_items:
            item = str(item)
            if exclude_history and item in history_set:
                continue
            recs.append(item)
            if len(recs) >= top_k:
                break

        return recs

    def _recommend_semantic_neighbor(
        self,
        history: Iterable[str],
        top_k: int,
        exclude_history: bool = True,
    ) -> list[str]:
        history = [str(x) for x in history]
        history_set = set(history)

        cluster_counter = Counter()

        for item in history:
            cid = self.get_cluster_id(item)
            if cid is not None:
                cluster_counter[str(cid)] += 1

        candidate_scores = Counter()

        for sid_key, cluster_weight in cluster_counter.most_common():
            items = self.sid_to_items.get(str(sid_key), [])
            for item in items:
                item = str(item)
                if exclude_history and item in history_set:
                    continue

                pop = self.item_popularity.get(item, 0)
                candidate_scores[item] += cluster_weight * 1000 + pop

        recs = []
        seen = set()

        for item, _ in candidate_scores.most_common():
            if item in seen:
                continue
            if exclude_history and item in history_set:
                continue

            recs.append(item)
            seen.add(item)

            if len(recs) >= top_k:
                return recs

        for item in self.global_popular_items:
            item = str(item)
            if item in seen:
                continue
            if exclude_history and item in history_set:
                continue

            recs.append(item)
            seen.add(item)

            if len(recs) >= top_k:
                break

        return recs

    def _recommend_genrec_gru(
        self,
        history: Iterable[str],
        top_k: int,
        top_clusters: int = 5,
        exclude_history: bool = True,
    ) -> list[str]:
        if not self.model_loaded or self.model is None:
            return self._recommend_semantic_neighbor(
                history=history,
                top_k=top_k,
                exclude_history=exclude_history,
            )

        history = [str(x) for x in history]
        history_set = set(history)

        max_history_len = int(self.model_args.get("max_history_len", 20))

        cluster_seq = []
        for item in history:
            cid = self.get_cluster_id(item)
            if cid is not None:
                cluster_seq.append(cid)

        if len(cluster_seq) == 0:
            return self._recommend_popularity(
                history=history,
                top_k=top_k,
                exclude_history=exclude_history,
            )

        cluster_seq = cluster_seq[-max_history_len:]
        token_seq = [cid + 1 for cid in cluster_seq]

        pad_len = max_history_len - len(token_seq)
        if pad_len > 0:
            token_seq = [0] * pad_len + token_seq

        input_ids = torch.tensor([token_seq], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_cluster_ids = np.argsort(-probs)[:top_clusters].tolist()

        candidate_scores = Counter()

        for cid in top_cluster_ids:
            sid_key = str(int(cid))
            cluster_prob = float(probs[cid])
            items = self.sid_to_items.get(sid_key, [])

            for item in items:
                item = str(item)
                if exclude_history and item in history_set:
                    continue

                pop = self.item_popularity.get(item, 0)
                candidate_scores[item] += cluster_prob * 100000 + pop

        recs = []
        seen = set()

        for item, _ in candidate_scores.most_common():
            if item in seen:
                continue
            if exclude_history and item in history_set:
                continue

            recs.append(item)
            seen.add(item)

            if len(recs) >= top_k:
                return recs

        for item in self.global_popular_items:
            item = str(item)
            if item in seen:
                continue
            if exclude_history and item in history_set:
                continue

            recs.append(item)
            seen.add(item)

            if len(recs) >= top_k:
                break

        return recs

    def recommend(
        self,
        history: Iterable[str],
        top_k: int = 10,
        mode: Literal["popularity", "semantic_neighbor", "genrec_gru"] = "genrec_gru",
        top_clusters: int = 5,
        exclude_history: bool = True,
    ) -> dict:
        history = [str(x) for x in history]

        if mode == "popularity":
            recs = self._recommend_popularity(
                history=history,
                top_k=top_k,
                exclude_history=exclude_history,
            )
            used_mode = "popularity"

        elif mode == "semantic_neighbor":
            recs = self._recommend_semantic_neighbor(
                history=history,
                top_k=top_k,
                exclude_history=exclude_history,
            )
            used_mode = "semantic_neighbor"

        elif mode == "genrec_gru":
            if self.model_loaded:
                recs = self._recommend_genrec_gru(
                    history=history,
                    top_k=top_k,
                    top_clusters=top_clusters,
                    exclude_history=exclude_history,
                )
                used_mode = "genrec_gru"
            else:
                recs = self._recommend_semantic_neighbor(
                    history=history,
                    top_k=top_k,
                    exclude_history=exclude_history,
                )
                used_mode = "semantic_neighbor_fallback"

        else:
            raise ValueError(f"Unknown recommend mode: {mode}")

        return {
            "items": recs,
            "requested_mode": mode,
            "used_mode": used_mode,
            "top_k": top_k,
            "top_clusters": top_clusters,
            "model_loaded": self.model_loaded,
            "load_error": self.load_error,
        }


def demo():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, nargs="+", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="genrec_gru",
        choices=["popularity", "semantic_neighbor", "genrec_gru"],
    )
    parser.add_argument("--top_clusters", type=int, default=5)
    parser.add_argument("--exclude_history", action="store_true")

    args = parser.parse_args()

    engine = GenRecInference()

    result = engine.recommend(
        history=args.history,
        top_k=args.top_k,
        mode=args.mode,
        top_clusters=args.top_clusters,
        exclude_history=args.exclude_history,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()