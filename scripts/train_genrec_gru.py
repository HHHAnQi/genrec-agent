import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight GRU GenRec model over Semantic ID clusters."
    )

    parser.add_argument("--train_path", type=str, default="datasets/processed/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="datasets/processed/valid.jsonl")
    parser.add_argument("--test_path", type=str, default="datasets/processed/test.jsonl")
    parser.add_argument("--semantic_ids_path", type=str, default="datasets/processed/semantic_ids.json")
    parser.add_argument("--sid_to_items_path", type=str, default="datasets/processed/sid_to_items.json")
    parser.add_argument("--output_dir", type=str, default="models")

    parser.add_argument("--num_clusters", type=int, default=128)
    parser.add_argument("--max_history_len", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--top_clusters", type=int, default=5)
    parser.add_argument("--exclude_history", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cluster_id(item_id: str, semantic_ids: dict) -> int | None:
    sid = semantic_ids.get(str(item_id))
    if sid is None:
        return None
    if isinstance(sid, list) and len(sid) > 0:
        return int(sid[0])
    return int(sid)


def rows_to_cluster_examples(
    rows: list[dict],
    semantic_ids: dict,
    max_history_len: int,
) -> list[dict]:
    examples = []

    for row in rows:
        history_items = [str(x) for x in row["history"]]
        target_item = str(row["target"])

        history_clusters = []
        for item in history_items:
            cid = get_cluster_id(item, semantic_ids)
            if cid is not None:
                history_clusters.append(cid)

        target_cluster = get_cluster_id(target_item, semantic_ids)

        if target_cluster is None or len(history_clusters) == 0:
            continue

        history_clusters = history_clusters[-max_history_len:]

        examples.append(
            {
                "user_id": row.get("user_id", ""),
                "history_items": history_items,
                "history_clusters": history_clusters,
                "target_item": target_item,
                "target_cluster": target_cluster,
            }
        )

    return examples


class ClusterSeqDataset(Dataset):
    def __init__(self, examples: list[dict], max_history_len: int):
        self.examples = examples
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = ex["history_clusters"][-self.max_history_len:]
        target = int(ex["target_cluster"])

        # reserve 0 for PAD; cluster ids become cluster_id + 1
        seq = [int(x) + 1 for x in seq]

        length = len(seq)
        pad_len = self.max_history_len - length
        if pad_len > 0:
            seq = [0] * pad_len + seq

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        }


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

        # +1 because 0 is PAD, real cluster token = cluster_id + 1
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
        output, hidden = self.gru(emb)

        # hidden: [num_layers, batch, hidden_dim]
        last_hidden = hidden[-1]
        logits = self.classifier(self.dropout(last_hidden))
        return logits


def build_item_popularity(train_rows: list[dict]) -> tuple[Counter, list[str]]:
    counter = Counter()
    for row in train_rows:
        counter[str(row["target"])] += 1

    ranked_items = [item for item, _ in counter.most_common()]
    return counter, ranked_items


def recall_at_k(recs: list[str], target: str, k: int) -> float:
    return 1.0 if target in recs[:k] else 0.0


def hitrate_at_k(recs: list[str], target: str, k: int) -> float:
    return 1.0 if target in recs[:k] else 0.0


def ndcg_at_k(recs: list[str], target: str, k: int) -> float:
    topk = recs[:k]
    if target not in topk:
        return 0.0
    rank = topk.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def make_recommendations_from_cluster_probs(
    cluster_probs: np.ndarray,
    history: Iterable[str],
    sid_to_items: dict,
    item_popularity: Counter,
    global_popular_items: list[str],
    k: int,
    top_clusters: int,
    exclude_history: bool,
) -> list[str]:
    history_set = set(str(x) for x in history)

    top_cluster_ids = np.argsort(-cluster_probs)[:top_clusters].tolist()

    candidate_scores = Counter()

    for rank, cid in enumerate(top_cluster_ids):
        cluster_score = float(cluster_probs[cid])
        sid_key = str(int(cid))
        items = sid_to_items.get(sid_key, [])

        for item in items:
            item = str(item)
            if exclude_history and item in history_set:
                continue

            pop = item_popularity.get(item, 0)
            # cluster probability dominates; popularity breaks ties within cluster
            candidate_scores[item] += cluster_score * 100000 + pop

    recs = []
    seen = set()

    for item, _ in candidate_scores.most_common():
        if item in seen:
            continue
        if exclude_history and item in history_set:
            continue
        recs.append(item)
        seen.add(item)
        if len(recs) >= k:
            return recs

    # fallback fill with global popularity
    for item in global_popular_items:
        item = str(item)
        if item in seen:
            continue
        if exclude_history and item in history_set:
            continue
        recs.append(item)
        seen.add(item)
        if len(recs) >= k:
            break

    return recs


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    examples: list[dict],
    sid_to_items: dict,
    item_popularity: Counter,
    global_popular_items: list[str],
    ks: list[int],
    max_history_len: int,
    top_clusters: int,
    exclude_history: bool,
    device: torch.device,
) -> dict:
    model.eval()

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = []
        metrics[f"HitRate@{k}"] = []
        metrics[f"NDCG@{k}"] = []

    for ex in examples:
        seq = ex["history_clusters"][-max_history_len:]
        seq = [int(x) + 1 for x in seq]

        pad_len = max_history_len - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq

        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        max_k = max(ks)
        recs = make_recommendations_from_cluster_probs(
            cluster_probs=probs,
            history=ex["history_items"],
            sid_to_items=sid_to_items,
            item_popularity=item_popularity,
            global_popular_items=global_popular_items,
            k=max_k,
            top_clusters=top_clusters,
            exclude_history=exclude_history,
        )

        target = str(ex["target_item"])

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(recs, target, k))
            metrics[f"HitRate@{k}"].append(hitrate_at_k(recs, target, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(recs, target, k))

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("📥 Loading data...")
    train_rows = read_jsonl(args.train_path)
    valid_rows = read_jsonl(args.valid_path)
    test_rows = read_jsonl(args.test_path)
    semantic_ids = load_json(args.semantic_ids_path)
    sid_to_items = load_json(args.sid_to_items_path)

    print(f"Train rows: {len(train_rows):,}")
    print(f"Valid rows: {len(valid_rows):,}")
    print(f"Test rows:  {len(test_rows):,}")
    print(f"Semantic IDs: {len(semantic_ids):,}")
    print(f"Clusters in sid_to_items: {len(sid_to_items):,}")

    print("🔁 Converting item histories to cluster histories...")
    train_examples = rows_to_cluster_examples(
        train_rows, semantic_ids, args.max_history_len
    )
    valid_examples = rows_to_cluster_examples(
        valid_rows, semantic_ids, args.max_history_len
    )
    test_examples = rows_to_cluster_examples(
        test_rows, semantic_ids, args.max_history_len
    )

    print(f"Train examples: {len(train_examples):,}")
    print(f"Valid examples: {len(valid_examples):,}")
    print(f"Test examples:  {len(test_examples):,}")

    item_popularity, global_popular_items = build_item_popularity(train_rows)

    train_dataset = ClusterSeqDataset(train_examples, args.max_history_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = GRUGenRec(
        num_clusters=args.num_clusters,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_valid_recall = -1.0
    best_epoch = -1
    patience_count = 0
    best_model_path = output_dir / "genrec_gru.pt"

    training_log = []

    print("\n🚀 Training GRU GenRec...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        valid_metrics = evaluate_model(
            model=model,
            examples=valid_examples,
            sid_to_items=sid_to_items,
            item_popularity=item_popularity,
            global_popular_items=global_popular_items,
            ks=args.ks,
            max_history_len=args.max_history_len,
            top_clusters=args.top_clusters,
            exclude_history=args.exclude_history,
            device=device,
        )

        monitor_key = "Recall@20" if "Recall@20" in valid_metrics else f"Recall@{max(args.ks)}"
        monitor_value = valid_metrics[monitor_key]

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_metrics": valid_metrics,
        }
        training_log.append(log_row)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"valid {monitor_key}={monitor_value:.4f} | "
            f"valid NDCG@20={valid_metrics.get('NDCG@20', 0.0):.4f}"
        )

        if monitor_value > best_valid_recall:
            best_valid_recall = monitor_value
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_valid_recall": best_valid_recall,
                },
                best_model_path,
            )
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print("\n📦 Loading best model...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("📊 Evaluating best model on valid/test...")
    best_valid_metrics = evaluate_model(
        model=model,
        examples=valid_examples,
        sid_to_items=sid_to_items,
        item_popularity=item_popularity,
        global_popular_items=global_popular_items,
        ks=args.ks,
        max_history_len=args.max_history_len,
        top_clusters=args.top_clusters,
        exclude_history=args.exclude_history,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model,
        examples=test_examples,
        sid_to_items=sid_to_items,
        item_popularity=item_popularity,
        global_popular_items=global_popular_items,
        ks=args.ks,
        max_history_len=args.max_history_len,
        top_clusters=args.top_clusters,
        exclude_history=args.exclude_history,
        device=device,
    )

    results = {
        "model": "gru_genrec_cluster",
        "best_epoch": best_epoch,
        "best_valid_recall": best_valid_recall,
        "args": vars(args),
        "num_train_examples": len(train_examples),
        "num_valid_examples": len(valid_examples),
        "num_test_examples": len(test_examples),
        "valid": best_valid_metrics,
        "test": test_metrics,
        "training_log": training_log,
        "model_path": str(best_model_path),
    }

    results_path = output_dir / "genrec_gru_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n========== GRU GenRec Results ==========")
    print(f"Best epoch: {best_epoch}")
    print("Valid:")
    for k, v in best_valid_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Test:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved model to:   {best_model_path}")
    print(f"Saved results to: {results_path}")
    print("=======================================")


if __name__ == "__main__":
    main()