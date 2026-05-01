import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item_texts_path",
        type=str,
        default="datasets/processed/item_texts.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/processed",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    item_texts_path = Path(args.item_texts_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading item_texts from: {item_texts_path}")
    df = pd.read_csv(item_texts_path)

    if "product_id" not in df.columns or "item_text" not in df.columns:
        raise ValueError("item_texts.csv must contain product_id and item_text columns.")

    product_ids = df["product_id"].astype(str).tolist()
    texts = df["item_text"].fillna("").astype(str).tolist()

    if len(product_ids) < args.n_clusters:
        raise ValueError(
            f"n_clusters={args.n_clusters} is larger than num_items={len(product_ids)}"
        )

    print(f"🧠 Loading sentence-transformer: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("🔢 Encoding item texts...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = normalize(embeddings)

    print(f"🧩 Running KMeans: n_clusters={args.n_clusters}")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(embeddings)

    semantic_ids = {}
    sid_to_items = {}

    for pid, cid in zip(product_ids, cluster_ids):
        sid = [int(cid)]
        semantic_ids[pid] = sid

        sid_key = str(int(cid))
        sid_to_items.setdefault(sid_key, []).append(pid)

    semantic_ids_path = output_dir / "semantic_ids.json"
    sid_to_items_path = output_dir / "sid_to_items.json"
    embeddings_path = output_dir / "item_embeddings.npy"
    cluster_centers_path = output_dir / "semantic_cluster_centers.npy"

    with open(semantic_ids_path, "w", encoding="utf-8") as f:
        json.dump(semantic_ids, f, ensure_ascii=False, indent=2)

    with open(sid_to_items_path, "w", encoding="utf-8") as f:
        json.dump(sid_to_items, f, ensure_ascii=False, indent=2)

    np.save(embeddings_path, embeddings)
    np.save(cluster_centers_path, kmeans.cluster_centers_)

    cluster_sizes = pd.Series(cluster_ids).value_counts().sort_index()

    print("\n========== Summary ==========")
    print(f"Items:                   {len(product_ids):,}")
    print(f"Clusters:                {args.n_clusters:,}")
    print(f"Min cluster size:        {cluster_sizes.min()}")
    print(f"Median cluster size:     {cluster_sizes.median():.2f}")
    print(f"Max cluster size:        {cluster_sizes.max()}")
    print(f"Saved semantic ids:      {semantic_ids_path}")
    print(f"Saved sid_to_items:      {sid_to_items_path}")
    print(f"Saved embeddings:        {embeddings_path}")
    print("=============================")


if __name__ == "__main__":
    main()