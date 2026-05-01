import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Amazon Reviews 2023 All Beauty dataset for GenRec-Agent."
    )

    parser.add_argument(
        "--reviews_path",
        type=str,
        default="datasets/raw/amazon_beauty/reviews.parquet",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="datasets/raw/amazon_beauty/metadata.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/processed",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="datasets/sample",
    )

    # 默认改成更适合项目训练的数据规模
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--min_user_interactions", type=int, default=3)
    parser.add_argument("--min_item_interactions", type=int, default=3)

    # 默认不只保留 verified purchase，避免数据被过滤得太小
    parser.add_argument(
        "--verified_only",
        action="store_true",
        help="If set, keep only verified_purchase=True interactions.",
    )

    # 默认不用循环 k-core，避免长尾电商数据被过滤到只剩很小核心
    parser.add_argument(
        "--iterative_kcore",
        action="store_true",
        help="If set, apply iterative k-core filtering until convergence.",
    )

    parser.add_argument("--sample_users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def stringify_value(x: Any) -> str:
    """Convert possible list/dict/None value from Amazon metadata into clean text."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, list):
        parts = []
        for v in x:
            s = stringify_value(v)
            if s:
                parts.append(s)
        return " ".join(parts)
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if v is not None:
                parts.append(f"{k}: {v}")
        return " ".join(parts)
    return str(x)


def parse_category(row: pd.Series) -> str:
    """Prefer the last category path; fallback to main_category."""
    categories = row.get("categories", None)
    main_category = row.get("main_category", None)

    if isinstance(categories, list) and len(categories) > 0:
        last = categories[-1]

        if isinstance(last, list) and len(last) > 0:
            parsed = stringify_value(last[-1])
            if parsed:
                return parsed

        parsed = stringify_value(last)
        if parsed:
            return parsed

    fallback = stringify_value(main_category)
    return fallback if fallback else "unknown"


def parse_price(x: Any) -> float:
    if x is None:
        return -1.0
    if isinstance(x, float) and pd.isna(x):
        return -1.0

    try:
        s = str(x).replace("$", "").replace(",", "").strip()
        if not s or s.lower() in {"none", "nan", "null"}:
            return -1.0
        return float(s)
    except Exception:
        return -1.0


def simple_filter(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """
    One-pass filter.
    This keeps more data than iterative k-core and is better for the first project version.
    """
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    cur = df[df["user_id"].isin(valid_users)].copy()

    item_counts = cur["product_id"].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    cur = cur[cur["product_id"].isin(valid_items)].copy()

    return cur.reset_index(drop=True)


def iterative_k_core_filter(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """
    Iterative k-core filter.
    More rigorous, but can be too aggressive for sparse/long-tail ecommerce data.
    """
    prev_len = -1
    cur = df.copy()
    round_id = 0

    while prev_len != len(cur):
        round_id += 1
        prev_len = len(cur)

        user_counts = cur["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        cur = cur[cur["user_id"].isin(valid_users)]

        item_counts = cur["product_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        cur = cur[cur["product_id"].isin(valid_items)]

        print(f"  k-core round {round_id}: {len(cur):,} interactions")

    return cur.reset_index(drop=True)


def build_user_sequences(interactions: pd.DataFrame) -> dict[str, list[str]]:
    interactions = interactions.sort_values(["user_id", "timestamp"])
    seqs = (
        interactions.groupby("user_id")["product_id"]
        .apply(lambda x: [str(v) for v in x.tolist()])
        .to_dict()
    )
    return seqs


def build_items(metadata: pd.DataFrame, valid_items: set[str], seed: int) -> pd.DataFrame:
    random.seed(seed)

    metadata = metadata[metadata["parent_asin"].notna()].copy()
    metadata = metadata.rename(columns={"parent_asin": "product_id"})
    metadata["product_id"] = metadata["product_id"].astype(str)
    metadata = metadata[metadata["product_id"].isin(valid_items)].copy()

    items = pd.DataFrame()
    items["product_id"] = metadata["product_id"]
    items["title"] = metadata["title"].apply(stringify_value)
    items["title"] = items["title"].replace("", "unknown")

    items["category"] = metadata.apply(parse_category, axis=1)
    items["category"] = items["category"].replace("", "unknown")

    items["brand"] = metadata["store"].apply(stringify_value)
    items["brand"] = items["brand"].replace("", "unknown")

    desc = metadata["description"].apply(stringify_value)
    feats = metadata["features"].apply(stringify_value)
    items["description"] = (desc + " " + feats).str.strip()
    items["description"] = items["description"].replace("", "unknown")

    items["price"] = metadata["price"].apply(parse_price)

    # Amazon metadata does not provide inventory; simulate it for Filter Agent.
    items["stock"] = [random.randint(0, 100) for _ in range(len(items))]

    items["average_rating"] = metadata["average_rating"].fillna(-1)
    items["rating_number"] = metadata["rating_number"].fillna(0)

    items = items.drop_duplicates("product_id").reset_index(drop=True)
    return items


def save_sample_files(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    user_sequences: dict[str, list[str]],
    sample_dir: Path,
    sample_users: int,
):
    sample_user_ids = list(user_sequences.keys())[:sample_users]

    sample_interactions = interactions[
        interactions["user_id"].isin(sample_user_ids)
    ].copy()

    sample_item_ids = set(sample_interactions["product_id"].unique())
    sample_items = items[items["product_id"].isin(sample_item_ids)].copy()

    sample_sequences = {
        uid: user_sequences[uid]
        for uid in sample_user_ids
        if uid in user_sequences
    }

    sample_interactions.to_csv(sample_dir / "interactions_sample.csv", index=False)
    sample_items.to_csv(sample_dir / "items_sample.csv", index=False)

    with open(sample_dir / "user_sequences_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_sequences, f, ensure_ascii=False, indent=2)

    return sample_sequences


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    sample_dir = Path(args.sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("========== Config ==========")
    print(f"reviews_path:              {args.reviews_path}")
    print(f"metadata_path:             {args.metadata_path}")
    print(f"output_dir:                {output_dir}")
    print(f"sample_dir:                {sample_dir}")
    print(f"min_rating:                {args.min_rating}")
    print(f"verified_only:             {args.verified_only}")
    print(f"min_user_interactions:     {args.min_user_interactions}")
    print(f"min_item_interactions:     {args.min_item_interactions}")
    print(f"iterative_kcore:           {args.iterative_kcore}")
    print("==========================\n")

    print("📥 Loading raw reviews...")
    reviews = pd.read_parquet(args.reviews_path)
    raw_reviews_count = len(reviews)
    print(f"Raw reviews: {raw_reviews_count:,}")

    print("🧹 Filtering interactions...")
    mask = (
        reviews["parent_asin"].notna()
        & reviews["user_id"].notna()
        & reviews["timestamp"].notna()
        & (reviews["rating"] >= args.min_rating)
    )

    if args.verified_only:
        mask = mask & (reviews["verified_purchase"] == True)

    interactions = reviews[mask].copy()
    after_rating_filter_count = len(interactions)
    print(f"After rating/valid field filter: {after_rating_filter_count:,}")

    interactions = interactions.rename(columns={"parent_asin": "product_id"})
    interactions["event_type"] = "purchase"

    interactions = interactions[
        ["user_id", "product_id", "event_type", "rating", "timestamp"]
    ].copy()

    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["product_id"] = interactions["product_id"].astype(str)
    interactions = interactions.sort_values(["user_id", "timestamp"])

    print("🔁 Applying interaction frequency filter...")
    if args.iterative_kcore:
        interactions = iterative_k_core_filter(
            interactions,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
    else:
        interactions = simple_filter(
            interactions,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )

    after_frequency_filter_count = len(interactions)
    print(f"After frequency filter: {after_frequency_filter_count:,}")

    valid_items = set(interactions["product_id"].unique())

    print("📥 Loading metadata...")
    metadata = pd.read_parquet(args.metadata_path)

    print("🧱 Building items.csv...")
    items = build_items(metadata, valid_items=valid_items, seed=args.seed)

    # Keep only interactions whose item metadata exists.
    item_set = set(items["product_id"].unique())
    interactions = interactions[interactions["product_id"].isin(item_set)].reset_index(
        drop=True
    )
    # Final user filter for next-item prediction.
    user_counts = interactions["user_id"].value_counts()
    valid_users = user_counts[user_counts >= args.min_user_interactions].index
    interactions = interactions[interactions["user_id"].isin(valid_users)].reset_index(drop=True)

    # Rebuild items after final user filtering.
    items = items[items["product_id"].isin(set(interactions["product_id"].unique()))]
    items = items.reset_index(drop=True)

    # Rebuild valid items after metadata filtering.
    items = items[items["product_id"].isin(set(interactions["product_id"].unique()))]
    items = items.reset_index(drop=True)

    print("📚 Building user_sequences.json...")
    user_sequences = build_user_sequences(interactions)

    print("💾 Saving processed files...")
    interactions_path = output_dir / "interactions.csv"
    items_path = output_dir / "items.csv"
    seqs_path = output_dir / "user_sequences.json"

    interactions.to_csv(interactions_path, index=False)
    items.to_csv(items_path, index=False)

    with open(seqs_path, "w", encoding="utf-8") as f:
        json.dump(user_sequences, f, ensure_ascii=False, indent=2)

    print("🧪 Building sample files...")
    sample_sequences = save_sample_files(
        interactions=interactions,
        items=items,
        user_sequences=user_sequences,
        sample_dir=sample_dir,
        sample_users=args.sample_users,
    )

    n_users = interactions["user_id"].nunique()
    n_items = interactions["product_id"].nunique()
    avg_user_len = len(interactions) / max(n_users, 1)

    user_lens = interactions.groupby("user_id").size()
    item_lens = interactions.groupby("product_id").size()

    print("\n========== Summary ==========")
    print(f"Raw reviews:                         {raw_reviews_count:,}")
    print(f"After rating/valid field filter:     {after_rating_filter_count:,}")
    print(f"After frequency filter:              {after_frequency_filter_count:,}")
    print(f"Final processed interactions:        {len(interactions):,}")
    print(f"Users:                               {n_users:,}")
    print(f"Items in interactions:               {n_items:,}")
    print(f"Items metadata:                      {len(items):,}")
    print(f"Avg interactions per user:           {avg_user_len:.2f}")
    print(f"Min interactions per user:           {user_lens.min() if len(user_lens) else 0}")
    print(f"Median interactions per user:        {user_lens.median() if len(user_lens) else 0:.2f}")
    print(f"Max interactions per user:           {user_lens.max() if len(user_lens) else 0}")
    print(f"Min interactions per item:           {item_lens.min() if len(item_lens) else 0}")
    print(f"Median interactions per item:        {item_lens.median() if len(item_lens) else 0:.2f}")
    print(f"Max interactions per item:           {item_lens.max() if len(item_lens) else 0}")
    print(f"Sample users:                        {len(sample_sequences):,}")
    print(f"Saved interactions:                  {interactions_path}")
    print(f"Saved items:                         {items_path}")
    print(f"Saved user sequences:                {seqs_path}")
    print("=============================")


if __name__ == "__main__":
    main()