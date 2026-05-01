import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items_path",
        type=str,
        default="datasets/processed/items.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/processed/item_texts.csv",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=1024,
    )
    return parser.parse_args()


def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ").strip()
    s = " ".join(s.split())
    return s


def build_item_text(row, max_text_chars: int):
    title = clean_text(row.get("title", ""))
    category = clean_text(row.get("category", ""))
    brand = clean_text(row.get("brand", ""))
    description = clean_text(row.get("description", ""))

    parts = []
    if title and title.lower() != "unknown":
        parts.append(f"Title: {title}")
    if category and category.lower() != "unknown":
        parts.append(f"Category: {category}")
    if brand and brand.lower() != "unknown":
        parts.append(f"Brand: {brand}")
    if description and description.lower() != "unknown":
        parts.append(f"Description: {description}")

    item_text = ". ".join(parts)
    item_text = item_text[:max_text_chars]

    return item_text


def main():
    args = parse_args()

    items_path = Path(args.items_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading items from: {items_path}")
    items = pd.read_csv(items_path)

    required_cols = ["product_id", "title", "category", "brand", "description"]
    for col in required_cols:
        if col not in items.columns:
            raise ValueError(f"Missing required column: {col}")

    print("🧱 Building item_text...")
    items["item_text"] = items.apply(
        lambda row: build_item_text(row, args.max_text_chars),
        axis=1,
    )

    before = len(items)
    items = items[items["item_text"].str.len() >= 10].copy()
    after = len(items)

    output_cols = [
        "product_id",
        "title",
        "category",
        "brand",
        "description",
        "item_text",
    ]

    items[output_cols].to_csv(output_path, index=False)

    print("\n========== Summary ==========")
    print(f"Input items:        {before:,}")
    print(f"Valid item_texts:   {after:,}")
    print(f"Removed items:      {before - after:,}")
    print(f"Saved to:           {output_path}")
    print("=============================")


if __name__ == "__main__":
    main()