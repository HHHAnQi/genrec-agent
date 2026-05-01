from datasets import load_dataset

SAVE_DIR = "datasets/raw/amazon_beauty"

def main():
    print("Downloading Amazon Reviews 2023 - All Beauty reviews...")
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True,
    )
    reviews.to_parquet(f"{SAVE_DIR}/reviews.parquet")
    print(f"Saved reviews: {len(reviews)} rows")

    print("Downloading Amazon Reviews 2023 - All Beauty metadata...")
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_All_Beauty",
        split="full",
        trust_remote_code=True,
    )
    meta.to_parquet(f"{SAVE_DIR}/metadata.parquet")
    print(f"Saved metadata: {len(meta)} rows")

if __name__ == "__main__":
    main()