import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_sequences_path",
        type=str,
        default="datasets/processed/user_sequences.json",
    )
    parser.add_argument(
        "--semantic_ids_path",
        type=str,
        default="datasets/processed/semantic_ids.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/processed",
    )
    parser.add_argument(
        "--min_history_len",
        type=int,
        default=2,
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading user sequences: {args.user_sequences_path}")
    with open(args.user_sequences_path, "r", encoding="utf-8") as f:
        user_sequences = json.load(f)

    print(f"📥 Loading semantic ids: {args.semantic_ids_path}")
    with open(args.semantic_ids_path, "r", encoding="utf-8") as f:
        semantic_ids = json.load(f)

    train_rows = []
    valid_rows = []
    test_rows = []

    skipped_users = 0

    for user_id, seq in user_sequences.items():
        seq = [str(x) for x in seq if str(x) in semantic_ids]

        if len(seq) < 4:
            skipped_users += 1
            continue

        # Leave-one-out:
        # train: sliding samples before valid/test
        # valid target: second last item
        # test target: last item
        valid_target = seq[-2]
        test_target = seq[-1]

        valid_history = seq[:-2]
        test_history = seq[:-1]

        if len(valid_history) >= args.min_history_len:
            valid_rows.append(
                {
                    "user_id": user_id,
                    "history": valid_history,
                    "target": valid_target,
                    "target_semantic_id": semantic_ids[valid_target],
                }
            )

        if len(test_history) >= args.min_history_len:
            test_rows.append(
                {
                    "user_id": user_id,
                    "history": test_history,
                    "target": test_target,
                    "target_semantic_id": semantic_ids[test_target],
                }
            )

        # Sliding training samples:
        # [i0, i1] -> i2
        # [i0, i1, i2] -> i3
        # Do not use final two targets.
        train_part = seq[:-2]
        for t in range(args.min_history_len, len(train_part)):
            history = train_part[:t]
            target = train_part[t]
            train_rows.append(
                {
                    "user_id": user_id,
                    "history": history,
                    "target": target,
                    "target_semantic_id": semantic_ids[target],
                }
            )

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path = output_dir / "test.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    write_jsonl(test_path, test_rows)

    summary = {
        "num_users": len(user_sequences),
        "skipped_users": skipped_users,
        "train_samples": len(train_rows),
        "valid_samples": len(valid_rows),
        "test_samples": len(test_rows),
        "min_history_len": args.min_history_len,
    }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Summary ==========")
    print(f"Users:              {len(user_sequences):,}")
    print(f"Skipped users:      {skipped_users:,}")
    print(f"Train samples:      {len(train_rows):,}")
    print(f"Valid samples:      {len(valid_rows):,}")
    print(f"Test samples:       {len(test_rows):,}")
    print(f"Saved train:        {train_path}")
    print(f"Saved valid:        {valid_path}")
    print(f"Saved test:         {test_path}")
    print(f"Saved summary:      {summary_path}")
    print("=============================")


if __name__ == "__main__":
    main()