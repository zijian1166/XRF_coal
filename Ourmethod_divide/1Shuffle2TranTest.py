#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle Data_10classification.csv and split into train/test/true CSVs"
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input CSV path (default: raw_data/Data_10classification.csv)",
    )
    parser.add_argument(
        "--ratio",
        default="8:2",
        help="Train:test ratio, e.g. 8:2 (default: 8:2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: intermediate/)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if args.input else base_dir / "raw_data" / "Data_10classification.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    try:
        train_ratio, test_ratio = [int(x) for x in args.ratio.split(":", 1)]
        if train_ratio <= 0 or test_ratio <= 0:
            raise ValueError
    except Exception as exc:
        raise ValueError("--ratio must be like 8:2 with positive integers") from exc

    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least one feature column and one label column")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    total = len(df)
    train_size = int(total * train_ratio / (train_ratio + test_ratio))
    train_df = df.iloc[:train_size].copy()
    true_df = df.iloc[train_size:].copy()
    test_df = true_df.iloc[:, :-1].copy()

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    true_path = out_dir / "true.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    true_df.to_csv(true_path, index=False)

    print(f"Input: {input_path}")
    print(f"Train: {train_path} ({len(train_df)})")
    print(f"Test:  {test_path} ({len(test_df)})")
    print(f"True:  {true_path} ({len(true_df)})")


if __name__ == "__main__":
    raise SystemExit(main())
