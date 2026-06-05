"""Export the exact held-out TEST split (labels + images) so accuracy can be
measured on the Pi against the *same* set that produced the 97.15% fp32 number.

Replicates deepweeds/data.py make_splits (stratified, seed=42, 15% test).

Example:
    python export_testset.py --data-dir data/images --labels-csv data/labels.csv --out testset
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out", default="testset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    # Same two-stage stratified split as deepweeds/data.py make_splits()
    _, temp_df = train_test_split(
        df, test_size=args.val_frac + args.test_frac, stratify=df["Label"], random_state=args.seed)
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    _, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["Label"], random_state=args.seed)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    test_df[["Filename", "Label"]].to_csv(out / "labels.csv", index=False)

    copied = 0
    for fn in test_df["Filename"]:
        src = Path(args.data_dir) / fn
        if src.exists():
            shutil.copy(src, out / "images" / fn)
            copied += 1

    print(f"Test split: {len(test_df)} rows, {copied} images copied to {out}/")
    print(f"  -> {out}/labels.csv  +  {out}/images/")


if __name__ == "__main__":
    main()
