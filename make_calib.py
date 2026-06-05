"""Build a class-balanced calibration image set for Hailo INT8 quantization.

The Hailo Dataflow Compiler uses a small set of representative images to choose
INT8 activation ranges. This samples ~N images evenly across the 9 DeepWeeds
classes into an output dir, ready to pass to `hailomz compile --calib-path`.

Example:
    python make_calib.py --data-dir data/images --labels-csv data/labels.csv \
        --out calib --total 64
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build a class-balanced Hailo calibration set")
    p.add_argument("--data-dir", required=True, help="Dir containing the image files")
    p.add_argument("--labels-csv", required=True, help="CSV with columns Filename,Label,Species")
    p.add_argument("--out", default="calib", help="Output dir for the calibration images")
    p.add_argument("--total", type=int, default=64, help="Approx total images, split across classes")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    df = pd.read_csv(args.labels_csv)

    by_class: dict[int, list[str]] = defaultdict(list)
    for fn, lbl in zip(df["Filename"], df["Label"]):
        by_class[int(lbl)].append(fn)

    n_classes = len(by_class)
    per_class = max(1, args.total // n_classes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for lbl in sorted(by_class):
        pick = random.sample(by_class[lbl], min(per_class, len(by_class[lbl])))
        for fn in pick:
            src = Path(args.data_dir) / fn
            if src.exists():
                shutil.copy(src, out / fn)
                copied += 1
        print(f"class {lbl}: {len(pick)} images")

    print(f"\nWrote {copied} calibration images to {out}/  ({n_classes} classes, ~{per_class}/class)")
    print(f"Next (on the x86 host): hailomz compile mobilenet_v3_large --ckpt model.onnx "
          f"--hw-arch hailo8 --calib-path {out} --classes 9")


if __name__ == "__main__":
    main()
