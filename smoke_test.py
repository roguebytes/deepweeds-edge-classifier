"""End-to-end smoke test on synthetic data — no real dataset or network needed.

Generates a tiny fake DeepWeeds dataset (random images, all 9 classes), then runs
train -> export -> benchmark with --no-pretrained and tiny settings to confirm the
whole pipeline wires together. Requires the deps in requirements.txt.

    python smoke_test.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from deepweeds.data import CLASSES

N_PER_CLASS = 12   # enough for a stratified train/val/test split across 9 classes
IMG = 64           # tiny images keep the smoke test fast


def make_dataset(root: Path):
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    rows = []
    for label, species in enumerate(CLASSES):
        for i in range(N_PER_CLASS):
            fname = f"cls{label}_{i}.jpg"
            arr = (np.random.rand(IMG, IMG, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(images_dir / fname)
            rows.append((fname, label, species))
    labels_csv = root / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Label", "Species"])
        writer.writerows(rows)
    return images_dir, labels_csv


def run(cmd):
    print("›", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    py = sys.executable
    repo = Path(__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        images_dir, labels_csv = make_dataset(tmp)
        out = tmp / "run"

        run([py, str(repo / "train.py"),
             "--data-dir", str(images_dir), "--labels-csv", str(labels_csv),
             "--arch", "mobilenet_v3_large", "--no-pretrained",
             "--epochs", "1", "--batch-size", "4", "--img-size", str(IMG),
             "--num-workers", "0", "--output-dir", str(out)])
        for artifact in ("best_model.pt", "metrics.json", "confusion_matrix.png"):
            assert (out / artifact).exists(), f"missing {artifact}"

        onnx_path = out / "model.onnx"
        run([py, str(repo / "export.py"),
             "--checkpoint", str(out / "best_model.pt"),
             "--output", str(onnx_path), "--img-size", str(IMG), "--quantize"])
        assert onnx_path.exists(), "missing model.onnx"
        assert onnx_path.with_suffix(".int8.onnx").exists(), "missing int8 model"

        run([py, str(repo / "benchmark.py"),
             "--onnx", str(onnx_path), "--img-size", str(IMG),
             "--runs", "10", "--warmup", "2"])

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
