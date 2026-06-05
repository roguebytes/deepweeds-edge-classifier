"""Measure top-1 accuracy of a compiled Hailo .hef over the DeepWeeds test split,
running on-device via HailoRT (intended for the Raspberry Pi 5 + Hailo-8).

The .hef was compiled from the Hailo Model Zoo mobilenet_v3 config, which applies
ImageNet normalization ON-CHIP — so we feed RAW uint8 RGB images resized to the
network input (no normalization in Python). If accuracy comes out near-random
(~11% for 9 classes), the on-chip normalization didn't match training and we adjust.

Example (quick check, then full):
    python3 hailo_accuracy.py --hef mobilenet_v3.hef --images testset/images \
        --labels-csv testset/labels.csv --limit 50
    python3 hailo_accuracy.py --hef mobilenet_v3.hef --images testset/images \
        --labels-csv testset/labels.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams,
                            FormatType)

CLASSES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
           "Rubber vine", "Siam weed", "Snake weed", "Negative"]


def load_rows(csv_path):
    with open(csv_path) as f:
        return [(r["Filename"], int(r["Label"])) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--images", required=True, help="dir of test images")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all) for a quick check")
    ap.add_argument("--bgr", action="store_true", help="feed BGR channel order (test if the .hef expects BGR)")
    args = ap.parse_args()

    rows = load_rows(args.labels_csv)
    if args.limit:
        rows = rows[:args.limit]

    hef = HEF(args.hef)
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    h, w, _ = in_info.shape  # NHWC
    print(f"input '{in_info.name}' shape={in_info.shape} | output '{out_info.name}' shape={out_info.shape}")
    print(f"evaluating {len(rows)} images at {w}x{h}, feeding raw uint8 (on-chip normalization)\n")

    correct = total = 0
    pc_correct = [0] * 9
    pc_total = [0] * 9

    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()
        in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        with InferVStreams(ng, in_params, out_params) as pipeline:
            with ng.activate(ng_params):
                for fn, label in rows:
                    img = Image.open(Path(args.images) / fn).convert("RGB").resize((w, h))
                    arr = np.array(img, dtype=np.uint8)  # (H, W, 3) RGB, writeable copy
                    if args.bgr:
                        arr = arr[:, :, ::-1]            # RGB -> BGR (some Hailo configs expect BGR)
                    # ascontiguousarray -> writeable + C-contiguous (HailoRT requires both)
                    arr = np.ascontiguousarray(arr)[None, ...]  # (1, H, W, 3)
                    res = pipeline.infer({in_info.name: arr})
                    logits = np.asarray(res[out_info.name]).reshape(-1)  # (9,)
                    pred = int(np.argmax(logits))
                    correct += int(pred == label)
                    total += 1
                    pc_total[label] += 1
                    pc_correct[label] += int(pred == label)
                    if total % 200 == 0:
                        print(f"  {total}/{len(rows)}  running top-1 = {correct / total:.4f}")

    print(f"\nTop-1 accuracy: {correct / total:.4f}  ({correct}/{total})")
    print("Per-class recall:")
    for i, name in enumerate(CLASSES):
        if pc_total[i]:
            print(f"  {name:16s} {pc_correct[i] / pc_total[i]:.3f}  ({pc_correct[i]}/{pc_total[i]})")


if __name__ == "__main__":
    main()
