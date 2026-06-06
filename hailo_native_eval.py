"""Full-precision (native, un-quantized) accuracy of the DeepWeeds ONNX, parsed directly
via the Hailo Dataflow Compiler. RUNS INSIDE the Hailo AI Software Suite container.

Purpose: isolate PARSE correctness from QUANTIZATION.
  - native top-1 ~= the 97.15% fp32 baseline  -> parse + weights are faithful; any
    on-device drop is PURELY quantization (fix: more calibration images + finetune).
  - native top-1 also low                      -> the parse/preprocessing is the problem.

We parse the BARE onnx (no on-chip normalization layer), so we feed PRE-NORMALIZED inputs
(ImageNet, exactly matching the PyTorch eval transform: resize 224 -> /255 -> (x-mean)/std).

Run in the suite container (cd /local/shared_with_docker first):
    # native (full-precision) accuracy — isolates parse from quant:
    python hailo_native_eval.py --images testset/images --labels-csv testset/labels.csv --limit 200
    # emulated quantized accuracy with a larger calibration set + finetune:
    python hailo_native_eval.py --images testset/images --labels-csv testset/labels.csv \
        --limit 200 --quantized --calib-dir calib512 --finetune
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner, InferenceContext

CLASSES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
           "Rubber vine", "Siam weed", "Snake weed", "Negative"]
# ImageNet stats SCALED to [0,255] for the on-chip normalization layer (added via the
# model script when --quantized is used). The Python preprocess feeds raw [0,255] uint8
# for the QUANTIZED path; the on-chip layer then subtracts MEAN_255 and divides by STD_255.
# For the NATIVE path the model has no on-chip norm, so we apply ImageNet norm in Python.
MEAN_255 = [123.675, 116.28, 103.53]   # (0.485,0.456,0.406)*255
STD_255 = [58.395, 57.12, 57.375]      # (0.229,0.224,0.225)*255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rows(csv_path):
    with open(csv_path) as f:
        return [(r["Filename"], int(r["Label"])) for r in csv.DictReader(f)]


def preprocess(path, size=224, raw_uint8=False):
    img = Image.open(path).convert("RGB").resize((size, size))
    if raw_uint8:
        # Feed [0,255] floats; on-chip normalization layer will subtract MEAN/STD.
        return np.asarray(img, dtype=np.float32)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC, [0,1]
    return (arr - MEAN) / STD                          # ImageNet-normalized, HWC float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quantized", action="store_true",
                    help="run SDK_QUANTIZED (emulated INT8) instead of SDK_NATIVE")
    ap.add_argument("--calib-dir", default="calib",
                    help="dir of calibration images (used only with --quantized)")
    ap.add_argument("--finetune", action="store_true",
                    help="enable post-quantization finetune (QAT) — slow but recovers accuracy")
    ap.add_argument("--finetune-epochs", type=int, default=8,
                    help="QAT epochs (8 default; 30+ for quant-hostile MobileNetV3 / low SNR)")
    args = ap.parse_args()

    rows = load_rows(args.labels_csv)
    if args.limit:
        rows = rows[:args.limit]

    print(f"Parsing {args.onnx} via the DFC ...")
    runner = ClientRunner(hw_arch="hailo8")
    # Minimal translate; if it complains about end nodes, add end_node_names=["logits"].
    runner.translate_onnx_model(args.onnx, "deepweeds_mnv3",
                                net_input_shapes={"input": [1, 3, 224, 224]})

    raw_uint8 = args.quantized   # quantized path uses on-chip norm; native path normalizes in Python
    if args.quantized:
        calib_paths = sorted(Path(args.calib_dir).glob("*.jpg")) + sorted(Path(args.calib_dir).glob("*.png"))
        if not calib_paths:
            raise SystemExit(f"no images in {args.calib_dir}/")
        print(f"Optimizing with {len(calib_paths)} calibration images (raw [0,255] uint8) "
              f"{'+ finetune' if args.finetune else '(no finetune)'} + on-chip ImageNet norm ...")
        calib_arr = np.stack([preprocess(p, raw_uint8=True) for p in calib_paths])  # (N, 224, 224, 3)
        # On-chip normalization (fused as first layer); MUST come before quantization commands.
        ms = f"normalization1 = normalization({MEAN_255}, {STD_255})\n"
        # Use ALL passed calibration images (the SDK truncates to a default ~64 otherwise).
        ms += f"model_optimization_config(calibration, batch_size=8, calibset_size={len(calib_paths)})\n"
        if args.finetune:
            # QAT: --finetune-epochs controls how long QAT runs. 8 default; bump for MobileNetV3-class.
            ms += ("post_quantization_optimization(finetune, policy=enabled, "
                   f"learning_rate=0.00005, epochs={args.finetune_epochs})\n")
        else:
            ms += "post_quantization_optimization(finetune, policy=disabled)\n"
        ms += "post_quantization_optimization(bias_correction, policy=enabled)\n"
        ms += "post_quantization_optimization(adaround, policy=enabled)\n"
        runner.load_model_script(ms)
        runner.optimize(calib_arr)
        ctx_kind = InferenceContext.SDK_QUANTIZED
        label = "QUANTIZED (emulated INT8)"
    else:
        ctx_kind = InferenceContext.SDK_NATIVE
        label = "NATIVE (full-precision)"

    correct = total = 0
    pc_c = [0] * 9
    pc_t = [0] * 9
    print(f"{label} inference over {len(rows)} images ...")
    with runner.infer_context(ctx_kind) as ctx:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            batch = np.stack([preprocess(Path(args.images) / fn, raw_uint8=raw_uint8)
                              for fn, _ in chunk])  # (B,224,224,3) float32
            out = runner.infer(ctx, batch)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            logits = np.asarray(out).reshape(len(chunk), -1)  # (B, 9)
            preds = logits.argmax(axis=1)
            for (fn, label), pred in zip(chunk, preds):
                correct += int(pred == label)
                total += 1
                pc_t[label] += 1
                pc_c[label] += int(pred == label)
            print(f"  {total}/{len(rows)}  running top-1 = {correct / total:.4f}")

    print(f"\n{label} Top-1: {correct / total:.4f}  ({correct}/{total})")
    for i, name in enumerate(CLASSES):
        if pc_t[i]:
            print(f"  {name:16s} {pc_c[i] / pc_t[i]:.3f}  ({pc_c[i]}/{pc_t[i]})")


if __name__ == "__main__":
    main()
