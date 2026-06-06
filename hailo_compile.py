"""Compile the optimized ONNX model to a deployable Hailo .hef using the same
recipe that produced the validated emulated-quantized accuracy (96.0% on V2).

Runs INSIDE the Hailo AI Software Suite container.

Example:
    python hailo_compile.py --onnx model.onnx --calib-dir calib1152 \
        --out mobilenet_v2.hef --finetune
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

MEAN_255 = [123.675, 116.28, 103.53]
STD_255 = [58.395, 57.12, 57.375]


def load_calib(calib_dir, size=224):
    paths = sorted(Path(calib_dir).glob("*.jpg")) + sorted(Path(calib_dir).glob("*.png"))
    if not paths:
        raise SystemExit(f"no images in {calib_dir}/")
    arr = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size)),
                               dtype=np.float32) for p in paths])
    return arr, len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--model-name", default="deepweeds")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", default="model.hef")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--finetune-epochs", type=int, default=8)
    args = ap.parse_args()

    print(f"Parsing {args.onnx} as '{args.model_name}' ...")
    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(args.onnx, args.model_name,
                                net_input_shapes={"input": [1, 3, 224, 224]})

    calib, n = load_calib(args.calib_dir)
    ms = f"normalization1 = normalization({MEAN_255}, {STD_255})\n"
    ms += f"model_optimization_config(calibration, batch_size=8, calibset_size={n})\n"
    if args.finetune:
        ms += ("post_quantization_optimization(finetune, policy=enabled, "
               f"learning_rate=0.00005, epochs={args.finetune_epochs})\n")
    else:
        ms += "post_quantization_optimization(finetune, policy=disabled)\n"
    ms += "post_quantization_optimization(bias_correction, policy=enabled)\n"
    ms += "post_quantization_optimization(adaround, policy=enabled)\n"

    print(f"Optimizing with {n} calibration images "
          f"{'+ finetune' if args.finetune else '(no finetune)'} ...")
    runner.load_model_script(ms)
    runner.optimize(calib)

    print("Compiling to HEF ...")
    hef = runner.compile()
    Path(args.out).write_bytes(hef)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
