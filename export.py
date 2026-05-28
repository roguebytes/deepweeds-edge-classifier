"""Export a trained checkpoint to ONNX, with optional dynamic INT8 quantization.

Example:
    python export.py --checkpoint runs/mnv3/best_model.pt --arch mobilenet_v3_large \
        --output runs/mnv3/model.onnx --quantize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepweeds.data import NUM_CLASSES
from deepweeds.model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Export DeepWeeds model to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arch", default=None, help="Override arch (else read from checkpoint)")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--output", default="model.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--quantize", action="store_true", help="Also write an INT8 dynamic-quantized model")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = args.arch or ckpt.get("arch", "resnet50")
    img_size = args.img_size or ckpt.get("img_size", 224)

    model = build_model(arch, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    export_kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    try:
        # Force the stable TorchScript exporter. PyTorch 2.x's new "dynamo" exporter
        # (default) can emit graphs that break onnxruntime quantization / opset
        # downconversion; the legacy path produces clean, quantizable ONNX for CNNs.
        torch.onnx.export(model, dummy, args.output, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch (<2.5) has no `dynamo` kwarg and always uses the legacy path.
        torch.onnx.export(model, dummy, args.output, **export_kwargs)
    print(f"Wrote {args.output} (arch={arch}, img_size={img_size})")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_out = str(Path(args.output).with_suffix(".int8.onnx"))
        quantize_dynamic(args.output, q_out, weight_type=QuantType.QInt8)
        print(f"Wrote {q_out} (INT8 dynamic)")


if __name__ == "__main__":
    main()
