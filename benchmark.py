"""Benchmark ONNX inference latency. Run this on the Pi/Jetson for the headline number.

Example:
    python benchmark.py --onnx runs/mnv3/model.onnx --runs 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark ONNX model latency")
    p.add_argument("--onnx", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = ORT default)")
    return p.parse_args()


def main():
    args = parse_args()
    opts = ort.SessionOptions()
    if args.threads > 0:
        opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, args.img_size, args.img_size).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {name: x})

    times = np.empty(args.runs)
    for i in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times[i] = (time.perf_counter() - t0) * 1000.0

    print(f"model: {args.onnx} | providers: {sess.get_providers()}")
    print(f"runs: {args.runs} (after {args.warmup} warmup)")
    print(f"latency ms  mean {times.mean():.2f} | std {times.std():.2f} | "
          f"p50 {np.percentile(times, 50):.2f} | p95 {np.percentile(times, 95):.2f}")
    print(f"throughput  {1000.0 / times.mean():.1f} FPS")


if __name__ == "__main__":
    main()
