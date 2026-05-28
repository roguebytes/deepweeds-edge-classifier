"""Bridge-to-aerial demo: tile a large aerial image, classify each tile with the
trained DeepWeeds model, and render a coarse weed-presence heatmap overlay.

CAVEAT — read before trusting the output:
  * DeepWeeds is trained on GROUND-LEVEL photos; running it on aerial tiles is
    out-of-distribution. Treat this as a proof-of-concept link from the classifier
    to aerial use, NOT an agronomic product.
  * The heatmap is in PIXEL space, not geo-referenced. Producing a real paddock
    map needs an orthomosaic + georeferencing (GeoTIFF), and ideally a model
    fine-tuned on labelled aerial tiles. Those are the next steps.

    python aerial_tiling.py --checkpoint runs/mnv3/best_model.pt \
        --image flight.jpg --tile 256 --stride 256 --output heatmap.png
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deepweeds.data import CLASSES, NUM_CLASSES, build_transforms
from deepweeds.model import build_model

NEGATIVE_IDX = CLASSES.index("Negative")


def parse_args():
    p = argparse.ArgumentParser(description="Tile-classify an aerial image into a weed heatmap")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--arch", default=None, help="Override arch (else read from checkpoint)")
    p.add_argument("--tile", type=int, default=256, help="Tile size in px (256 ~ DeepWeeds native)")
    p.add_argument("--stride", type=int, default=256, help="Step in px; < tile gives overlap")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity")
    p.add_argument("--output", default="heatmap.png")
    p.add_argument("--csv", default=None, help="Optional per-tile predictions CSV")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tile_origins(extent: int, tile: int, stride: int):
    """Start coords covering [0, extent) with a final tile flush to the edge."""
    if extent <= tile:
        return [0]
    xs = list(range(0, extent - tile + 1, stride))
    if xs[-1] != extent - tile:
        xs.append(extent - tile)
    return xs


@torch.no_grad()
def run_batch(model, transform, image, boxes, device):
    tensors = [transform(image.crop(b)) for b in boxes]
    logits = model(torch.stack(tensors).to(device))
    return F.softmax(logits, dim=1).cpu().numpy()


def main():
    args = parse_args()
    device = pick_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    arch = args.arch or ckpt.get("arch", "resnet50")
    img_size = ckpt.get("img_size", 224)
    model = build_model(arch, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    transform = build_transforms(img_size, train=False)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size
    tile = min(args.tile, W, H)
    xs = tile_origins(W, tile, args.stride)
    ys = tile_origins(H, tile, args.stride)

    boxes = [(x, y, x + tile, y + tile) for y in ys for x in xs]
    print(f"{args.image}: {W}x{H} -> {len(boxes)} tiles ({len(xs)}x{len(ys)}), tile={tile}")

    # Per-pixel accumulation correctly handles overlapping tiles (stride < tile).
    heat_sum = np.zeros((H, W), dtype=np.float32)
    heat_cnt = np.zeros((H, W), dtype=np.float32)
    rows = []

    for i in range(0, len(boxes), args.batch_size):
        chunk = boxes[i:i + args.batch_size]
        probs = run_batch(model, transform, image, chunk, device)
        for (x0, y0, x1, y1), p in zip(chunk, probs):
            weed_prob = float(1.0 - p[NEGATIVE_IDX])
            top = int(p.argmax())
            heat_sum[y0:y1, x0:x1] += weed_prob
            heat_cnt[y0:y1, x0:x1] += 1.0
            rows.append([x0, y0, x1, y1, CLASSES[top], f"{float(p[top]):.4f}", f"{weed_prob:.4f}"])

    heat = heat_sum / np.clip(heat_cnt, 1.0, None)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(image)
    hm = ax.imshow(heat, cmap="jet", alpha=args.alpha, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04, label="weed probability (1 - P[negative])")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output} | mean weed prob {heat.mean():.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x0", "y0", "x1", "y1", "top_class", "top_conf", "weed_prob"])
            writer.writerows(rows)
        print(f"Wrote {args.csv} ({len(rows)} tiles)")


if __name__ == "__main__":
    main()
