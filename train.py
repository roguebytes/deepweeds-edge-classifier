"""Train a weed classifier on DeepWeeds and report test metrics.

Example:
    python train.py --data-dir data/images --labels-csv data/labels.csv \
        --arch resnet50 --epochs 15 --batch-size 32 --output-dir runs/resnet50
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from deepweeds.data import CLASSES, NUM_CLASSES, build_dataloaders, compute_class_weights
from deepweeds.engine import evaluate, train_one_epoch
from deepweeds.model import SUPPORTED_ARCHS, build_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepWeeds classifier")
    p.add_argument("--data-dir", required=True, help="Directory of images")
    p.add_argument("--labels-csv", required=True, help="labels.csv (Filename, Label, Species)")
    p.add_argument("--train-csv", default=None, help="Optional explicit split CSVs")
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--class-weights", action="store_true", help="Weight loss by inverse class freq")
    p.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet weights (offline/smoke test)")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="runs/exp")
    return p.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion(cm, classes, out_path):
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    use_amp = (device.type == "cuda") and not args.no_amp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | AMP: {use_amp} | arch: {args.arch}")

    loaders, (train_df, _, _) = build_dataloaders(
        images_dir=args.data_dir, labels_csv=args.labels_csv, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        pin_memory=(device.type == "cuda"),
        train_csv=args.train_csv, val_csv=args.val_csv, test_csv=args.test_csv,
    )

    model = build_model(args.arch, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)

    weight = compute_class_weights(train_df).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    best_acc, best_path = 0.0, out_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, use_amp,
        )
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()
        print(f"epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "img_size": args.img_size, "classes": CLASSES}, best_path)

    # Final evaluation on the held-out test set using the best checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_loss, test_acc, preds, targets = evaluate(model, loaders["test"], criterion, device)
    labels = list(range(NUM_CLASSES))
    report = classification_report(targets, preds, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    plot_confusion(confusion_matrix(targets, preds, labels=labels), CLASSES,
                   out_dir / "confusion_matrix.png")

    metrics = {
        "arch": args.arch,
        "params": count_parameters(model),
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "per_class": report,
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nTest accuracy: {test_acc:.4f} | params: {metrics['params']:,}")
    print(f"Saved best model, metrics.json, confusion_matrix.png to {out_dir}/")


if __name__ == "__main__":
    main()
