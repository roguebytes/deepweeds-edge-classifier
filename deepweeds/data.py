"""DeepWeeds dataset, transforms, and dataloaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Label index -> species (8 weed species + negative). Matches DeepWeeds labels.csv.
CLASSES = [
    "Chinee apple",    # 0
    "Lantana",         # 1
    "Parkinsonia",     # 2
    "Parthenium",      # 3
    "Prickly acacia",  # 4
    "Rubber vine",     # 5
    "Siam weed",       # 6
    "Snake weed",      # 7
    "Negative",        # 8
]
NUM_CLASSES = len(CLASSES)

# ImageNet stats — required to match torchvision pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.images_dir / row["Filename"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["Label"])


def build_transforms(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_splits(labels_csv, val_frac=0.15, test_frac=0.15, seed=42,
                train_csv=None, val_csv=None, test_csv=None):
    """Use explicit split CSVs if given, else a stratified random split of labels.csv."""
    if train_csv and val_csv and test_csv:
        return pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)

    df = pd.read_csv(labels_csv)
    train_df, temp_df = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df["Label"], random_state=seed,
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["Label"], random_state=seed,
    )
    return train_df, val_df, test_df


def compute_class_weights(train_df: pd.DataFrame, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights — DeepWeeds is dominated by the negative class."""
    counts = train_df["Label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(images_dir, labels_csv, img_size=224, batch_size=32, num_workers=4,
                      seed=42, pin_memory=False, val_frac=0.15, test_frac=0.15,
                      train_csv=None, val_csv=None, test_csv=None):
    train_df, val_df, test_df = make_splits(
        labels_csv, val_frac, test_frac, seed, train_csv, val_csv, test_csv,
    )
    datasets = {
        "train": DeepWeedsDataset(train_df, images_dir, build_transforms(img_size, train=True)),
        "val": DeepWeedsDataset(val_df, images_dir, build_transforms(img_size, train=False)),
        "test": DeepWeedsDataset(test_df, images_dir, build_transforms(img_size, train=False)),
    }
    loaders = {
        split: DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=pin_memory, drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }
    return loaders, (train_df, val_df, test_df)
