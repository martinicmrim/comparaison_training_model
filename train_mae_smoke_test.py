#!/usr/bin/env python3
"""
Minimal MAE-like SSL smoke test on a CSV image manifest.

This is intentionally simple and low-dependency. It trains a small convolutional
autoencoder on randomly masked images, only to verify that the SSL pipeline works.
It is NOT meant to be a state-of-the-art dental foundation model.

Input manifest must contain a column named `path`.

Dependencies:
  pip install torch torchvision pandas pillow tqdm

Example:
  python train_mae_smoke_test.py \
    --manifest ssl_manifest_10k.csv \
    --out-dir ssl_smoke_mae \
    --epochs 5 \
    --batch-size 64 \
    --image-size 224
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ManifestImageDataset(Dataset):
    def __init__(self, manifest: Path, transform=None, max_retry: int = 20):
        self.df = pd.read_csv(manifest)
        if "path" not in self.df.columns:
            raise ValueError("Manifest must contain a `path` column.")
        self.paths = self.df["path"].astype(str).tolist()
        self.transform = transform
        self.max_retry = max_retry
        self.bad_paths = set()
        self.failed_reads = 0

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: str):
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, idx: int):
        n = len(self.paths)

        for offset in range(self.max_retry):
            j = (idx + offset) % n
            path = self.paths[j]

            if path in self.bad_paths:
                continue

            try:
                img = self._load_image(path)
                if self.transform is not None:
                    img = self.transform(img)
                return img

            except Exception as e:
                self.bad_paths.add(path)
                self.failed_reads += 1
                if self.failed_reads <= 20:
                    print(f"[WARNING] Could not read image: {path} ({e})")
                continue

        raise RuntimeError(
            f"Could not read a valid image after {self.max_retry} retries "
            f"starting from index {idx}."
        )

class SmallConvAutoEncoder(nn.Module):
    def __init__(self, latent_channels: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 4, 2, 1), nn.BatchNorm2d(latent_channels), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def random_block_mask(x: torch.Tensor, mask_ratio: float, patch_size: int) -> torch.Tensor:
    if mask_ratio <= 0:
        return x
    b, c, h, w = x.shape
    gh = h // patch_size
    gw = w // patch_size
    mask = torch.rand((b, 1, gh, gw), device=x.device) < mask_ratio
    mask = F.interpolate(mask.float(), size=(h, w), mode="nearest")
    return x * (1.0 - mask)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.50)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
    ])
    ds = ManifestImageDataset(args.manifest, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print("Training images:", len(ds))

    model = SmallConvAutoEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images in pbar:
            images = images.to(device, non_blocking=True)
            masked = random_block_mask(images, args.mask_ratio, args.patch_size)
            recon = model(masked)
            loss = F.mse_loss(recon, images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * images.size(0)
            total_n += images.size(0)
            pbar.set_postfix(loss=loss.item())
        epoch_loss = total_loss / max(total_n, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch}: loss={epoch_loss:.6f}")
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, args.out_dir / "last_checkpoint.pt")

    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    print("Saved checkpoint to:", args.out_dir / "last_checkpoint.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
