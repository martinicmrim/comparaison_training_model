#!/usr/bin/env python3
"""
Evaluate the MAE smoke-test encoder with the same linear-probe protocol as DINOv2.

This version supports both split formats:
  1) H5/JacksParo splits with file_id + H5 index.
  2) CSV/image-folder splits with image_path columns, as produced by
     run_jacksparo_dino_probe_clinical.py in --input_mode csv.

For CSV mode, the split files must contain:
  image_path,label
and preferably image,participant_id columns.

Example for clinical signs CSV splits:
  python run_mae_checkpoint_probe_from_splits_flexible.py \
    --input_mode csv \
    --split_dir outputs_dino_inflammation_severe/splits \
    --checkpoint ssl_smoke_mae/last_checkpoint.pt \
    --output_dir outputs_mae_inflammation_severe \
    --budgets 10 15 25 35 50 75 100 150 250 500 full

Example for H5 diagnosis splits:
  python run_mae_checkpoint_probe_from_splits_flexible.py \
    --input_mode h5 \
    --data_dir ../../../../data2/jacksparo/batch4/ \
    --split_dir outputs_dino_diagnosis_multiclass/splits \
    --checkpoint ssl_smoke_mae/last_checkpoint.pt \
    --output_dir outputs_mae_probe_diagnosis_multiclass \
    --budgets 25 50 100 250 500 full
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_H5 = "data_512_with_img_cropped.h5"
DEFAULT_H5_INDEX = "table_id_name_512_with_img_cropped.csv"
DEFAULT_CORRECTIONS = "correction_images.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class JacksParoH5Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, h5_path: Path, h5_index_path: Path, corrections_path: Path, transform=None) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        idx_df = pd.read_csv(h5_index_path)
        self.index_by_name = dict(zip(idx_df["name_file"].astype(str), idx_df["index_h5"].astype(int)))
        if corrections_path.exists():
            corr = pd.read_csv(corrections_path, sep=";")
            self.rotations = dict(zip(corr["name_file"].astype(str), corr["nb_rotation"].astype(int)))
        else:
            self.rotations = {}
        self.h5_path = h5_path
        self.h5_file = None
        self.h5_img = None

    def _ensure_h5_open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.h5_img = self.h5_file["img_cropped"]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        self._ensure_h5_open()
        row = self.frame.iloc[idx]
        file_id = str(row["file_id"])
        if file_id not in self.index_by_name:
            raise KeyError(f"Image {file_id} not found in H5 index")
        h5_idx = self.index_by_name[file_id]
        image = self.h5_img[h5_idx]
        if file_id in self.rotations:
            image = np.rot90(image, int(self.rotations[file_id])).copy()
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(row["label"])}


class CsvImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        if "image_path" not in self.frame.columns:
            raise ValueError("CSV split must contain an image_path column. Re-run the DINO CSV script so splits include image_path.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = Path(str(row["image_path"]))
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(row["label"])}


def get_transform(image_size: int, normalize: bool) -> transforms.Compose:
    steps = [
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(steps)


def load_mae_encoder(checkpoint_path: Path, device: torch.device) -> SmallConvAutoEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SmallConvAutoEncoder().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model: SmallConvAutoEncoder, dataset: Dataset, device: torch.device, batch_size: int, num_workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    xs, ys = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        z = model.encoder(images)
        z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        xs.append(z.detach().cpu().float().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def sample_budget_indices(y: np.ndarray, budget: str, seed: int) -> np.ndarray:
    if budget == "full":
        return np.arange(len(y))
    n = int(budget)
    if n >= len(y):
        return np.arange(len(y))
    idx = np.arange(len(y))
    values, counts = np.unique(y, return_counts=True)
    if len(values) >= 2 and np.all(counts >= 2) and n >= len(values):
        _, sub_idx = train_test_split(idx, test_size=n, random_state=seed, stratify=y)
        return np.sort(sub_idx)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=n, replace=False))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    if probs.ndim == 1:
        conf = probs
        pred = (probs >= 0.5).astype(int)
    else:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int):
    all_classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = int(all_classes.max()) + 1
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        raise ValueError("Training subset contains fewer than 2 classes. Increase budget or change split.")

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(x_train, y_train)
    prob_raw = clf.predict_proba(x_test)

    probs = np.zeros((len(y_test), n_classes), dtype=float)
    for j, cls in enumerate(clf.classes_):
        probs[:, int(cls)] = prob_raw[:, j]
    pred = probs.argmax(axis=1)

    out = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "ece": expected_calibration_error(y_test, probs),
    }
    for c in range(n_classes):
        out[f"f1_class_{int(c)}"] = f1_score(y_test == c, pred == c, zero_division=0)
        out[f"recall_class_{int(c)}"] = recall_score(y_test == c, pred == c, zero_division=0)
        out[f"precision_class_{int(c)}"] = precision_score(y_test == c, pred == c, zero_division=0)

    try:
        if n_classes == 2:
            prob_pos = probs[:, 1]
            out["auc"] = roc_auc_score(y_test, prob_pos)
            out["average_precision"] = average_precision_score((y_test == 1).astype(int), prob_pos)
        else:
            out["auc_ovr_macro"] = roc_auc_score(y_test, probs, multi_class="ovr", average="macro", labels=list(range(n_classes)))
            out["auc_ovo_macro"] = roc_auc_score(y_test, probs, multi_class="ovo", average="macro", labels=list(range(n_classes)))
    except ValueError:
        if n_classes == 2:
            out["auc"] = np.nan
            out["average_precision"] = np.nan
        else:
            out["auc_ovr_macro"] = np.nan
            out["auc_ovo_macro"] = np.nan
    return out, pred, probs


def save_probe_predictions(test_df: pd.DataFrame, y_test: np.ndarray, pred: np.ndarray, probs: np.ndarray, metadata: dict, out_path: Path) -> pd.DataFrame:
    out = test_df.reset_index(drop=True).copy()
    out["y_true"] = y_test.astype(int)
    out["y_pred"] = pred.astype(int)
    out["correct"] = (out["y_true"].to_numpy() == out["y_pred"].to_numpy()).astype(int)
    for c in range(probs.shape[1]):
        out[f"prob_{c}"] = probs[:, c]
    for key, value in metadata.items():
        out[key] = value
    preferred = [
        "task", "strategy", "model", "checkpoint", "fold", "budget", "budget_seed", "n_train",
        "participant_id", "image", "image_name", "view", "label_name",
        "y_true", "y_pred", "correct",
    ]
    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    remaining = [c for c in out.columns if c not in preferred + prob_cols]
    ordered = [c for c in preferred if c in out.columns] + prob_cols + remaining
    out = out[ordered]
    out.to_csv(out_path, index=False)
    return out


def find_folds(split_dir: Path) -> list[int]:
    folds = []
    for p in split_dir.glob("fold*_train.csv"):
        m = re.match(r"fold(\d+)_train\.csv", p.name)
        if m:
            folds.append(int(m.group(1)))
    return sorted(folds)


def build_dataset(input_mode: str, df: pd.DataFrame, args, transform):
    if input_mode == "csv":
        return CsvImageDataset(df, transform)
    if input_mode == "h5":
        if args.data_dir is None:
            raise ValueError("--data_dir is required for --input_mode h5")
        return JacksParoH5Dataset(
            df,
            args.data_dir / args.h5,
            args.data_dir / args.h5_index,
            args.data_dir / args.corrections,
            transform,
        )
    raise ValueError(f"Unknown input mode: {input_mode}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", default="h5", choices=["h5", "csv"])
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--h5_index", default=DEFAULT_H5_INDEX)
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS)
    parser.add_argument("--budgets", nargs="+", default=["25", "50", "100", "250", "500", "full"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--normalize", action="store_true", help="Apply ImageNet normalization before MAE encoder. Default is no normalization, matching MAE smoke training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embeddings").mkdir(exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Input mode:", args.input_mode)

    model = load_mae_encoder(args.checkpoint, device)
    transform = get_transform(args.image_size, normalize=args.normalize)
    folds = find_folds(args.split_dir)
    if not folds:
        raise ValueError(f"No fold*_train.csv files found in {args.split_dir}")
    print("Folds:", folds)

    all_results = []
    all_predictions = []
    for fold in folds:
        print(f"\n=== Fold {fold} ===")
        train_df = pd.read_csv(args.split_dir / f"fold{fold}_train.csv")
        test_df = pd.read_csv(args.split_dir / f"fold{fold}_test.csv")
        print("Train/test:", len(train_df), len(test_df))
        print("Train distribution:")
        print(train_df["label"].value_counts().sort_index())
        print("Test distribution:")
        print(test_df["label"].value_counts().sort_index())

        t0 = time.time()
        train_ds = build_dataset(args.input_mode, train_df, args, transform)
        test_ds = build_dataset(args.input_mode, test_df, args, transform)
        x_train_all, y_train_all = extract_embeddings(model, train_ds, device, args.batch_size, args.num_workers)
        x_test, y_test = extract_embeddings(model, test_ds, device, args.batch_size, args.num_workers)
        extract_time = time.time() - t0

        np.savez_compressed(
            args.output_dir / "embeddings" / f"fold{fold}_mae_encoder.npz",
            x_train=x_train_all,
            y_train=y_train_all,
            x_test=x_test,
            y_test=y_test,
        )

        n_classes = len(np.unique(np.concatenate([y_train_all, y_test])))
        for budget in args.budgets:
            sub_idx = sample_budget_indices(y_train_all, budget, seed=args.seed + fold)
            metrics_out, pred, probs = evaluate_probe(x_train_all[sub_idx], y_train_all[sub_idx], x_test, y_test, seed=args.seed)
            row = {
                "fold": fold,
                "strategy": "mae_ssl_encoder_linear_probe",
                "checkpoint": str(args.checkpoint),
                "input_mode": args.input_mode,
                "budget": budget,
                "n_train": int(len(sub_idx)),
                "n_test": int(len(y_test)),
                "n_classes": int(n_classes),
                "embedding_dim": int(x_train_all.shape[1]),
                "extract_time_seconds": extract_time,
                "trainable_params_probe": int(x_train_all.shape[1] * n_classes + n_classes),
            }
            row.update(metrics_out)
            all_results.append(row)
            pred_meta = {
                "task": "from_splits",
                "strategy": "mae_ssl_encoder_linear_probe",
                "model": "mae_encoder",
                "checkpoint": str(args.checkpoint),
                "fold": fold,
                "budget": budget,
                "budget_seed": args.seed + fold,
                "n_train": int(len(sub_idx)),
                "input_mode": args.input_mode,
            }
            pred_path = args.output_dir / "predictions" / f"fold{fold}_budget{budget}_predictions.csv"
            pred_df = save_probe_predictions(test_df, y_test, pred, probs, pred_meta, pred_path)
            all_predictions.append(pred_df)
            print(row)

    results = pd.DataFrame(all_results)
    out_path = args.output_dir / "results_mae_linear_probe.csv"
    results.to_csv(out_path, index=False)
    if all_predictions:
        predictions_all = pd.concat(all_predictions, ignore_index=True)
        predictions_path = args.output_dir / "predictions_all_folds.csv"
        predictions_all.to_csv(predictions_path, index=False)
        print("Saved predictions to:", predictions_path)
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    print("\nSaved results to:", out_path)
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
