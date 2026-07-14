#!/usr/bin/env python3
"""
train_dinov2_partial_finetune.py

Supervised partial fine-tuning of DINOv2 on pre-defined splits.

Supported input modes:
  - h5: JacksParo H5 dataset, using split CSVs produced by run_jacksparo_dino_probe_*.py
  - csv: flat image dataset, using split CSVs produced by run_jacksparo_dino_probe_*.py

Fine-tuning modes:
  - head: freeze DINOv2, train only classifier head
  - last_block: train last transformer block + classifier head
  - last_2_blocks: train last 2 transformer blocks + classifier head
  - last_4_blocks: train last 4 transformer blocks + classifier head
  - full: train entire DINOv2 + classifier head
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_budget_frame(df: pd.DataFrame, budget: str, seed: int) -> pd.DataFrame:
    if budget == "full":
        return df.reset_index(drop=True)
    n = int(budget)
    if n >= len(df):
        return df.reset_index(drop=True)
    y = df["label"].to_numpy()
    idx = np.arange(len(df))
    values, counts = np.unique(y, return_counts=True)
    if len(values) >= 2 and np.all(counts >= 2) and n >= len(values):
        _, sub_idx = train_test_split(idx, test_size=n, random_state=seed, stratify=y)
    else:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(idx, size=n, replace=False)
    return df.iloc[np.sort(sub_idx)].reset_index(drop=True)


class JacksParoH5Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, h5_path: Path, h5_index_path: Path, corrections_path: Path, transform=None) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.h5_path = h5_path
        self.transform = transform
        idx_df = pd.read_csv(h5_index_path)
        self.index_by_name = dict(zip(idx_df["name_file"].astype(str), idx_df["index_h5"].astype(int)))
        if corrections_path.exists():
            corr = pd.read_csv(corrections_path, sep=";")
            self.rotations = dict(zip(corr["name_file"].astype(str), corr["nb_rotation"].astype(int)))
        else:
            self.rotations = {}
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
        if "file_id" in row and pd.notna(row["file_id"]):
            file_id = str(row["file_id"])
        else:
            image_value = str(row["image"])
            marker = "https://drive.google.com/open?id="
            file_id = image_value.split(marker, 1)[1] if marker in image_value else image_value
        if file_id not in self.index_by_name:
            raise KeyError(f"Image {file_id} not found in H5 index")
        image = self.h5_img[self.index_by_name[file_id]]
        if file_id in self.rotations:
            image = np.rot90(image, int(self.rotations[file_id])).copy()
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(row["label"]), "image_name": str(row.get("image", "")), "participant_id": str(row.get("participant_id", ""))}


class FlatImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(row["label"]), "image_name": str(row.get("image", row.get("image_path", ""))), "participant_id": str(row.get("participant_id", ""))}


def build_transforms(image_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, test_tfms


def build_dataset(input_mode: str, frame: pd.DataFrame, args, transform):
    if input_mode == "h5":
        return JacksParoH5Dataset(frame, args.data_dir / args.h5, args.data_dir / args.h5_index, args.data_dir / args.corrections, transform)
    if input_mode == "csv":
        return FlatImageDataset(frame, transform)
    raise ValueError(input_mode)


class DinoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return self.classifier(features)


def load_dino_backbone(model_name: str, device: torch.device) -> tuple[nn.Module, int]:
    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    backbone.to(device)
    embed_dims = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
    return backbone, embed_dims[model_name]


def set_finetune_mode(model: DinoClassifier, mode: str) -> int:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    blocks = getattr(model.backbone, "blocks", None)
    if mode == "head":
        pass
    elif mode == "last_block":
        for p in blocks[-1].parameters():
            p.requires_grad = True
    elif mode == "last_2_blocks":
        for block in blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
    elif mode == "last_4_blocks":
        for block in blocks[-4:]:
            for p in block.parameters():
                p.requires_grad = True
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(mode)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes)
    total = counts.sum()
    return torch.tensor([total / (num_classes * c) if c > 0 else 0.0 for c in counts], dtype=torch.float32)


def expected_calibration_error_multiclass(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def expected_calibration_error_binary(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_pos > lo) & (prob_pos <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(y_true[mask].mean() - prob_pos[mask].mean())
    return float(ece)


def evaluate_arrays(y_true: np.ndarray, probs: np.ndarray) -> dict:
    n_classes = probs.shape[1]
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "ece": expected_calibration_error_multiclass(y_true, probs),
    }
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    for i in range(n_classes):
        out[f"confusion_true{i}_pred{i}"] = int(cm[i, i])
        out[f"f1_class_{i}"] = f1_score(y_true == i, pred == i, zero_division=0)
        out[f"recall_class_{i}"] = recall_score(y_true == i, pred == i, zero_division=0)
        out[f"precision_class_{i}"] = precision_score(y_true == i, pred == i, zero_division=0)
    if n_classes == 2:
        prob_pos = probs[:, 1]
        pred_bin = (prob_pos >= 0.5).astype(int)
        out.update({
            "f1_positive": f1_score(y_true, pred_bin, pos_label=1, zero_division=0),
            "precision_positive": precision_score(y_true, pred_bin, pos_label=1, zero_division=0),
            "recall_positive": recall_score(y_true, pred_bin, pos_label=1, zero_division=0),
            "ece_binary": expected_calibration_error_binary(y_true, prob_pos),
        })
        try:
            out["auc"] = roc_auc_score(y_true, prob_pos)
        except ValueError:
            out["auc"] = np.nan
        try:
            out["average_precision"] = average_precision_score(y_true, prob_pos)
        except ValueError:
            out["average_precision"] = np.nan
    else:
        try:
            out["auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        except ValueError:
            out["auc_ovr_macro"] = np.nan
        try:
            out["auc_ovo_macro"] = roc_auc_score(y_true, probs, multi_class="ovo", average="macro")
        except ValueError:
            out["auc_ovo_macro"] = np.nan
    return out


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 0.0) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, probs, image_names, participant_ids = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        p = torch.softmax(model(images), dim=1)
        y_true.extend(batch["label"].cpu().numpy().tolist())
        probs.append(p.cpu().numpy())
        image_names.extend([str(x) for x in batch["image_name"]])
        participant_ids.extend([str(x) for x in batch["participant_id"]])
    return np.asarray(y_true, dtype=int), np.concatenate(probs, axis=0), image_names, participant_ids


def make_predictions_frame(y_true, probs, image_names, participant_ids, metadata):
    pred = probs.argmax(axis=1)
    out = pd.DataFrame({"participant_id": participant_ids, "image_name": image_names, "y_true": y_true, "y_pred": pred, "correct": (y_true == pred).astype(int)})
    for key, value in metadata.items():
        out[key] = value
    for c in range(probs.shape[1]):
        out[f"prob_{c}"] = probs[:, c]
    return out


def get_split_files(split_dir: Path):
    splits = []
    for train_path in sorted(split_dir.glob("fold*_train.csv")):
        fold = int(train_path.stem.replace("fold", "").replace("_train", ""))
        val_path = split_dir / f"fold{fold}_val.csv"
        test_path = split_dir / f"fold{fold}_test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test split for fold {fold}: {test_path}")
        splits.append({
            "fold": fold,
            "train": train_path,
            "val": val_path if val_path.exists() else None,
            "test": test_path,
        })
    if not splits:
        raise FileNotFoundError(f"No fold*_train.csv files found in {split_dir}")
    return splits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", choices=["h5", "csv"], required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("../../../../data2/jacksparo/batch4/"))
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--h5_index", default=DEFAULT_H5_INDEX)
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS)
    parser.add_argument("--model", default="dinov2_vitb14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    parser.add_argument("--finetune_modes", nargs="+", default=["head"], choices=["head", "last_block", "last_2_blocks", "last_4_blocks", "full"])
    parser.add_argument("--budgets", nargs="+", default=["25", "100", "500", "full"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_class_weights", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)
    (args.output_dir / "histories").mkdir(exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    split_files = get_split_files(args.split_dir)
    train_tfms, test_tfms = build_transforms(args.image_size)
    all_results, all_predictions = [], []
    for split_info in split_files:
        fold = split_info["fold"]
        base_train_df = pd.read_csv(split_info["train"])
        test_df = pd.read_csv(split_info["test"])
        val_df = pd.read_csv(split_info["val"]) if split_info["val"] is not None else None

        label_frames = [base_train_df["label"], test_df["label"]]
        if val_df is not None:
            label_frames.append(val_df["label"])
        n_classes = int(pd.concat(label_frames).astype(int).nunique())
        print(f"\n=== Fold {fold} | classes={n_classes} ===")
        for budget in args.budgets:
            train_df = sample_budget_frame(base_train_df, budget, seed=args.seed + fold)
            if train_df["label"].nunique() < 2:
                print(f"Skipping fold={fold}, budget={budget}: fewer than 2 classes.")
                continue
            train_ds = build_dataset(args.input_mode, train_df, args, train_tfms)
            test_ds = build_dataset(args.input_mode, test_df, args, test_tfms)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            val_loader = None
            if val_df is not None and len(val_df) > 0:
                val_ds = build_dataset(args.input_mode, val_df, args, test_tfms)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            for mode in args.finetune_modes:
                print(f"\nFold={fold} Budget={budget} Mode={mode} n_train={len(train_df)}")
                set_seed(args.seed + fold)
                backbone, embed_dim = load_dino_backbone(args.model, device)
                model = DinoClassifier(backbone, embed_dim, n_classes, dropout=args.dropout).to(device)
                trainable_params = set_finetune_mode(model, mode)
                head_params = [p for p in model.classifier.parameters() if p.requires_grad]
                backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": args.lr_backbone},
                    {"params": head_params, "lr": args.lr_head},
                ], weight_decay=args.weight_decay)
                criterion = nn.CrossEntropyLoss() if args.no_class_weights else nn.CrossEntropyLoss(weight=compute_class_weights(train_df["label"].to_numpy(), n_classes).to(device))
                best_score, best_state, best_epoch, bad_epochs = -math.inf, None, 0, 0
                history = []
                start = time.time()
                for epoch in range(1, args.epochs + 1):
                    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=args.grad_clip)
                    history_row = {
                        "fold": fold,
                        "budget": budget,
                        "finetune_mode": mode,
                        "epoch": epoch,
                        "train_loss": train_loss,
                    }

                    if val_loader is not None:
                        y_val, p_val, _, _ = predict(model, val_loader, device)
                        val_metrics = evaluate_arrays(y_val, p_val)
                        score = val_metrics["f1_macro"]
                        history_row.update({
                            "val_f1_macro": val_metrics["f1_macro"],
                            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                        })
                        print(f"epoch={epoch:03d} loss={train_loss:.4f} val_f1={score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_state = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
                            bad_epochs = 0
                        else:
                            bad_epochs += 1
                        history.append(history_row)
                        if bad_epochs >= args.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                    else:
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        history.append(history_row)
                        print(f"epoch={epoch:03d} loss={train_loss:.4f}")
                train_time = time.time() - start
                if best_state is not None:
                    model.load_state_dict(best_state)
                y_test, probs, image_names, participant_ids = predict(model, test_loader, device)
                metrics_out = evaluate_arrays(y_test, probs)
                metadata = {
                    "fold": fold,
                    "budget": str(budget),
                    "n_train": int(len(train_df)),
                    "n_val": int(len(val_df)) if val_df is not None else 0,
                    "n_test": int(len(test_df)),
                    "model": args.model,
                    "strategy": f"dinov2_partial_finetune_{mode}",
                    "finetune_mode": mode,
                    "best_epoch": int(best_epoch),
                    "selected_by": "validation_f1_macro" if val_loader is not None else "last_epoch",
                }
                pred_df = make_predictions_frame(y_test, probs, image_names, participant_ids, metadata)
                pred_path = args.output_dir / "predictions" / f"fold{fold}_budget{budget}_{mode}_predictions.csv"
                pred_df.to_csv(pred_path, index=False)
                all_predictions.append(pred_df)
                pd.DataFrame(history).to_csv(args.output_dir / "histories" / f"history_fold{fold}_budget{budget}_{mode}.csv", index=False)
                row = {
                    "fold": fold,
                    "budget": str(budget),
                    "n_train": int(len(train_df)),
                    "n_val": int(len(val_df)) if val_df is not None else 0,
                    "n_test": int(len(test_df)),
                    "model": args.model,
                    "strategy": f"dinov2_partial_finetune_{mode}",
                    "finetune_mode": mode,
                    "n_classes": n_classes,
                    "trainable_params": int(trainable_params),
                    "best_epoch": int(best_epoch),
                    "selected_by": "validation_f1_macro" if val_loader is not None else "last_epoch",
                    "best_val_f1_macro": float(best_score) if val_loader is not None else np.nan,
                    "train_time_seconds": float(train_time),
                }
                row.update(metrics_out)
                all_results.append(row)
                print(row)
                # torch.save({"state_dict": model.state_dict(), "args": vars(args), "fold": fold, "budget": str(budget), "finetune_mode": mode, "n_classes": n_classes}, args.output_dir / "checkpoints" / f"fold{fold}_budget{budget}_{mode}.pt")
                del model, backbone
                torch.cuda.empty_cache()
    results = pd.DataFrame(all_results)
    results.to_csv(args.output_dir / "results_finetune.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(args.output_dir / "predictions_all_folds.csv", index=False)
    print("\nSaved results to:", args.output_dir / "results_finetune.csv")
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
