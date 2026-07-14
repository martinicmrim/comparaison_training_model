#!/usr/bin/env python3
"""
Train a ResNet50 baseline on JacksParo diagnosis labels and save per-image predictions.

Purpose:
  Naive supervised baseline for ODIN experiments.
  Supports binary or 3-class diagnosis, patient/group splits, optional K-fold, and
  prediction exports for downstream statistical analyses such as mixed-effects models.

Important:
  Unlike the previous quick baseline, this script does not select the best model on the
  test set. If --val_size > 0, the best checkpoint is selected on the validation set.
  If --val_size 0, the final epoch is used and evaluated once on the test set.

Example ImageNet multiclass 5-fold:
python train_resnet50_diagnosis_baseline_with_predictions.py \
  --data_dir ../../../../data2/jacksparo/batch4/ \
  --diagnosis_mode multiclass \
  --views "Photo FACE" \
  --n_splits 5 \
  --val_size 0.15 \
  --epochs 30 \
  --batch_size 32 \
  --pretrained imagenet \
  --output_dir outputs_resnet50_imagenet_diagnosis_multiclass_5fold

Example from scratch:
python train_resnet50_diagnosis_baseline_with_predictions.py \
  --data_dir ../../../../data2/jacksparo/batch4/ \
  --diagnosis_mode multiclass \
  --views "Photo FACE" \
  --n_splits 5 \
  --val_size 0.15 \
  --epochs 50 \
  --batch_size 32 \
  --pretrained none \
  --output_dir outputs_resnet50_scratch_diagnosis_multiclass_5fold
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEFAULT_H5 = "data_512_with_img_cropped.h5"
DEFAULT_H5_INDEX = "table_id_name_512_with_img_cropped.csv"
DEFAULT_CORRECTIONS = "correction_images.csv"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_drive_id(value: str) -> str:
    value = str(value)
    marker = "https://drive.google.com/open?id="
    if marker in value:
        return value.split(marker, 1)[1]
    return value


def load_h5_diagnosis_frame(
    annotations: pd.DataFrame,
    logs: pd.DataFrame,
    viewing_angles: list[str],
    diagnosis_mode: str,
) -> pd.DataFrame:
    selected = (
        annotations["Controle qualité Technique"].isin(["Conforme"])
        & annotations["Controle qualité Fonctionnel"].isin(["Conforme", "Défaut"])
        & annotations["Diagnostic patient ?"].isin(["Sain", "Gingivite", "Parodontite active ", "Parodontite active"])
    )
    df = annotations[selected].copy()
    df = df[~df["Statut données"].isin(["Refusé", "Attente"])]
    df = df[~df["codeDentiste"].isin([0])]

    rows = []
    for view in viewing_angles:
        if view not in df.columns:
            print(f"Warning: missing view column: {view}")
            continue
        keep_cols = [view, "Diagnostic patient ?", "Response Number"]
        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={view: "image", "Diagnostic patient ?": "diagnosis", "Response Number": "participant_id"})
        tmp["view"] = view
        rows.append(tmp)

    if not rows:
        raise ValueError("No valid view columns found.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["image", "participant_id"])

    ok_files = set(logs.loc[logs["status"].isin(["ok"]), "file"].astype(str).tolist())
    out = out[out["image"].astype(str).isin(ok_files)].copy()

    if diagnosis_mode == "binary":
        out["label"] = out["diagnosis"].map({
            "Sain": 0,
            "Gingivite": 0,
            "Parodontite active ": 1,
            "Parodontite active": 1,
        })
        out["label_name"] = out["label"].map({0: "non_parodontite_active", 1: "parodontite_active"})
    elif diagnosis_mode == "multiclass":
        out["label"] = out["diagnosis"].map({
            "Sain": 0,
            "Gingivite": 1,
            "Parodontite active ": 2,
            "Parodontite active": 2,
        })
        out["label_name"] = out["label"].map({0: "sain", 1: "gingivite", 2: "parodontite_active"})
    else:
        raise ValueError(f"Unknown diagnosis_mode: {diagnosis_mode}")

    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out["participant_id"] = out["participant_id"].astype(str)
    out["file_id"] = out["image"].map(parse_drive_id)
    return out.reset_index(drop=True)

def clean_scalar(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def load_csv_image_frame(
    csv_path: Path,
    id_col: str,
    label_col: str,
    group_col: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if id_col not in df.columns:
        raise ValueError(f"Missing id column {id_col!r}. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}. Available: {list(df.columns)}")
    if "image_path" not in df.columns:
        raise ValueError("CSV mode requires an image_path column.")

    out = df.copy()
    out[id_col] = out[id_col].map(clean_scalar)
    out[label_col] = out[label_col].map(clean_scalar)
    out["image_path"] = out["image_path"].map(clean_scalar)

    out = out[out["image_path"].map(lambda x: Path(x).exists())].copy()

    labels = sorted([x for x in out[label_col].unique().tolist() if clean_scalar(x) != ""])
    label_to_idx = {label: i for i, label in enumerate(labels)}

    out["label"] = out[label_col].map(label_to_idx).astype(int)
    out["label_name"] = out[label_col].map(lambda x: f"MGI{x}" if str(x).isdigit() else str(x))
    out["image"] = out[id_col]
    out["view"] = "csv"

    if group_col and group_col in out.columns:
        out["participant_id"] = out[group_col].map(clean_scalar)
    else:
        out["participant_id"] = out[id_col].map(clean_scalar)

    return out.reset_index(drop=True)

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
        return image, int(row["label"])


class CsvImageDataset(Dataset):
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

        return image, int(row["label"])

def build_transforms(image_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, test_tfms


def build_resnet50(num_classes: int, pretrained: str) -> nn.Module:
    if pretrained == "imagenet":
        weights = models.ResNet50_Weights.DEFAULT
    elif pretrained == "none":
        weights = None
    else:
        raise ValueError(pretrained)
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes)
    total = counts.sum()
    weights = []
    for c in counts:
        weights.append(total / (num_classes * c) if c > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def make_splits(
    df: pd.DataFrame,
    seed: int,
    n_splits: int,
    test_size: float,
    val_size: float,
    use_predefined_splits: bool = False,
    split_col: str = "split",
):
    if use_predefined_splits:
        if split_col not in df.columns:
            raise ValueError(f"Missing split column {split_col!r}. Available columns: {list(df.columns)}")

        split_values = df[split_col].astype(str).str.lower().str.strip()

        train_idx = df.index[split_values == "train"].to_numpy()
        val_idx = df.index[split_values.isin(["val", "valid", "validation"])].to_numpy()
        test_idx = df.index[split_values == "test"].to_numpy()

        if len(train_idx) == 0:
            raise ValueError("No train samples found.")
        if len(test_idx) == 0:
            raise ValueError("No test samples found.")

        yield 1, train_idx, val_idx, test_idx
        return

    groups = df["participant_id"].to_numpy()
    y = df["label"].to_numpy()
    indices = np.arange(len(df))

    if n_splits <= 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        iterator = splitter.split(indices, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(indices, y, groups)

    for fold, (train_val_idx, test_idx) in enumerate(iterator, start=1):
        if val_size and val_size > 0:
            train_val_groups = groups[train_val_idx]
            train_val_y = y[train_val_idx]
            val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + fold)
            rel_train_idx, rel_val_idx = next(
                val_splitter.split(train_val_idx, train_val_y, train_val_groups)
            )
            train_idx = train_val_idx[rel_train_idx]
            val_idx = train_val_idx[rel_val_idx]
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)

        yield fold, train_idx, val_idx, test_idx


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        pred = p.argmax(dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        probs.append(p.cpu().numpy())
    return np.asarray(y_true), np.asarray(y_pred), np.concatenate(probs, axis=0)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)

def evaluate(y_true, y_pred, probs, num_classes: int):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "ece": expected_calibration_error(y_true, probs),
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    for i in range(num_classes):
        out[f"confusion_true{i}_pred{i}"] = int(cm[i, i])
        out[f"f1_class_{i}"] = f1_score(y_true == i, y_pred == i, zero_division=0)
        out[f"recall_class_{i}"] = recall_score(y_true == i, y_pred == i, zero_division=0)
        out[f"precision_class_{i}"] = precision_score(y_true == i, y_pred == i, zero_division=0)
    if num_classes == 2:
        try:
            out["auc"] = roc_auc_score(y_true, probs[:, 1])
        except ValueError:
            out["auc"] = np.nan
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


def regenerate_predictions(args, device, test_tfms, num_classes, task_name):
    strategy = f"resnet50_{args.pretrained}_full_finetune"

    prediction_files = []

    for fold in range(1, args.n_splits + 1):

        print(f"\nRegenerating predictions for fold {fold}")

        test_df = pd.read_csv(
            args.output_dir / "splits" / f"fold{fold}_test.csv"
        )

        if args.input_mode == "h5":
            test_ds = JacksParoH5Dataset(
                test_df,
                args.data_dir / args.h5,
                args.data_dir / args.h5_index,
                args.data_dir / args.corrections,
                test_tfms,
            )
        else:
            test_ds = CsvImageDataset(test_df, test_tfms)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = build_resnet50(
            num_classes=num_classes,
            pretrained=args.pretrained,
        ).to(device)

        checkpoint = (
            args.output_dir
            / "checkpoints"
            / f"resnet50_{args.pretrained}_fold{fold}.pt"
        )

        if not checkpoint.exists():
            print(f"Warning: missing checkpoint, skipping fold {fold}: {checkpoint}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        checkpoint_data = torch.load(checkpoint, map_location=device)
        if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
            checkpoint_data = checkpoint_data["state_dict"]

        model.load_state_dict(checkpoint_data)
        model.eval()

        y_true, y_pred, probs = predict(
            model,
            test_loader,
            device,
        )

        pred_file = (
            args.output_dir
            / "predictions"
            / f"fold{fold}_{args.pretrained}_test_predictions.csv"
        )

        save_predictions(
            test_df,
            y_true,
            y_pred,
            probs,
            pred_file,
            fold=fold,
            strategy=strategy,
            task_name=task_name,
            split_name="test",
        )

        prediction_files.append(pred_file)

    prediction_files = sorted(prediction_files)

    if not prediction_files:
        raise FileNotFoundError(
            f"No predictions were generated for pretrained={args.pretrained!r}. "
            "Check checkpoint and split paths."
        )

    all_predictions = pd.concat(
        [pd.read_csv(f) for f in prediction_files],
        ignore_index=True,
    )

    out_file = (
        args.output_dir
        / f"predictions_all_folds_{args.pretrained}.csv"
    )

    all_predictions.to_csv(out_file, index=False)

    print(f"\nSaved {out_file}")

def save_predictions(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    fold: int,
    strategy: str,
    task_name: str,
    split_name: str,
) -> None:
    out = frame.reset_index(drop=True).copy()
    if len(out) != len(y_true):
        raise ValueError(f"Prediction length mismatch: frame={len(out)} predictions={len(y_true)}")

    label_map = out[["label", "label_name"]].drop_duplicates().set_index("label")["label_name"].to_dict()
    out["fold"] = fold
    out["split"] = split_name
    out["strategy"] = strategy
    out["task"] = task_name
    out["y_true"] = y_true.astype(int)
    out["y_pred"] = y_pred.astype(int)
    out["true_label_name"] = [label_map.get(int(v), str(v)) for v in y_true]
    out["pred_label_name"] = [label_map.get(int(v), str(v)) for v in y_pred]
    out["correct"] = (y_true == y_pred).astype(int)
    out["pred_confidence"] = probs.max(axis=1)
    out["prob_true_class"] = [probs[i, int(y_true[i])] for i in range(len(y_true))]
    for c in range(probs.shape[1]):
        out[f"prob_class_{c}"] = probs[:, c]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("../../../../data2/jacksparo/batch4/"))
    parser.add_argument("--annotations", default="annotations_modified.csv")
    parser.add_argument("--logs", default="logs.csv")
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--h5_index", default=DEFAULT_H5_INDEX)
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS)
    parser.add_argument("--views", nargs="+", default=["Photo FACE"])
    parser.add_argument("--diagnosis_mode", default="multiclass", choices=["binary", "multiclass"])
    parser.add_argument("--pretrained", default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_resnet50_diagnosis"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.0, help="Validation fraction within train_val. Use 0 to train all non-test data and select final epoch.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--input_mode", default="h5", choices=["h5", "csv"])

    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--id_col", default="image_name")
    parser.add_argument("--label_col", default="global_mgi")
    parser.add_argument("--group_col", default="participant_id")

    parser.add_argument("--use_predefined_splits", action="store_true")
    parser.add_argument("--split_col", default="split")

    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Reload existing checkpoints and regenerate predictions only.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits").mkdir(exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.input_mode == "h5":
        annotations = pd.read_csv(args.data_dir / args.annotations, sep=",")
        logs = pd.read_csv(args.data_dir / args.logs, sep=";")
        df = load_h5_diagnosis_frame(annotations, logs, args.views, args.diagnosis_mode)
        task_name = f"diagnosis_{args.diagnosis_mode}"
    else:
        if args.csv is None:
            raise ValueError("--csv is required when --input_mode csv")
        df = load_csv_image_frame(
            csv_path=args.csv,
            id_col=args.id_col,
            label_col=args.label_col,
            group_col=args.group_col,
        )
        task_name = f"{args.label_col}_multiclass"

    num_classes = int(df["label"].nunique())
    strategy = f"resnet50_{args.pretrained}_full_finetune"

    print("Task:", task_name)
    print("Strategy:", strategy)
    print("Dataset size:", len(df))
    print("Groups:", df["participant_id"].nunique())
    print("Class distribution:")
    print(df["label"].value_counts().sort_index())
    print("Label mapping:")
    print(df[["label", "label_name"]].drop_duplicates().sort_values("label").to_string(index=False))
    df.to_csv(args.output_dir / "selected_data.csv", index=False)

    train_tfms, test_tfms = build_transforms(args.image_size)
    all_results = []

    if args.predict_only:
        regenerate_predictions(
            args=args,
            device=device,
            test_tfms=test_tfms,
            num_classes=num_classes,
            task_name=task_name,
        )

        return

    for fold, train_idx, val_idx, test_idx in make_splits(
        df,
        args.seed,
        args.n_splits,
        args.test_size,
        args.val_size,
        use_predefined_splits=args.use_predefined_splits,
        split_col=args.split_col,
    ):
        print(f"\n=== Fold {fold} ===")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True) if len(val_idx) else pd.DataFrame(columns=df.columns)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        print("Train/val/test:", len(train_df), len(val_df), len(test_df))
        print("Train distribution:")
        print(train_df["label"].value_counts().sort_index())
        if len(val_df):
            print("Val distribution:")
            print(val_df["label"].value_counts().sort_index())
        print("Test distribution:")
        print(test_df["label"].value_counts().sort_index())

        train_df.to_csv(args.output_dir / "splits" / f"fold{fold}_train.csv", index=False)
        if len(val_df):
            val_df.to_csv(args.output_dir / "splits" / f"fold{fold}_val.csv", index=False)
        test_df.to_csv(args.output_dir / "splits" / f"fold{fold}_test.csv", index=False)

        if args.input_mode == "h5":
            train_ds = JacksParoH5Dataset(train_df, args.data_dir / args.h5, args.data_dir / args.h5_index, args.data_dir / args.corrections, train_tfms)
            test_ds = JacksParoH5Dataset(test_df, args.data_dir / args.h5, args.data_dir / args.h5_index, args.data_dir / args.corrections, test_tfms)
        else:
            train_ds = CsvImageDataset(train_df, train_tfms)
            test_ds = CsvImageDataset(test_df, test_tfms)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        val_loader = None
        if len(val_df):
            if args.input_mode == "h5":
                val_ds = JacksParoH5Dataset(val_df, args.data_dir / args.h5, args.data_dir / args.h5_index, args.data_dir / args.corrections, test_tfms)
            else:
                val_ds = CsvImageDataset(val_df, test_tfms)

            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = build_resnet50(num_classes, args.pretrained).to(device)
        weights = compute_class_weights(train_df["label"].to_numpy(), num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = copy.deepcopy(model.state_dict())
        best_val_f1 = -1.0
        history = []
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            row = {"fold": fold, "epoch": epoch, "train_loss": loss}
            if val_loader is not None:
                y_val, pred_val, prob_val = predict(model, val_loader, device)
                val_metrics = evaluate(y_val, pred_val, prob_val, num_classes)
                row.update({f"val_{k}": v for k, v in val_metrics.items()})
                print(f"Fold {fold} | Epoch {epoch:03d}/{args.epochs} | loss={loss:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f} | val_bal_acc={val_metrics['balanced_accuracy']:.4f}")
                if val_metrics["f1_macro"] > best_val_f1:
                    best_val_f1 = val_metrics["f1_macro"]
                    best_state = copy.deepcopy(model.state_dict())
            else:
                best_state = copy.deepcopy(model.state_dict())
                print(f"Fold {fold} | Epoch {epoch:03d}/{args.epochs} | loss={loss:.4f}")
            history.append(row)

        pd.DataFrame(history).to_csv(args.output_dir / f"history_fold{fold}.csv", index=False)
        model.load_state_dict(best_state)
        train_time = time.time() - t0
        torch.save(model.state_dict(), args.output_dir / "checkpoints" / f"resnet50_{args.pretrained}_fold{fold}.pt")

        y_test, pred_test, prob_test = predict(model, test_loader, device)
        test_metrics = evaluate(y_test, pred_test, prob_test, num_classes)
        save_predictions(
            test_df,
            y_test,
            pred_test,
            prob_test,
            args.output_dir / "predictions" / f"fold{fold}_{args.pretrained}_test_predictions.csv",
            fold=fold,
            strategy=strategy,
            task_name=task_name,
            split_name="test",
        )

        row = {
            "fold": fold,
            "strategy": strategy,
            "task": task_name,
            "views": "+".join(args.views),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_classes": num_classes,
            "epochs": args.epochs,
            "selected_by": "val_f1_macro" if val_loader is not None else "last_epoch",
            "best_val_f1_macro": best_val_f1 if val_loader is not None else np.nan,
            "train_time_seconds": train_time,
        }
        row.update(test_metrics)
        all_results.append(row)
        print("Final fold test result:", row)

    results = pd.DataFrame(all_results)
    results.to_csv(args.output_dir / "results_resnet50.csv", index=False)

    prediction_files = sorted(
        (args.output_dir / "predictions").glob(
            f"fold*_{args.pretrained}_test_predictions.csv"
        )
    )
    predictions_output = (
        args.output_dir / f"predictions_all_folds_{args.pretrained}.csv"
    )
    if prediction_files:
        all_predictions = pd.concat(
            [pd.read_csv(p) for p in prediction_files],
            ignore_index=True,
        )
        all_predictions.to_csv(predictions_output, index=False)

    print("\nSaved results to:", args.output_dir / "results_resnet50.csv")
    print("Saved predictions to:", predictions_output)
    print(results)


if __name__ == "__main__":
    main()
