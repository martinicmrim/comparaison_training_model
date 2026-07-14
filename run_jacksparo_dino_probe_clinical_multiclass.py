#!/usr/bin/env python3
"""
Minimal DINOv2 frozen-embedding + linear-probe experiments for JacksParo tasks.

Supports:
  - H5 mode with diagnosis labels either binary or multiclass.
  - H5 mode with clinical_signs/custom binary labels.
  - CSV mode with binary or multiclass labels.

Baseline: DINOv2 frozen embeddings + scikit-learn logistic regression.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
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
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_H5 = "data_512_with_img_cropped.h5"
DEFAULT_H5_INDEX = "table_id_name_512_with_img_cropped.csv"
DEFAULT_CORRECTIONS = "correction_images.csv"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_scalar(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def parse_drive_id(value: str) -> str:
    value = str(value)
    marker = "https://drive.google.com/open?id="
    if marker in value:
        return value.split(marker, 1)[1]
    return value


def map_binary_label(value: str, positive_values: list[str], negative_values: list[str]) -> Optional[int]:
    value = clean_scalar(value)
    pos = {clean_scalar(v) for v in positive_values}
    neg = {clean_scalar(v) for v in negative_values}
    if value in pos:
        return 1
    if value in neg:
        return 0
    return None


def get_view_sign_column(view: str) -> Optional[str]:
    mapping = {
        "Photo FACE": "Photo FACE - Signes cliniques visibles ?",
        "Photo DROITE\u00a0- Secteur 1 et 4": "Photo DROITE - Signes cliniques visibles ?",
        "Photo DROITE - Secteur 1 et 4": "Photo DROITE - Signes cliniques visibles ?",
        "Photo GAUCHE - Secteur 2 et 3": "Photo GAUCHE - Signes cliniques visibles ?",
    }
    return mapping.get(view)


def load_h5_annotation_frame(
    annotations: pd.DataFrame,
    logs: pd.DataFrame,
    viewing_angles: list[str],
    technical_quality: list[str],
    functional_quality: list[str],
    diagnostic: list[str],
    label_mode: str,
    diagnosis_mode: str,
    custom_label_col: Optional[str],
    positive_values: list[str],
    negative_values: list[str],
) -> pd.DataFrame:
    selected = (
        annotations["Controle qualité Technique"].isin(technical_quality)
        & annotations["Controle qualité Fonctionnel"].isin(functional_quality)
        & annotations["Diagnostic patient ?"].isin(diagnostic)
    )
    df = annotations[selected].copy()
    df = df[~df["Statut données"].isin(["Refusé", "Attente"])]
    df = df[~df["codeDentiste"].isin([0])]

    rows = []
    for view in viewing_angles:
        if view not in df.columns:
            print(f"Warning: missing view column: {view}")
            continue
        clinical_col = get_view_sign_column(view)
        keep_cols = [view, "Diagnostic patient ?", "Response Number"]
        if clinical_col and clinical_col in df.columns:
            keep_cols.append(clinical_col)
        if custom_label_col and custom_label_col not in keep_cols and custom_label_col in df.columns:
            keep_cols.append(custom_label_col)

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={view: "image", "Diagnostic patient ?": "diagnosis", "Response Number": "participant_id"})
        if clinical_col and clinical_col in tmp.columns:
            tmp = tmp.rename(columns={clinical_col: "clinical_signs"})
        else:
            tmp["clinical_signs"] = ""
        tmp["view"] = view
        rows.append(tmp)

    if not rows:
        raise ValueError("No valid view columns found.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["image", "participant_id"])

    ok_files = set(logs.loc[logs["status"].isin(["ok"]), "file"].astype(str).tolist())
    out = out[out["image"].astype(str).isin(ok_files)].copy()

    if label_mode == "diagnosis":
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
    elif label_mode == "clinical_signs":
        out["label"] = out["clinical_signs"].map(lambda x: map_binary_label(x, positive_values, negative_values))
        out["label_name"] = out["label"].map({0: "negative", 1: "positive"})
    elif label_mode == "custom":
        if not custom_label_col:
            raise ValueError("--custom_label_col is required when --label_mode custom")
        if custom_label_col not in out.columns:
            raise ValueError(f"Custom label column {custom_label_col!r} not found. Available: {list(out.columns)}")
        out["label"] = out[custom_label_col].map(lambda x: map_binary_label(x, positive_values, negative_values))
        out["label_name"] = out["label"].map({0: "negative", 1: "positive"})
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out["participant_id"] = out["participant_id"].astype(str)
    out["file_id"] = out["image"].map(parse_drive_id)
    return out.reset_index(drop=True)


def find_image_path(images_dir: Path, image_id: str) -> Optional[Path]:
    image_id = clean_scalar(image_id)
    if image_id == "":
        return None
    p = images_dir / image_id
    if p.exists():
        return p
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    stem = Path(image_id).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_csv_image_frame(
    csv_path: Path,
    images_dir: Path,
    id_col: str,
    label_col: str,
    group_col: Optional[str],
    task_type: str,
    positive_values: list[str],
    negative_values: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"Missing id column {id_col!r}. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}. Available: {list(df.columns)}")

    out = df.copy()
    out[id_col] = out[id_col].map(clean_scalar)
    out[label_col] = out[label_col].map(clean_scalar)

    if task_type == "binary":
        out["label"] = out[label_col].map(lambda x: map_binary_label(x, positive_values, negative_values))
        out["label_name"] = out["label"].map({0: "negative", 1: "positive"})
    elif task_type == "multiclass":
        labels = sorted([x for x in out[label_col].unique().tolist() if clean_scalar(x) != ""])
        label_to_idx = {label: i for i, label in enumerate(labels)}
        out["label"] = out[label_col].map(label_to_idx)
        out["label_name"] = out[label_col]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)

    # If the CSV already provides absolute image paths, use them.
    if "image_path" in out.columns:

        out["image_path"] = out["image_path"].map(clean_scalar)

        missing = (~out["image_path"].map(lambda x: Path(x).exists())).sum()

        out = out[out["image_path"].map(lambda x: Path(x).exists())].copy()

        if missing:
            print(f"Warning: ignored {missing} rows because image file was not found.")

    # Otherwise, search the images in images_dir (legacy behaviour).
    else:

        paths = []
        missing = 0

        for image_id in out[id_col].tolist():
            p = find_image_path(images_dir, image_id)

            if p is None:
                paths.append("")
                missing += 1
            else:
                paths.append(str(p))

        out["image_path"] = paths
        out = out[out["image_path"] != ""].copy()

        if missing:
            print(f"Warning: ignored {missing} rows because image file was not found.")

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
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(row["label"]), "image_name": str(row["image"]), "participant_id": str(row["participant_id"])}


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
        return {"image": image, "label": int(row["label"]), "image_name": str(row["image"]), "participant_id": str(row["participant_id"])}


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int, num_workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_features = []
    all_labels = []
    all_names = []
    all_participants = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = model(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        all_features.append(features.detach().cpu().float().numpy())
        all_labels.append(batch["label"].numpy())
        all_names.extend(batch["image_name"])
        all_participants.extend(batch["participant_id"])
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0), all_names, all_participants


def expected_calibration_error_binary(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    prob_pos = prob_pos.astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_pos > lo) & (prob_pos <= hi)
        if not np.any(mask):
            continue
        conf = prob_pos[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def expected_calibration_error_multiclass(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


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


def evaluate_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> dict[str, float]:
    classes = np.unique(y_train)
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    if len(classes) < 2:
        raise ValueError("Training subset contains fewer than 2 classes. Increase budget or change split.")

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(x_train, y_train)
    prob_raw = clf.predict_proba(x_test)

    # Align probability columns in case a small budget misses a class.
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
        "ece": expected_calibration_error_multiclass(y_test, probs),
    }

    cm = confusion_matrix(y_test, pred, labels=list(range(n_classes)))
    for i in range(n_classes):
        out[f"confusion_true{i}_pred{i}"] = int(cm[i, i])

    if n_classes == 2:
        prob_pos = probs[:, 1]
        pred_bin = (prob_pos >= 0.5).astype(int)
        out.update({
            "f1_positive": f1_score(y_test, pred_bin, pos_label=1, zero_division=0),
            "precision_positive": precision_score(y_test, pred_bin, pos_label=1, zero_division=0),
            "recall_positive": recall_score(y_test, pred_bin, pos_label=1, zero_division=0),
            "ece_binary": expected_calibration_error_binary(y_test, prob_pos),
        })
        try:
            out["auc"] = roc_auc_score(y_test, prob_pos)
        except ValueError:
            out["auc"] = np.nan
        try:
            out["average_precision"] = average_precision_score(y_test, prob_pos)
        except ValueError:
            out["average_precision"] = np.nan
    else:
        out.update({"f1_positive": np.nan, "precision_positive": np.nan, "recall_positive": np.nan, "ece_binary": np.nan})
        try:
            out["auc_ovr_macro"] = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        except ValueError:
            out["auc_ovr_macro"] = np.nan
        try:
            out["auc_ovo_macro"] = roc_auc_score(y_test, probs, multi_class="ovo", average="macro")
        except ValueError:
            out["auc_ovo_macro"] = np.nan
        for cls in range(n_classes):
            out[f"f1_class_{cls}"] = f1_score(y_test == cls, pred == cls, zero_division=0)
            out[f"recall_class_{cls}"] = recall_score(y_test == cls, pred == cls, zero_division=0)
            out[f"precision_class_{cls}"] = precision_score(y_test == cls, pred == cls, zero_division=0)

    return out


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
        train_val_groups = groups[train_val_idx]
        train_val_y = y[train_val_idx]

        if val_size is None or val_size <= 0:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)
        else:
            val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + fold)
            rel_train_idx, rel_val_idx = next(
                val_splitter.split(train_val_idx, train_val_y, train_val_groups)
            )
            train_idx = train_val_idx[rel_train_idx]
            val_idx = train_val_idx[rel_val_idx]

        yield fold, train_idx, val_idx, test_idx


def build_dataset(input_mode: str, frame: pd.DataFrame, args, transform):
    if input_mode == "h5":
        return JacksParoH5Dataset(frame, args.data_dir / args.h5, args.data_dir / args.h5_index, args.data_dir / args.corrections, transform)
    if input_mode == "csv":
        return FlatImageDataset(frame, transform)
    raise ValueError(input_mode)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", default="h5", choices=["h5", "csv"])

    parser.add_argument("--data_dir", type=Path, default=Path("../../../../data2/jacksparo/batch4/"))
    parser.add_argument("--annotations", default="annotations_modified.csv")
    parser.add_argument("--logs", default="logs.csv")
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--h5_index", default=DEFAULT_H5_INDEX)
    parser.add_argument("--corrections", default=DEFAULT_CORRECTIONS)
    parser.add_argument("--views", nargs="+", default=["Photo FACE"])
    parser.add_argument("--label_mode", default="clinical_signs", choices=["diagnosis", "clinical_signs", "custom"])
    parser.add_argument("--diagnosis_mode", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--custom_label_col", default=None)

    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--images_dir", type=Path, default=None)
    parser.add_argument("--id_col", default="Photo_name")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--task_type", default="binary", choices=["binary", "multiclass"])

    parser.add_argument("--positive_values", nargs="+", default=["Oui"])
    parser.add_argument("--negative_values", nargs="+", default=["Non"])

    parser.add_argument("--output_dir", type=Path, default=Path("outputs_dino_clinical"))
    parser.add_argument("--model", default="dinov2_vitb14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    parser.add_argument("--budgets", nargs="+", default=["25", "50", "100", "250", "500", "full"])
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--use_predefined_splits", action="store_true")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--image_path_col", default="image_path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits").mkdir(exist_ok=True)
    (args.output_dir / "embeddings").mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.input_mode == "h5":
        annotations = pd.read_csv(args.data_dir / args.annotations, sep=",")
        logs = pd.read_csv(args.data_dir / args.logs, sep=";")
        df = load_h5_annotation_frame(
            annotations=annotations,
            logs=logs,
            viewing_angles=args.views,
            technical_quality=["Conforme"],
            functional_quality=["Conforme", "Défaut"],
            diagnostic=["Sain", "Gingivite", "Parodontite active ", "Parodontite active"],
            label_mode=args.label_mode,
            diagnosis_mode=args.diagnosis_mode,
            custom_label_col=args.custom_label_col,
            positive_values=args.positive_values,
            negative_values=args.negative_values,
        )
        task_name = args.label_mode if args.label_mode != "custom" else args.custom_label_col
        if args.label_mode == "diagnosis":
            task_name = f"diagnosis_{args.diagnosis_mode}"
    else:
        if args.csv is None or args.images_dir is None or args.label_col is None:
            raise ValueError("CSV mode requires --csv, --images_dir, and --label_col")
        df = load_csv_image_frame(args.csv, args.images_dir, args.id_col, args.label_col, args.group_col, args.task_type, args.positive_values, args.negative_values)
        task_name = f"{args.label_col}_{args.task_type}"

    print("Task:", task_name)
    print("Dataset size:", len(df))
    print("Participants/groups:", df["participant_id"].nunique())
    print("Class distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())
    if "label_name" in df.columns:
        print("Label mapping:")
        print(df[["label", "label_name"]].drop_duplicates().sort_values("label").to_string(index=False))
    if len(df) == df["participant_id"].nunique():
        print("Note: one image per group in the selected data. This is expected for FACE-only datasets.")
    df.to_csv(args.output_dir / "selected_data.csv", index=False)

    model = load_dinov2(args.model, device)
    transform = get_transform()
    all_results = []

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
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        print("Train/val/test:", len(train_df), len(val_df), len(test_df))
        print("Train distribution:")
        print(train_df["label"].value_counts().sort_index())
        print("Test distribution:")
        print(test_df["label"].value_counts().sort_index())
        print("Shared groups train/test:", set(train_df.participant_id) & set(test_df.participant_id))
        print("Shared groups train/val:", set(train_df.participant_id) & set(val_df.participant_id))

        train_df.to_csv(args.output_dir / "splits" / f"fold{fold}_train.csv", index=False)
        val_df.to_csv(args.output_dir / "splits" / f"fold{fold}_val.csv", index=False)
        test_df.to_csv(args.output_dir / "splits" / f"fold{fold}_test.csv", index=False)

        t0 = time.time()
        train_ds = build_dataset(args.input_mode, train_df, args, transform)
        test_ds = build_dataset(args.input_mode, test_df, args, transform)
        x_train_all, y_train_all, _, _ = extract_embeddings(model, train_ds, device, args.batch_size, args.num_workers)
        x_test, y_test, _, _ = extract_embeddings(model, test_ds, device, args.batch_size, args.num_workers)
        extract_time = time.time() - t0

        np.savez_compressed(args.output_dir / "embeddings" / f"fold{fold}_{args.model}.npz", x_train=x_train_all, y_train=y_train_all, x_test=x_test, y_test=y_test)

        for budget in args.budgets:
            sub_idx = sample_budget_indices(y_train_all, budget, seed=args.seed + fold)
            try:
                metrics_out = evaluate_probe(x_train_all[sub_idx], y_train_all[sub_idx], x_test, y_test, seed=args.seed)
            except ValueError as exc:
                print(f"Skipping budget={budget}: {exc}")
                continue
            n_classes = int(len(np.unique(np.concatenate([y_train_all, y_test]))))
            row = {
                "fold": fold,
                "strategy": "dinov2_frozen_linear_probe",
                "model": args.model,
                "task": str(task_name),
                "input_mode": args.input_mode,
                "views": "+".join(args.views) if args.input_mode == "h5" else "csv",
                "budget": budget,
                "n_train": int(len(sub_idx)),
                "n_test": int(len(y_test)),
                "n_classes": n_classes,
                "extract_time_seconds": extract_time,
                "trainable_params": int(x_train_all.shape[1] * n_classes + n_classes),
            }
            row.update(metrics_out)
            all_results.append(row)
            print(row)

    results = pd.DataFrame(all_results)
    results_path = args.output_dir / "results_linear_probe.csv"
    results.to_csv(results_path, index=False)
    print("\nSaved results to:", results_path)
    print(results)


if __name__ == "__main__":
    main()
