# Comparing Pretraining Strategies for Intra-Oral Image Classification

Research code for comparing supervised learning, transfer learning, and self-supervised visual representations on intra-oral photographs, with a focus on **annotation efficiency**.

This repository accompanies ongoing research on classification of periodontal conditions and gingival inflammation. It contains experimental scripts for ResNet-50, DINOv2, and a lightweight masked-image reconstruction baseline, together with prediction exports and statistical-analysis code.

> **Research code:** This is an experimental repository, not a clinical device or a validated diagnostic tool. The scripts have different interfaces and some experiments require private datasets.

## Methods

| Approach | Script | Adaptation |
|---|---|---|
| ResNet-50, ImageNet pretrained | `train_resnet50_diagnosis_baseline.py` | Supervised end-to-end training |
| ResNet-50, random initialization | `train_resnet50_diagnosis_baseline.py` | Supervised training from scratch |
| DINOv2 frozen features | `run_jacksparo_dino_probe_clinical_multiclass.py` | Class-balanced logistic regression on extracted embeddings |
| DINOv2 partial fine-tuning | `train_dinov2_partial_finetune.py` | Classification head and selected transformer blocks |
| Masked convolutional autoencoder | `train_mae_smoke_test.py` | Self-supervised image reconstruction |
| Masked-autoencoder feature probe | `run_mae_checkpoint_probe_from_splits_flexible_with_predictions.py` | Downstream evaluation from a saved checkpoint |

**Important:** The masked reconstruction model is a small *convolutional autoencoder with random block masking*, not the original ViT-based MAE architecture. Its implementation is explicitly a smoke-test baseline rather than a state-of-the-art foundation model.

## Datasets and tasks

### Periodontal diagnosis (JacksParo)

The scripts support the project-specific JacksParo data stored in HDF5, with associated annotation, log, HDF5-index and image-correction files. Diagnosis labels can be configured as binary or three-class (`healthy`, `gingivitis`, `active periodontitis`). The selected intra-oral view can be specified with `--views`, for example `Photo FACE`.

These images and metadata are **not included in the repository**. Access must be obtained separately from the data owner.

### Gingivitis (Duy 2024)

`transform_gingivitis_dataset.py` converts the dataset's YOLO-style image annotations into an image-level classification manifest, using the maximum mapped severity observed in an image. It preserves the source `Training`, `Validation` and `Test` partitions.

**Check the mapping before using the labels:** the current conversion script merges YOLO class IDs 2, 3 and 4 into output class 2, while class IDs 5 and 6 map to output classes 3 and 4. Consequently, the current implementation is a **three-category grouped target**, not a five-class MGI0–MGI4 target. The mapping is a project-specific experimental choice and should be described explicitly when reporting results.

Expected directory structure:

```text
Dataset/
├── Training/
│   ├── Images/
│   └── Labels/
├── Validation/
│   ├── Images/
│   └── Labels/
└── Test/
    ├── Images/
    └── Labels/
```

Generate the classification manifest:

```bash
python transform_gingivitis_dataset.py \
  --dataset_dir /path/to/Dataset \
  --output_csv gingivitis_global_mgi.csv
```

The output CSV contains image paths, derived labels, source partitions and sample identifiers. Verify the printed split and class distributions before training.

## Installation

The code is Python-based and uses PyTorch, torchvision, pandas, NumPy, scikit-learn, Pillow and h5py; some scripts also use `tqdm`. The analysis uses R and packages including `tidyverse`, `lme4`, `emmeans`, and `broom.mixed`.

A historical Docker setup is provided under `docker/`, but it is **not a fully verified, reproducible environment specification**. Check CUDA, Python and package compatibility for your machine.

Example installation in an existing compatible PyTorch environment:

```bash
pip install numpy pandas scikit-learn pillow h5py tqdm
```

Install `torch` and `torchvision` using the instructions appropriate for your CUDA/CPU environment.

DINOv2 is loaded through `torch.hub`; the first run may require internet access and will download pretrained weights.

## Experiments

### 1. ResNet-50 baseline on JacksParo

ImageNet initialization:

```bash
python train_resnet50_diagnosis_baseline.py \
  --input_mode h5 \
  --data_dir /path/to/jacksparo \
  --diagnosis_mode multiclass \
  --views "Photo FACE" \
  --n_splits 5 \
  --val_size 0.15 \
  --pretrained imagenet \
  --epochs 30 \
  --batch_size 32 \
  --output_dir outputs_resnet50_imagenet
```

For random initialization, change `--pretrained imagenet` to `--pretrained none` and use a separate output directory.

**Note:** The ResNet script in this archive does **not** expose a `--budgets` option. It evaluates the available training partition, not a full annotation-budget curve.

### 2. DINOv2 frozen-feature probe

```bash
python run_jacksparo_dino_probe_clinical_multiclass.py \
  --input_mode h5 \
  --data_dir /path/to/jacksparo \
  --label_mode diagnosis \
  --diagnosis_mode multiclass \
  --views "Photo FACE" \
  --n_splits 5 \
  --budgets 10 25 50 100 250 full \
  --output_dir outputs_dino_diagnosis
```

This method extracts frozen DINOv2 representations and trains a class-balanced logistic regression classifier for each requested annotation budget.

**Known issue in the uploaded snapshot:** `run_jacksparo_dino_probe_clinical_multiclass.py` contains an extra closing parenthesis in the `--data_dir` argument declaration (around line 546). Remove that parenthesis before running the script. The other Python files passed a syntax-compilation check, but their full execution has not been verified here.

### 3. DINOv2 partial fine-tuning

This script reads train/test split CSV files generated by a preceding experiment; an optional validation CSV can be supplied per fold.

```bash
python train_dinov2_partial_finetune.py \
  --input_mode h5 \
  --data_dir /path/to/jacksparo \
  --split_dir outputs_dino_diagnosis/splits \
  --output_dir outputs_dino_partialft \
  --finetune_modes head last_block last_2_blocks last_4_blocks \
  --budgets 10 25 50 100 250 full \
  --epochs 20 \
  --batch_size 32
```

For a dataset with CSV image paths, change `--input_mode h5` to `--input_mode csv`, and use a compatible `--split_dir`.

Modes supported by the code: `head`, `last_block`, `last_2_blocks`, `last_4_blocks`, `full`. With a validation partition, model selection uses validation macro-F1; without validation, the final epoch is retained. The script writes budget- and mode-specific histories, alongside aggregate result files.

### 4. Masked-image reconstruction pretraining

Prepare a CSV manifest with a column named `path`, containing accessible image paths.

```bash
python train_mae_smoke_test.py \
  --manifest ssl_manifest.csv \
  --out-dir outputs_mae_smoke \
  --epochs 5 \
  --batch-size 64 \
  --image-size 224 \
  --mask-ratio 0.5 \
  --patch-size 16
```

Default optimization: AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size `64`, and MSE reconstruction loss. A checkpoint is written to `last_checkpoint.pt`.

### 5. Probe the masked-autoencoder checkpoint

```bash
python run_mae_checkpoint_probe_from_splits_flexible_with_predictions.py \
  --input_mode h5 \
  --data_dir /path/to/jacksparo \
  --split_dir outputs_dino_diagnosis/splits \
  --checkpoint outputs_mae_smoke/last_checkpoint.pt \
  --output_dir outputs_mae_probe \
  --budgets 10 25 50 100 250 full
```

## Annotation budgets and evaluation

The experiment scripts can use budgets such as `10`, `25`, `50`, `100`, `250`, and `full`, where numerical budgets denote **total selected training images**, not images per class. `full` means all images available in the training partition for the corresponding fold.

The scripts support group-aware splitting for JacksParo and predefined train/validation/test partitions for CSV datasets. **For meaningful comparisons, reuse the same partitions across methods.** Model selection should rely on validation data where available; the test partition is reserved for final evaluation.

Depending on the script, outputs include per-fold metrics, per-image predictions, class probabilities, training histories and checkpoints. Metrics include macro-F1 and other classification measures; the analysis scripts additionally compute quadratic weighted kappa (QWK).

## Statistical analysis

- `statistical_analysis.ipynb`: notebook-based analyses.
- `statistical_analysis/graphs.R`: aggregate analysis and figure generation from experiment prediction files.
- `statistical_analysis/mixed_models_analyses`: additional mixed-model analysis material.

The R scripts reference **specific experiment output directory names**. Update those paths to match your runs before executing them. The figure budgets specified in `graphs.R` are `10, 25, 50, 100, 250, full`. Mixed-effects analyses should be reported with their exact fitted formulas and the subset of budgets actually analyzed.

## Repository layout

```text
.
├── docker/
├── run_jacksparo_dino_probe_clinical_multiclass.py
├── run_mae_checkpoint_probe_from_splits_flexible_with_predictions.py
├── train_dinov2_partial_finetune.py
├── train_mae_smoke_test.py
├── train_resnet50_diagnosis_baseline.py
├── transform_gingivitis_dataset.py
├── statistical_analysis.ipynb
└── statistical_analysis/
    ├── graphs.R
    └── mixed_models_analyses
```

## Reproducibility and limitations

- Dataset files, model checkpoints and experimental outputs are not bundled with this repository.
- The code contains several independent command-line interfaces; not all options are shared by all methods.
- Predefined dataset partitions and group-aware cross-validation represent different experimental settings and should not be conflated.
- Very small annotation budgets can omit classes; inspect class counts for every sampled subset.
- Results may depend on pretrained checkpoint versions, CUDA/PyTorch versions and data availability.
- The current DINO probe script requires the syntax fix noted above.
- The lightweight masked autoencoder should not be presented as an implementation of the original MAE architecture.

## Citation

If you use this code in your research, please cite our paper:

[Comparing Pretraining Strategies for Intra-Oral Image Analysis Under Limited Annotation Budgets](https://papers.miccai.org/miccai-2026-sat/ODIN_004.html)

```bibtex
@InProceedings{MarNic_Comparing_MICCAISAT2026,
        author = { Martin, Nicolas AND Chevallet, Jean-Pierre AND Mulhem, Philippe},
        title = { { Comparing Pretraining Strategies for Intra-Oral Image Analysis Under Limited Annotation Budgets } },
        booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
        year = {2026},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 17270},
        month = {pending},
        page = {pending}
}
```

## License

See [LICENSE](LICENSE) for the repository's license terms. Dataset licenses and access conditions must be checked separately.
