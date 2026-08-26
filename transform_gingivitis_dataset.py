#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


YOLO_TO_MGI = {
    2: 2,  # MGI0
    3: 2,  # MGI1
    4: 2,  # MGI2
    5: 3,  # MGI3
    6: 4,  # MGI4
}

SPLITS = {
    "Training": "train",
    "Validation": "val",
    "Test": "test",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Duy 2024 gingivitis YOLO annotations to image-level MGI classification."
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Root directory containing Training/, Validation/, and Test/."
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("gingivitis_global_mgi.csv"),
        help="Output CSV path."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser()

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    rows = []

    for folder_name, split_name in SPLITS.items():
        image_dir = dataset_dir / folder_name / "Images"
        label_dir = dataset_dir / folder_name / "Labels"

        if not image_dir.exists():
            print(f"Warning: missing image directory: {image_dir}")
            continue

        if not label_dir.exists():
            print(f"Warning: missing label directory: {label_dir}")
            continue

        label_files = sorted(label_dir.glob("*.txt"))

        print(
            f"{folder_name}: "
            f"{len(label_files)} annotation files found"
        )

        for label_path in label_files:
            image_name = label_path.stem + ".jpg"
            image_path = image_dir / image_name

            if not image_path.exists():
                print(f"Warning: missing image {image_path}")
                continue

            mgi_scores = []

            with label_path.open("r") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    class_id = int(line.split()[0])

                    if class_id in YOLO_TO_MGI:
                        mgi_scores.append(
                            YOLO_TO_MGI[class_id]
                        )

            if not mgi_scores:
                print(
                    f"Warning: no MGI label in {label_path}"
                )
                continue

            # Image-level label =
            # highest MGI category present in the image
            global_mgi = max(mgi_scores)

            rows.append({
                "image_name": image_name,
                "image_path": str(image_path.resolve()),
                "global_mgi": global_mgi,
                "label_name": f"MGI{global_mgi}",
                "split": split_name,
                "participant_id": label_path.stem,
                "source_folder": folder_name,
            })

    if not rows:
        raise RuntimeError(
            "No valid samples were found. "
            "Check dataset path and folder structure."
        )

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print("\nSaved:", output_csv)
    print("Total:", len(df))

    print("\nSplit distribution:")
    print(
        df["split"]
        .value_counts()
        .sort_index()
    )

    print("\nClass distribution:")
    print(
        df["global_mgi"]
        .value_counts()
        .sort_index()
    )

    print("\nClass distribution by split:")
    print(
        pd.crosstab(
            df["split"],
            df["global_mgi"]
        )
    )


if __name__ == "__main__":
    main()
