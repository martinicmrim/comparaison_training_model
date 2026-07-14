from pathlib import Path
import pandas as pd
import os 

DATASET_DIR = Path(os.path.expandvars("$HOME/data2/duy_2024_gingivitis"))
OUTPUT_CSV = Path("gingivitis_global_mgi.csv")

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

rows = []

for folder_name, split_name in SPLITS.items():
    image_dir = DATASET_DIR / folder_name / "Images"
    label_dir = DATASET_DIR / folder_name / "Labels"

    for label_path in sorted(label_dir.glob("*.txt")):
        image_name = label_path.stem + ".jpg"
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"Warning: missing image {image_path}")
            continue

        mgi_scores = []

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                class_id = int(line.split()[0])

                if class_id in YOLO_TO_MGI:
                    mgi_scores.append(YOLO_TO_MGI[class_id])

        if not mgi_scores:
            print(f"Warning: no MGI label in {label_path}")
            continue

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

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Total:", len(df))
print("\nSplit distribution:")
print(df["split"].value_counts())
print("\nClass distribution:")
print(df["global_mgi"].value_counts().sort_index())
print("\nClass distribution by split:")
print(pd.crosstab(df["split"], df["global_mgi"]))