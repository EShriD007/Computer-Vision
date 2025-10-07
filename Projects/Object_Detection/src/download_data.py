import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from pathlib import Path
import os, sys

MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", 300))
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "coco"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Read the classes that we are targetting with the number of images per class
classes = [c.strip() for c in open(BASE / "configs" / "classes.txt").read().splitlines() if c.strip()]

print(f"Downloading Open Images subset for {len(classes)} classes (max {MAX_PER_CLASS} per class)…")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"], # This gives the bounding box annotations enclosing the object that we are targetting. We can also put classifications to just get the class, segmentation to get the exact polygon covering the object
    classes=classes,
    max_samples=MAX_PER_CLASS * len(classes),
    only_matching=True,     # this will remove any other annotations in the image rather than what we have given in the classes.txt
)

# Find the actual column that will have the detected output
schema = dataset.get_field_schema()
candidates = ["detections", "ground_truth", "open_images_detections", "labels"]
label_field = next((c for c in candidates if c in schema), None)
# fallback: auto-detect first Detections-like field
if label_field is None:
    for name, f in schema.items():
        # crude but effective: FiftyOne shows 'Detections' in the repr of the field's document type
        if "Detections" in str(f):
            label_field = name
            break
if label_field is None:
    print("Could not find a detections field on the samples.")
    print("Fields:", schema)
    sys.exit(1)

print(f"Using label field: '{label_field}'")

# Keep only the sample that has detections and images with empty detection field will be removed.
dataset = dataset.match(F(label_field).exists())

count_all = len(dataset)
if len(dataset) == 0:
    print("After filtering, no samples have detections. Try increasing MAX_PER_CLASS or adjusting classes.")
    sys.exit(1)

# Split the images for training, validation and Testing
dataset.shuffle()  # ok whether it's a Dataset or a View
n_train = int(0.70 * count_all)
n_val   = int(0.15 * count_all)
dataset.match_tags([])  # clear any prior tags
dataset.take(n_train).tag_samples("train")
dataset.skip(n_train).take(n_val).tag_samples("val")
dataset.skip(n_train + n_val).tag_samples("test")

# Save the image in the folder
export_root = str(DATA_DIR)
print("Exporting to COCO format…")
for split in ("train", "val", "test"):
    view = dataset.match_tags(split).match(F(label_field).exists())
    count = view.count()
    if count == 0:
        print(f"No samples in split '{split}' — skipping export.")
        continue
    outdir = os.path.join(export_root, split)
    view.export(
        export_dir=outdir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,   # <- use the detected field
    )
    print(f"{split}: {count} samples -> {outdir}")

print("\nCOCO exported under:", export_root)
