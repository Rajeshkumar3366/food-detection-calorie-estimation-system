# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""Synthesize (area_ratio, nutrition) training pairs per class.

The object-detection dataset has no ground-truth nutrition/portion labels, so
we calibrate a portion-size -> nutrition mapping from two things we *do* have:

1. The actual distribution of detected-box area ratios (box_w * box_h, already
   normalized 0-1 in YOLO label format) per class, from labels/train.
2. The reference nutrition table (data/food_calories.csv), which gives real
   calories/protein/carbs/fat/fiber values at 3 serving sizes (small/regular/
   large) per class.

For each class we anchor the "regular" serving's values to that class's
median observed area_ratio, then scale linearly: an area_ratio twice the
median implies roughly twice the regular serving's calories, protein, carbs,
fat, and fiber. The small and large serving rows are blended in as additional
calibration anchors so the mapping isn't just extrapolated from a single
point. All five nutrition targets share the same area_ratio axis, since a
class's small/regular/large servings scale (almost exactly) proportionally
across every target, not just calories, so one shared anchor position per
serving row is a very close approximation to five separately-derived ones.
"""

import argparse
from pathlib import Path

import pandas as pd

from classes import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent

NUTRITION_TARGETS = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]


def load_area_ratios(labels_dir: Path) -> pd.DataFrame:
    rows = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        with open(txt_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                rows.append({"class_name": CLASS_NAMES[cls_id], "area_ratio": w * h})
    return pd.DataFrame(rows)


def build_calorie_samples(boxes: pd.DataFrame, nutrition: pd.DataFrame) -> pd.DataFrame:
    samples = []
    for class_name, group in boxes.groupby("class_name"):
        class_rows = nutrition[nutrition["class_name"] == class_name]
        if class_rows.empty:
            continue

        # Sort the class's serving rows by grams so index 0/1/2 = small/regular/large.
        class_rows = class_rows.sort_values("serving_grams").reset_index(drop=True)
        regular = class_rows.iloc[len(class_rows) // 2]
        median_area = group["area_ratio"].median()
        if median_area <= 0:
            continue
        rate_per_area = {t: regular[t] / median_area for t in NUTRITION_TARGETS}

        # 1) One synthetic point per real observed box, scaled from the regular
        # anchor - independently for each of the 5 nutrition targets.
        for area_ratio in group["area_ratio"]:
            row = {"class_name": class_name, "area_ratio": area_ratio}
            for t in NUTRITION_TARGETS:
                row[t] = area_ratio * rate_per_area[t]
            samples.append(row)

        # 2) Extra calibration anchors from the small/large serving rows: place
        # them at area ratios scaled proportionally to their calories relative
        # to the regular serving (calories is the anchor-position driver; the
        # other 4 targets scale almost identically across servings, see the
        # module docstring), so the real multi-serving nutrition data directly
        # informs the fit for all 5 targets, not just the observed area
        # distribution.
        for _, serving_row in class_rows.iterrows():
            if serving_row["calories"] == regular["calories"]:
                continue
            anchor_area = median_area * (serving_row["calories"] / regular["calories"])
            row = {"class_name": class_name, "area_ratio": anchor_area}
            for t in NUTRITION_TARGETS:
                row[t] = serving_row[t]
            samples.append(row)

    return pd.DataFrame(samples)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels-dir",
        default=str(ROOT / "dataset" / "dataset" / "labels" / "train"),
    )
    p.add_argument("--nutrition", default=str(ROOT / "data" / "food_calories.csv"))
    p.add_argument("--output", default=str(ROOT / "data" / "calorie_training_data.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    boxes = load_area_ratios(Path(args.labels_dir))
    nutrition = pd.read_csv(args.nutrition)
    samples = build_calorie_samples(boxes, nutrition)
    samples.to_csv(args.output, index=False)
    print(f"Wrote {len(samples)} samples across {samples['class_name'].nunique()} classes -> {args.output}")


if __name__ == "__main__":
    main()
