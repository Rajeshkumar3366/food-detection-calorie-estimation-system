# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""Fit per-class linear regressions: area_ratio -> calories/protein/carbs/fat/fiber."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent

NUTRITION_TARGETS = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", default=str(ROOT / "data" / "calorie_training_data.csv")
    )
    p.add_argument("--nutrition", default=str(ROOT / "data" / "food_calories.csv"))
    p.add_argument("--model-out", default=str(ROOT / "models" / "calorie_model.joblib"))
    p.add_argument("--meta-out", default=str(ROOT / "models" / "calorie_meta.json"))
    return p.parse_args()


def main():
    args = parse_args()
    samples = pd.read_csv(args.data)
    nutrition = pd.read_csv(args.nutrition)

    models = {}
    fit_summary = {}
    for class_name, group in samples.groupby("class_name"):
        X = group[["area_ratio"]].values
        class_servings = nutrition[nutrition["class_name"] == class_name]

        # area_min/area_max bound the area_ratio range actually observed in
        # training (shared across all 5 targets - they're all fit on the same
        # rows); predict.py clips to it first so a box far outside it doesn't
        # extrapolate off the end of any of the lines. That alone isn't
        # enough though: some training images are themselves close-up shots
        # with a large area_ratio, so the line can still legitimately reach
        # very large outputs within that "observed" range. Each target's own
        # min/max - its smallest/largest serving value for this class in the
        # nutrition table - is the real backstop: predict.py clips each
        # target's final output to it, so no prediction can exceed what a
        # plausible real serving of that dish actually contains.
        target_models = {}
        target_r2 = {}
        for target in NUTRITION_TARGETS:
            y = group[target].values
            reg = LinearRegression()
            reg.fit(X, y)
            target_models[target] = {
                "coef": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "min": float(class_servings[target].min()),
                "max": float(class_servings[target].max()),
            }
            target_r2[target] = round(float(reg.score(X, y)), 4)

        models[class_name] = {
            "area_min": float(group["area_ratio"].min()),
            "area_max": float(group["area_ratio"].max()),
            "targets": target_models,
        }
        fit_summary[class_name] = {
            "n_samples": len(group),
            "r2": target_r2,
        }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.model_out)

    meta = {
        "nutrition_source": args.nutrition,
        "training_data_source": args.data,
        "targets": NUTRITION_TARGETS,
        "fit_summary": fit_summary,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(models)} per-class nutrition models -> {args.model_out}")
    for class_name, info in sorted(fit_summary.items()):
        r2_str = " ".join(f"{t.split('_')[0]}={v}" for t, v in info["r2"].items())
        print(f"  {class_name:20s} n={info['n_samples']:4d}  R2: {r2_str}")


if __name__ == "__main__":
    main()
