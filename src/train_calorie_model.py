# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""Fit one linear regression per class: area_ratio -> calories."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent


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
        y = group["calories"].values
        reg = LinearRegression()
        reg.fit(X, y)

        class_servings = nutrition[nutrition["class_name"] == class_name]["calories"]

        # area_min/area_max bound the area_ratio range actually observed in
        # training; predict.py clips to it first so a box far outside it
        # doesn't extrapolate off the end of the line. That alone isn't
        # enough though: some training images are themselves close-up shots
        # with a large area_ratio, so the line can still legitimately reach
        # very large outputs within that "observed" range. calorie_min/max -
        # the smallest/largest serving for this class in the nutrition table
        # - is the real backstop: predict.py clips the final calorie output
        # to it, so no prediction can exceed what a plausible real serving
        # of that dish actually contains.
        models[class_name] = {
            "coef": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
            "area_min": float(group["area_ratio"].min()),
            "area_max": float(group["area_ratio"].max()),
            "calorie_min": float(class_servings.min()),
            "calorie_max": float(class_servings.max()),
        }
        fit_summary[class_name] = {
            "n_samples": len(group),
            "r2": round(float(reg.score(X, y)), 4),
        }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, args.model_out)

    meta = {
        "nutrition_source": args.nutrition,
        "training_data_source": args.data,
        "fit_summary": fit_summary,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(models)} per-class linear models -> {args.model_out}")
    for class_name, info in sorted(fit_summary.items()):
        print(f"  {class_name:20s} n={info['n_samples']:4d}  R2={info['r2']}")


if __name__ == "__main__":
    main()
