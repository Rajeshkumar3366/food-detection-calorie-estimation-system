# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""End-to-end inference: image -> detected food items -> calorie estimate.

Usage:
    python src/predict.py --image path/to/photo.jpg
"""

import argparse
from pathlib import Path

import cv2
import joblib
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument(
        "--weights",
        default=str(ROOT / "models" / "runs" / "yolo_food" / "weights" / "best.pt"),
    )
    p.add_argument(
        "--calorie-model", default=str(ROOT / "models" / "calorie_model.joblib")
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--merge-iou",
        type=float,
        default=0.2,
        help="Same-class boxes overlapping above this IoU are treated as the "
        "same physical item and merged before calorie scoring (keeping the "
        "highest-confidence box), instead of each being scored and summed "
        "separately. The detector's own NMS only collapses near-duplicate "
        "boxes (IoU > ~0.7); it doesn't catch several looser, imprecise "
        "boxes covering one item, which is common for touching/stacked food.",
    )
    p.add_argument("--output", default=None, help="Path to save the annotated image")
    return p.parse_args()


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping_detections(detections: list, iou_thresh: float) -> list:
    """Collapse same-class detections that overlap above iou_thresh, keeping
    the highest-confidence box in each overlapping group and dropping the
    rest - so one physical item detected as several imprecise, partially
    overlapping boxes gets scored once, not once per box."""
    kept = []
    by_class = {}
    for det in detections:
        by_class.setdefault(det["class_name"], []).append(det)

    for class_name, dets in by_class.items():
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        suppressed = [False] * len(dets)
        for i, det_i in enumerate(dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if box_iou(det_i["box"], dets[j]["box"]) > iou_thresh:
                    suppressed[j] = True
    return kept


def estimate_calories(calorie_models: dict, class_name: str, area_ratio: float) -> float:
    model = calorie_models.get(class_name)
    if model is None:
        return 0.0
    # 1) Clip area_ratio to the range observed in training, so a box far
    # outside it doesn't extrapolate off the end of the line.
    clipped_area = min(max(area_ratio, model["area_min"]), model["area_max"])
    calories = model["coef"] * clipped_area + model["intercept"]
    # 2) Clip the resulting calories to this class's [smallest, largest]
    # reference serving. This is the real backstop: some training images are
    # themselves close-up shots with a large area_ratio, so step 1 alone
    # still lets the line reach implausible values within that "observed"
    # range. No single detected item should be estimated above what a
    # plausible large real-world serving of that dish contains.
    return min(max(calories, model["calorie_min"]), model["calorie_max"])


def load_models(weights_path: str, calorie_model_path: str):
    """Load the detector and calorie models once, so callers that run
    inference on many images (e.g. the GUI) don't reload them per image."""
    detector = YOLO(weights_path)
    calorie_models = joblib.load(calorie_model_path)
    return detector, calorie_models


def run_inference(
    detector, calorie_models, image_path: str, conf: float = 0.25, merge_iou: float = 0.2
):
    """Detect food items in one image and score each with its calorie
    estimate. Returns (items, total_calories, n_merged); each item has
    class_name, confidence, box, area_ratio, and calories. n_merged is how
    many raw detections were dropped as overlapping duplicates."""
    results = detector.predict(image_path, conf=conf, verbose=False)
    result = results[0]
    img_h, img_w = result.orig_shape

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        class_name = result.names[cls_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            }
        )

    n_raw = len(detections)
    detections = merge_overlapping_detections(detections, merge_iou)
    n_merged = n_raw - len(detections)

    items = []
    total_calories = 0.0
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        calories = estimate_calories(calorie_models, det["class_name"], area_ratio)
        total_calories += calories
        items.append({**det, "area_ratio": area_ratio, "calories": calories})

    return items, total_calories, n_merged


def annotate_image(image_path: str, items: list):
    """Draw a green box + class/calorie label for each item onto a copy of
    the source image. Returns a BGR numpy array (as cv2 reads it)."""
    img = cv2.imread(image_path)
    for item in items:
        x1, y1, x2, y2 = [int(v) for v in item["box"]]
        label = f"{item['class_name']} ~{item['calories']:.0f}kcal"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            img, label, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
        )
    return img


def main():
    args = parse_args()
    detector, calorie_models = load_models(args.weights, args.calorie_model)

    items, total_calories, n_merged = run_inference(
        detector, calorie_models, args.image, conf=args.conf, merge_iou=args.merge_iou
    )

    print(f"\nDetected {len(items)} item(s) in {args.image}", end="")
    print(f" ({n_merged} overlapping duplicate(s) merged away)" if n_merged else "", end="")
    print(":")
    for item in items:
        print(
            f"  {item['class_name']:20s} conf={item['confidence']:.2f}  "
            f"~{item['calories']:.0f} kcal"
        )
    print(f"\nTotal estimated calories: {total_calories:.0f} kcal\n")

    if args.output:
        annotated = annotate_image(args.image, items)
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved to {args.output}")


if __name__ == "__main__":
    main()
