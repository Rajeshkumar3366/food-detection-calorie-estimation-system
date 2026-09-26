# Code written by Rajesh Kumar
# Cambridge Institute of Technology
# Food Detection and Calorie Estimation System

"""Render training-curve PNGs from the exported per-epoch metrics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default=str(ROOT / "models" / "training_metrics.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "models" / "plots"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    epoch = df["epoch"]

    best_map50_epoch = df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"]
    best_map50 = df["metrics/mAP50(B)"].max()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epoch, df["metrics/mAP50(B)"], label="mAP50", color="#2a78d6", linewidth=2)
    ax.plot(epoch, df["metrics/mAP50-95(B)"], label="mAP50-95", color="#eb6834", linewidth=2)
    ax.axvline(best_map50_epoch, color="#898781", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"best mAP50={best_map50:.3f}\n(epoch {int(best_map50_epoch)})",
        xy=(best_map50_epoch, best_map50), xytext=(10, -25),
        textcoords="offset points", fontsize=9, color="#52514e",
    )
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_title("Detection accuracy over training")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epoch, df["train/box_loss"], label="box_loss", color="#2a78d6", linewidth=2)
    ax.plot(epoch, df["train/cls_loss"], label="cls_loss", color="#eb6834", linewidth=2)
    ax.plot(epoch, df["train/dfl_loss"], label="dfl_loss", color="#1baf7a", linewidth=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epoch, df["metrics/precision(B)"], label="precision", color="#4a3aa7", linewidth=2)
    ax.plot(epoch, df["metrics/recall(B)"], label="recall", color="#e34948", linewidth=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_title("Precision / recall over training")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    epoch_time = df["time"].diff().fillna(df["time"].iloc[0])
    ax.bar(epoch, epoch_time, color="#c98500", width=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("seconds")
    ax.set_title("Wall-clock time per epoch")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_time.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 4 plots to {out_dir}")


if __name__ == "__main__":
    main()
