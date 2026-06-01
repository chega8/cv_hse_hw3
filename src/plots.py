"""Построение графиков потерь (classification / bbox regression)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(history, fig_dir: Path) -> None:
    """Строит кривые total / classification / bbox-regression loss и mAP по эпохам."""
    if not history:
        return
    epochs = [r["epoch"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- (1) компоненты loss ---
    ax = axes[0]
    ax.plot(epochs, [r["loss"] for r in history], "o-", label="total", color="black")
    ax.plot(epochs, [r["loss_ce"] for r in history], "s-", label="classification (loss_ce)")
    bbox_reg = [r["loss_bbox"] + r["loss_giou"] for r in history]
    ax.plot(epochs, bbox_reg, "^-", label="bbox regression (L1+GIoU)")
    ax.plot(epochs, [r["loss_bbox"] for r in history], "--", alpha=0.6, label="loss_bbox (L1)")
    ax.plot(epochs, [r["loss_giou"] for r in history], ":", alpha=0.8, label="loss_giou")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss components")
    ax.legend(); ax.grid(alpha=0.3)

    # --- (2) метрики ---
    ax = axes[1]
    ax.plot(epochs, [r["map"] for r in history], "o-", label="mAP@[.5:.95]")
    ax.plot(epochs, [r["map_50"] for r in history], "s-", label="mAP@0.5")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP"); ax.set_title("Validation mAP")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "loss_and_map.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots] -> {out}")
