"""Error analysis + визуализация предсказаний DETR.

Загружает чекпойнт (по умолчанию best.pt), прогоняет валидацию и:
  * рисует predicted vs ground-truth боксы на нескольких примерах;
  * классифицирует ошибки на категории (TIDE-подобно):
      - correct            : IoU>=0.5 и класс верный;
      - localization error : класс верный, но 0.1<=IoU<0.5;
      - classification error: IoU>=0.5, но класс неверный;
      - duplicate          : дубль на уже найденный объект;
      - background FP      : предсказание без объекта (IoU<0.1);
      - missed GT (FN)     : объект не найден;
  * сохраняет сводную таблицу и bar-chart.

Запуск:
  python src/error_analysis.py --ckpt outputs/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import DetrDataset, labels_to_xyxy, load_splits, make_collate_fn
from model import build_model, build_processor

IOU_FG = 0.5      # порог "правильной" локализации
IOU_BG = 0.1      # ниже этого — фон/ложное срабатывание


def classify_image(pred_boxes, pred_labels, gt_boxes, gt_labels):
    """Категоризует предсказания и GT одного изображения. pred отсортированы по score (desc)."""
    counts = Counter()
    n_gt = gt_boxes.shape[0]
    gt_matched = [False] * n_gt

    if pred_boxes.shape[0] and n_gt:
        ious = box_iou(pred_boxes, gt_boxes)  # (P, G)
    else:
        ious = None

    for p in range(pred_boxes.shape[0]):
        if ious is None or n_gt == 0:
            counts["background_fp"] += 1
            continue
        iou_row = ious[p]
        best_iou, best_g = torch.max(iou_row, dim=0)
        best_iou = best_iou.item(); best_g = best_g.item()

        if best_iou < IOU_BG:
            counts["background_fp"] += 1
        elif best_iou < IOU_FG:
            # достаточно перекрытия для попытки, но локализация плохая
            if pred_labels[p].item() == gt_labels[best_g].item():
                counts["localization_err"] += 1
            else:
                counts["localization_err"] += 1  # плохая локализация (класс вторичен)
        else:  # хорошая локализация
            if pred_labels[p].item() != gt_labels[best_g].item():
                counts["classification_err"] += 1
            elif gt_matched[best_g]:
                counts["duplicate"] += 1
            else:
                gt_matched[best_g] = True
                counts["correct"] += 1

    counts["missed_gt"] += sum(1 for m in gt_matched if not m)
    return counts


@torch.no_grad()
def run(cfg: Config, ckpt_path: Path, n_viz: int = 8):
    device = cfg.device()
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    processor = build_processor(cfg)
    train_raw, val_raw, id2label, label2id = load_splits(cfg)

    model = build_model(cfg, id2label, label2id).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[eval] loaded {ckpt_path} (epoch {state.get('epoch')})")

    collate = make_collate_fn(processor)
    val_ds = DetrDataset(val_raw, processor, augment=False)
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    total = Counter()
    viz_items = []  # (raw_index, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    raw_idx = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = batch["labels"]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.stack([t["orig_size"] for t in labels]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=cfg.viz_threshold, target_sizes=target_sizes
        )

        for res, label in zip(results, labels):
            pb = res["boxes"].cpu(); pl = res["labels"].cpu(); ps = res["scores"].cpu()
            order = torch.argsort(ps, descending=True)
            pb, pl, ps = pb[order], pl[order], ps[order]
            gb, gl = labels_to_xyxy(label)
            total += classify_image(pb, pl, gb, gl)
            if len(viz_items) < n_viz:
                viz_items.append((raw_idx, pb, pl, ps, gb, gl))
            raw_idx += 1

    save_error_table(total, cfg, id2label)
    save_error_chart(total, cfg.fig_dir)
    save_visualizations(viz_items, val_raw, id2label, cfg.fig_dir)
    return total


def save_error_table(counts: Counter, cfg: Config, id2label):
    order = ["correct", "localization_err", "classification_err",
             "duplicate", "background_fp", "missed_gt"]
    rus = {
        "correct": "Верно (класс+локализация)",
        "localization_err": "Ошибка локализации (0.1<=IoU<0.5)",
        "classification_err": "Ошибка классификации (IoU>=0.5, класс неверный)",
        "duplicate": "Дубликаты",
        "background_fp": "Ложные срабатывания (фон, IoU<0.1)",
        "missed_gt": "Пропущенные объекты (FN)",
    }
    total_pred = sum(counts[k] for k in ("correct", "localization_err",
                                         "classification_err", "duplicate", "background_fp"))
    lines = ["| Категория | Кол-во | % от предсказаний |",
             "|-----------|-------:|------------------:|"]
    for k in order:
        c = counts.get(k, 0)
        pct = (100 * c / total_pred) if total_pred and k != "missed_gt" else float("nan")
        pct_s = "—" if k == "missed_gt" else f"{pct:.1f}%"
        lines.append(f"| {rus[k]} | {c} | {pct_s} |")
    text = "\n".join(lines)
    (cfg.out / "error_analysis.md").write_text(text)
    (cfg.out / "error_analysis.json").write_text(json.dumps(dict(counts), indent=2, ensure_ascii=False))
    print("\n=== Error analysis ===")
    print(text)


def save_error_chart(counts: Counter, fig_dir: Path):
    order = ["correct", "localization_err", "classification_err",
             "duplicate", "background_fp", "missed_gt"]
    labels = ["correct", "localization", "classification", "duplicate", "background FP", "missed GT"]
    vals = [counts.get(k, 0) for k in order]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars)
    ax.set_ylabel("count"); ax.set_title("DETR error analysis (val)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = fig_dir / "error_analysis.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots] -> {out}")


def save_visualizations(viz_items, val_raw, id2label, fig_dir: Path):
    """Рисует predicted (сплошные) и GT (пунктир) боксы на примерах."""
    n = len(viz_items)
    if n == 0:
        return
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (idx, pb, pl, ps, gb, gl) in zip(axes, viz_items):
        image = val_raw[idx]["image"].convert("RGB")
        ax.imshow(image)
        # GT — пунктирные зелёные
        for (x1, y1, x2, y2), c in zip(gb.tolist(), gl.tolist()):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor="lime", linewidth=2, linestyle="--"))
        # Предсказания — сплошные красные с подписью
        for (x1, y1, x2, y2), c, s in zip(pb.tolist(), pl.tolist(), ps.tolist()):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor="red", linewidth=2))
            ax.text(x1, max(y1 - 4, 0), f"{id2label.get(c, c)} {s:.2f}",
                    color="white", fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.6, pad=1, edgecolor="none"))
        ax.set_title(f"img #{idx}  (green=GT, red=pred)")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    out = fig_dir / "predictions.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plots] -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    p.add_argument("--dataset-name", default="cppe-5")
    p.add_argument("--val-size", type=int, default=40)
    p.add_argument("--image-short", type=int, default=480)
    p.add_argument("--image-long", type=int, default=800)
    p.add_argument("--n-viz", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.3,
                   help="порог уверенности для отбора предсказаний")
    a = p.parse_args()
    cfg = Config(dataset_name=a.dataset_name, val_size=a.val_size,
                 image_short=a.image_short, image_long=a.image_long,
                 viz_threshold=a.threshold)
    cfg.make_dirs()
    run(cfg, Path(a.ckpt), n_viz=a.n_viz)


if __name__ == "__main__":
    main()
