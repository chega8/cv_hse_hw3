"""Тренировочный и валидационный циклы DETR."""
from __future__ import annotations

import math
import time

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import labels_to_xyxy


def _move_labels(labels, device):
    return [{k: v.to(device) for k, v in t.items()} for t in labels]


def train_one_epoch(model, loader, optimizer, device, cfg, writer, epoch, global_step):
    """Одна эпоха обучения. Логирует компоненты loss в TensorBoard."""
    model.train()
    running = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    n_batches = len(loader)
    t0 = time.time()

    for it, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = _move_labels(batch["labels"], device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # --- агрегируем компоненты ---
        running["loss"] += loss.item()
        for k in ("loss_ce", "loss_bbox", "loss_giou"):
            if k in loss_dict:
                running[k] += loss_dict[k].item()

        if it % cfg.log_every == 0:
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/loss_ce", loss_dict.get("loss_ce", torch.tensor(0.0)).item(), global_step)
            writer.add_scalar("train/loss_bbox", loss_dict.get("loss_bbox", torch.tensor(0.0)).item(), global_step)
            writer.add_scalar("train/loss_giou", loss_dict.get("loss_giou", torch.tensor(0.0)).item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            print(f"  epoch {epoch} | step {it:>4}/{n_batches} | "
                  f"loss {loss.item():.3f} (ce {loss_dict.get('loss_ce', 0):.3f}, "
                  f"bbox {loss_dict.get('loss_bbox', 0):.3f}, giou {loss_dict.get('loss_giou', 0):.3f})")

        global_step += 1

    dt = time.time() - t0
    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["sec_per_epoch"] = dt
    return avg, global_step


@torch.no_grad()
def evaluate(model, loader, processor, device, cfg):
    """Считает mAP / mAP50 на валидации через torchmetrics."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.stack([t["orig_size"] for t in labels]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=cfg.eval_threshold, target_sizes=target_sizes
        )

        preds, targets = [], []
        for res, label in zip(results, labels):
            preds.append({
                "boxes": res["boxes"].cpu(),
                "scores": res["scores"].cpu(),
                "labels": res["labels"].cpu(),
            })
            gt_boxes, gt_labels = labels_to_xyxy(label)
            targets.append({"boxes": gt_boxes, "labels": gt_labels})
        metric.update(preds, targets)

    res = metric.compute()
    out = {
        "map": float(res["map"]),
        "map_50": float(res["map_50"]),
        "map_75": float(res["map_75"]),
        "mar_100": float(res["mar_100"]),
    }
    # NaN -> 0 (бывает при слишком малой валидации)
    return {k: (0.0 if math.isnan(v) else v) for k, v in out.items()}
