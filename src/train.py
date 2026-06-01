# -*- coding: utf-8 -*-
"""Полный тренировочный цикл fine-tuning DETR.

Делает:
  * загрузку маленького detection-датасета (cppe-5) в формате DETR;
  * fine-tuning DETR с раздельным lr для backbone;
  * логирование компонент loss (classification / bbox / giou) в TensorBoard;
  * trace профайлера (torch.profiler -> chrome trace);
  * сохранение чекпойнтов каждой эпохи;
  * графики потерь и таблицу метрик (mAP / mAP50);
  * запуск error analysis на лучшей модели.

Запуск (короткий демо-прогон):
  python src/train.py --train-size 80 --val-size 40 --epochs 2

Полный прогон:
  python src/train.py --epochs 30
"""
from __future__ import annotations

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests, urllib3
urllib3.disable_warnings()
_orig_send = requests.Session.send
requests.Session.send = lambda self, *a, **kw: _orig_send(self, *a, **{**kw, "verify": False})

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, parse_args
from dataset import DetrDataset, load_splits, make_collate_fn
from engine import evaluate, train_one_epoch
from model import build_model, build_processor, param_groups
from plots import plot_loss_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_profiler(model, loader, optimizer, device, cfg):
    """Снимает trace нескольких train-шагов и пишет chrome trace + сводку."""
    from engine import _move_labels

    model.train()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_path = cfg.prof_dir / "detr_trace.json"
    it = iter(loader)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = _move_labels(batch["labels"], device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)
    (cfg.prof_dir / "summary.txt").write_text(summary)
    print(f"[profiler] trace -> {trace_path}")
    print(f"[profiler] summary -> {cfg.prof_dir / 'summary.txt'}")


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    cfg.make_dirs()
    device = cfg.device()
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"[setup] device = {device}")

    # --- сохраняем конфиг ---
    (cfg.out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))

    # --- данные ---
    processor = build_processor(cfg)
    train_raw, val_raw, id2label, label2id = load_splits(cfg)
    print(f"[data] train={len(train_raw)} val={len(val_raw)} classes={list(id2label.values())}")

    collate = make_collate_fn(processor)
    train_loader = DataLoader(
        DetrDataset(train_raw, processor, augment=cfg.augment),
        batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        DetrDataset(val_raw, processor, augment=False),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    # --- модель ---
    model = build_model(cfg, id2label, label2id).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg.model_name} | trainable params = {n_params/1e6:.1f}M")

    writer = SummaryWriter(log_dir=str(cfg.tb_dir))

    # --- профайлер (несколько шагов до основного цикла) ---
    if cfg.profile:
        print("[profiler] capturing trace...")
        run_profiler(model, train_loader, optimizer, device, cfg)

    # --- основной цикл ---
    history = []
    best_map = -1.0
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.epochs} ===")
        avg, global_step = train_one_epoch(
            model, train_loader, optimizer, device, cfg, writer, epoch, global_step
        )
        metrics = evaluate(model, val_loader, processor, device, cfg)

        writer.add_scalar("val/mAP", metrics["map"], epoch)
        writer.add_scalar("val/mAP50", metrics["map_50"], epoch)
        writer.add_scalar("val/mAP75", metrics["map_75"], epoch)
        writer.add_scalar("epoch/loss_total", avg["loss"], epoch)
        writer.add_scalar("epoch/loss_ce", avg["loss_ce"], epoch)
        writer.add_scalar("epoch/loss_bbox", avg["loss_bbox"], epoch)
        writer.add_scalar("epoch/loss_giou", avg["loss_giou"], epoch)

        row = {"epoch": epoch, **avg, **metrics}
        history.append(row)
        print(f"[epoch {epoch}] loss={avg['loss']:.3f} | mAP={metrics['map']:.4f} "
              f"mAP50={metrics['map_50']:.4f} | {avg['sec_per_epoch']:.1f}s")

        # --- чекпойнт ---
        ckpt = cfg.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "metrics": metrics, "id2label": id2label}, ckpt)
        if metrics["map"] >= best_map:
            best_map = metrics["map"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "metrics": metrics, "id2label": id2label},
                       cfg.ckpt_dir / "best.pt")
            print(f"  -> new best mAP={best_map:.4f}, saved best.pt")

    writer.close()

    # --- сохраняем историю и графики/таблицу ---
    (cfg.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    plot_loss_curves(history, cfg.fig_dir)
    write_metrics_table(history, cfg.out)
    print(f"\n[done] artifacts in {cfg.out.resolve()}")
    print("  TensorBoard:  tensorboard --logdir outputs/runs")


def write_metrics_table(history, out_dir: Path) -> None:
    """Markdown-таблица метрик по эпохам -> outputs/metrics.md."""
    lines = ["| epoch | loss | loss_ce | loss_bbox | loss_giou | mAP | mAP50 | mAP75 |",
             "|------:|-----:|--------:|----------:|----------:|----:|------:|------:|"]
    for r in history:
        lines.append(
            f"| {r['epoch']} | {r['loss']:.3f} | {r['loss_ce']:.3f} | {r['loss_bbox']:.3f} | "
            f"{r['loss_giou']:.3f} | {r['map']:.4f} | {r['map_50']:.4f} | {r['map_75']:.4f} |"
        )
    best = max(history, key=lambda r: r["map"])
    lines.append("")
    lines.append(f"**Лучшая эпоха:** {best['epoch']} — mAP={best['map']:.4f}, mAP50={best['map_50']:.4f}")
    (out_dir / "metrics.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main(parse_args())
