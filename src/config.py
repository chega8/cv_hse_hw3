"""Конфигурация эксперимента fine-tuning DETR.

Все гиперпараметры собраны здесь и могут быть переопределены из CLI (см. train.py).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch


@dataclass
class Config:
    # --- Данные ---
    dataset_name: str = "cppe-5"          # маленький готовый detection-датасет (COCO-формат боксов)
    train_size: int = 0                    # 0 = весь train split, иначе обрезаем для демо
    val_size: int = 0                      # 0 = весь test/val split
    val_fraction: float = 0.1              # доля train, уходящая в валидацию, если в датасете нет val

    # --- Препроцессинг (DETR image processor) ---
    image_short: int = 480                 # короткая сторона при resize (default DETR = 800, уменьшаем для скорости)
    image_long: int = 800                  # длинная сторона
    augment: bool = True                   # random horizontal flip

    # --- Модель ---
    model_name: str = "facebook/detr-resnet-50"

    # --- Оптимизация ---
    epochs: int = 10
    batch_size: int = 2
    lr: float = 1e-4                       # learning rate для transformer-головы
    lr_backbone: float = 1e-5              # пониженный lr для backbone (ResNet-50)
    weight_decay: float = 1e-4
    grad_clip: float = 0.1                 # клиппинг градиентов (как в оригинальном DETR)

    # --- Инфраструктура ---
    num_workers: int = 0                   # на MPS безопаснее 0
    seed: int = 42
    output_dir: str = "outputs"
    log_every: int = 10                    # шагов между логами в консоль/TensorBoard
    eval_threshold: float = 0.0            # порог уверенности при подаче в метрику mAP (0 = все боксы)
    viz_threshold: float = 0.5             # порог для визуализаций / error analysis
    profile: bool = True                   # снимать ли trace профайлера

    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # --- Пути для артефактов ---
    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    @property
    def ckpt_dir(self) -> Path:
        return self.out / "checkpoints"

    @property
    def tb_dir(self) -> Path:
        return self.out / "runs"

    @property
    def prof_dir(self) -> Path:
        return self.out / "profiler"

    @property
    def fig_dir(self) -> Path:
        return self.out / "figures"

    def make_dirs(self) -> None:
        for d in (self.ckpt_dir, self.tb_dir, self.prof_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_args() -> Config:
    """Собирает Config из аргументов командной строки."""
    cfg = Config()
    p = argparse.ArgumentParser(description="Fine-tuning DETR на маленьком detection-датасете")
    p.add_argument("--dataset-name", default=cfg.dataset_name)
    p.add_argument("--train-size", type=int, default=cfg.train_size)
    p.add_argument("--val-size", type=int, default=cfg.val_size)
    p.add_argument("--image-short", type=int, default=cfg.image_short)
    p.add_argument("--image-long", type=int, default=cfg.image_long)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--model-name", default=cfg.model_name)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--lr-backbone", type=float, default=cfg.lr_backbone)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--output-dir", default=cfg.output_dir)
    p.add_argument("--log-every", type=int, default=cfg.log_every)
    p.add_argument("--no-profile", action="store_true")
    a = p.parse_args()

    return Config(
        dataset_name=a.dataset_name,
        train_size=a.train_size,
        val_size=a.val_size,
        image_short=a.image_short,
        image_long=a.image_long,
        augment=not a.no_augment,
        model_name=a.model_name,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        lr_backbone=a.lr_backbone,
        weight_decay=a.weight_decay,
        grad_clip=a.grad_clip,
        num_workers=a.num_workers,
        seed=a.seed,
        output_dir=a.output_dir,
        log_every=a.log_every,
        profile=not a.no_profile,
    )
