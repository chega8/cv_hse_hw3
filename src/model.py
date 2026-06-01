"""Сборка модели DETR и image processor под наш набор классов."""
from __future__ import annotations

from transformers import AutoImageProcessor, DetrForObjectDetection


def build_processor(cfg):
    """DetrImageProcessor с уменьшенным размером ресайза (для скорости на MPS/CPU)."""
    processor = AutoImageProcessor.from_pretrained(
        cfg.model_name,
        size={"shortest_edge": cfg.image_short, "longest_edge": cfg.image_long},
        use_fast=False,
    )
    return processor


def build_model(cfg, id2label: dict, label2id: dict) -> DetrForObjectDetection:
    """Загружает предобученный DETR, заменяя голову классификации под наши классы."""
    model = DetrForObjectDetection.from_pretrained(
        cfg.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # переинициализируем classification head
    )
    return model


def param_groups(model: DetrForObjectDetection, cfg):
    """Раздельные learning rate для backbone и остальной сети (как в DETR)."""
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": other_params, "lr": cfg.lr},
        {"params": backbone_params, "lr": cfg.lr_backbone},
    ]
