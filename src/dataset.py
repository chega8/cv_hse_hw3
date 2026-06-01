"""Загрузка и препроцессинг detection-датасета под формат DETR.

Используется маленький готовый набор `cppe-5` (1000 train изображений, 5 классов,
боксы в COCO-формате [x, y, w, h]). Код dataset-agnostic: достаточно, чтобы у
датасета было поле `objects` с ключами `bbox`/`category`/`area` — тогда можно
подставить любой COCO-subset (в т.ч. реальный с >=10 классами).
"""
from __future__ import annotations

import random
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def load_splits(cfg) -> tuple[Any, Any, dict, dict]:
    """Возвращает (train_ds, val_ds, id2label, label2id).

    Если в датасете есть только train — отрезаем часть в валидацию.
    """
    ds = load_dataset(cfg.dataset_name)

    # Определяем train/val сплиты
    if "validation" in ds:
        train_raw, val_raw = ds["train"], ds["validation"]
    elif "test" in ds and len(ds["test"]) >= 10:
        train_raw, val_raw = ds["train"], ds["test"]
    else:
        split = ds["train"].train_test_split(test_size=cfg.val_fraction, seed=cfg.seed)
        train_raw, val_raw = split["train"], split["test"]

    # Обрезка для быстрого демо
    if cfg.train_size and cfg.train_size < len(train_raw):
        train_raw = train_raw.select(range(cfg.train_size))
    if cfg.val_size and cfg.val_size < len(val_raw):
        val_raw = val_raw.select(range(cfg.val_size))

    # Имена классов из ClassLabel (datasets 4.x: objects — dict; старые версии: Sequence)
    obj_feat = train_raw.features["objects"]
    cat_feat = obj_feat.feature["category"] if hasattr(obj_feat, "feature") else obj_feat["category"]
    cat_feat = getattr(cat_feat, "feature", cat_feat)  # разворачиваем List(ClassLabel)
    categories = cat_feat.names
    id2label = {i: n for i, n in enumerate(categories)}
    label2id = {n: i for i, n in id2label.items()}
    return train_raw, val_raw, id2label, label2id


def _hflip(image, objects):
    """Горизонтальный флип PIL-изображения и боксов [x, y, w, h]."""
    W = image.width
    image = image.transpose(0)  # PIL.Image.FLIP_LEFT_RIGHT == 0
    new_boxes = []
    for (x, y, w, h) in objects["bbox"]:
        new_x = W - (x + w)
        new_boxes.append([new_x, y, w, h])
    objects = dict(objects)
    objects["bbox"] = new_boxes
    return image, objects


def _format_anns(image_id: int, objects: dict) -> dict:
    """Преобразует записи датасета в COCO-аннотации для DetrImageProcessor."""
    anns = []
    for i in range(len(objects["bbox"])):
        anns.append(
            {
                "image_id": image_id,
                "category_id": int(objects["category"][i]),
                "bbox": list(objects["bbox"][i]),
                "area": float(objects["area"][i]) if "area" in objects else
                        float(objects["bbox"][i][2] * objects["bbox"][i][3]),
                "iscrowd": 0,
            }
        )
    return {"image_id": image_id, "annotations": anns}


class DetrDataset(Dataset):
    """Оборачивает HF-датасет: на выходе pixel_values + labels в формате DETR."""

    def __init__(self, hf_ds, processor, augment: bool = False):
        self.ds = hf_ds
        self.processor = processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        ex = self.ds[idx]
        image = ex["image"].convert("RGB")
        objects = ex["objects"]
        image_id = ex.get("image_id", idx)

        if self.augment and random.random() < 0.5 and len(objects["bbox"]) > 0:
            image, objects = _hflip(image, objects)

        target = _format_anns(image_id, objects)
        enc = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "labels": enc["labels"][0],
        }


def make_collate_fn(processor):
    """collate_fn: паддинг pixel_values до общего размера + список labels."""

    def collate(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": [item["labels"] for item in batch],
        }

    return collate


def labels_to_xyxy(label: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """GT из формата DETR (cxcywh, нормализованные) -> (boxes_xyxy_abs, class_ids)."""
    boxes = label["boxes"]            # (N, 4) cxcywh в [0, 1]
    h, w = label["orig_size"].tolist()
    if boxes.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - 0.5 * bw) * w
    y1 = (cy - 0.5 * bh) * h
    x2 = (cx + 0.5 * bw) * w
    y2 = (cy + 0.5 * bh) * h
    return torch.stack([x1, y1, x2, y2], dim=-1), label["class_labels"]
