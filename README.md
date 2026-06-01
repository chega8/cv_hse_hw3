# ДЗ 2 — Fine-tuning DETR на маленьком detection-датасете

Реализация **пункта 2** задания «Минимальный детектор на DETR/Deformable-DETR»:
запуск fine-tuning DETR (`facebook/detr-resnet-50`) на готовом detection-датасете
в формате COCO, с полным тренировочным циклом, логированием, профилированием и
error analysis.

> Минимальная подготовка датасета (пункт 1) включена, т.к. без неё fine-tuning не
> запустить. Используется маленький готовый набор **CPPE-5** (1000 train изображений,
> боксы в COCO-формате `[x, y, w, h]`, 5 классов: `Coverall, Face_Shield, Gloves,
> Goggles, Mask`). Пайплайн **dataset-agnostic** — достаточно подставить любой
> COCO-subset (в т.ч. реальный с ≥10 классами) через `--dataset-name`.

## Структура

```
src/
  config.py          # все гиперпараметры + разбор CLI
  dataset.py         # загрузка датасета, COCO-аннотации, аугментация (flip), collate
  model.py           # DETR + image processor, раздельные lr для backbone/головы
  engine.py          # train_one_epoch + evaluate (mAP через torchmetrics)
  train.py           # полный цикл: TensorBoard, профайлер, чекпойнты, графики, метрики
  plots.py           # графики потерь (classification / bbox regression) и mAP
  error_analysis.py  # категоризация ошибок + визуализация боксов
outputs/             # артефакты прогона (создаётся автоматически)
  runs/              #   логи TensorBoard
  checkpoints/       #   чекпойнты эпох + best.pt
  profiler/          #   trace профайлера (chrome trace) + summary.txt
  figures/           #   графики потерь, mAP, предсказания, error analysis
  config.json        #   конфиг прогона
  history.json       #   история loss/метрик по эпохам
  metrics.md         #   таблица метрик
  error_analysis.md  #   таблица ошибок
```

## Установка

```bash
pip install -r requirements.txt
# на старых окружениях могут потребоваться апгрейды:
pip install --upgrade "scipy>=1.10,<1.14" "pillow>=10,<11" timm
```

## Запуск

### Colab (GPU)

Готовая обёртка: [`notebooks/detr_finetune_colab.ipynb`](notebooks/detr_finetune_colab.ipynb) —
прогоняет весь пайплайн на бесплатном GPU T4 (установка, обучение, метрики, графики,
error analysis, TensorBoard, скачивание артефактов). Впишите `REPO_URL` своего форка
в ячейке №2.

### Локально

Короткий демо-прогон (использован в этом репозитории, ~10 мин на Apple M-series / MPS):

```bash
python src/train.py --train-size 160 --val-size 64 --epochs 6 --batch-size 2
python src/error_analysis.py --ckpt outputs/checkpoints/best.pt --val-size 64 --threshold 0.1
tensorboard --logdir outputs/runs
```

> `--threshold` для error analysis намеренно низкий (0.1): после 6 эпох голова
> ещё недоучена, и max score предсказаний ≈ 0.24, поэтому при стандартном 0.5
> боксов не остаётся. На полном прогоне порог поднимают до 0.5–0.7.

Полноценный прогон (весь датасет, дольше — лучше на GPU/Colab):

```bash
python src/train.py --epochs 30 --image-short 800 --image-long 1333
```

## Эксперимент и гиперпараметры

| Параметр              | Значение (демо)            | Комментарий |
|-----------------------|----------------------------|-------------|
| Модель                | `facebook/detr-resnet-50`  | DETR, ResNet-50 backbone, 100 object queries |
| Голова классификации  | переинициализирована        | `num_labels=5` (+ no-object), `ignore_mismatched_sizes` |
| Оптимизатор           | AdamW                       | `weight_decay=1e-4` |
| LR (transformer)      | `1e-4`                      | голова/энкодер/декодер |
| LR (backbone)         | `1e-5`                      | пониженный, как в оригинальном DETR |
| Grad clip             | `0.1`                       | как в DETR |
| Resize                | short=480, long=800         | уменьшен с 800/1333 ради скорости на MPS |
| Аугментация           | random horizontal flip      | с коррекцией боксов |
| Batch size            | 2                           | ограничение памяти MPS |
| Эпохи                 | 6 (демо)                    | для полного качества нужно 30–50 |
| Устройство            | MPS (Apple Silicon)         | авто-выбор cuda→mps→cpu |

**Loss DETR** (Hungarian matching + set prediction loss) состоит из:
- `loss_ce` — **classification** (cross-entropy по сопоставленным queries);
- `loss_bbox` — **bbox regression**, L1 по координатам;
- `loss_giou` — generalized IoU loss (тоже часть локализации).

## Метрики

Таблица по эпохам — `outputs/metrics.md` (генерируется автоматически).
Метрики считаются через `torchmetrics` `MeanAveragePrecision` (COCO-style):

| Метрика | Что это |
|---------|---------|
| mAP@[.5:.95] | основная COCO-метрика |
| mAP@0.5 | mAP при IoU≥0.5 |
| mAP@0.75 | строгий порог |

**Результаты демо-прогона** (160 train / 64 val, 6 эпох, resize 480/800, MPS):

| epoch | loss | loss_ce | loss_bbox | loss_giou | mAP | mAP50 | mAP75 |
|------:|-----:|--------:|----------:|----------:|----:|------:|------:|
| 1 | 3.203 | 1.641 | 0.124 | 0.472 | 0.0113 | 0.0272 | 0.0100 |
| 2 | 3.153 | 1.595 | 0.124 | 0.469 | 0.0113 | 0.0260 | 0.0098 |
| 3 | 3.014 | 1.549 | 0.115 | 0.444 | 0.0148 | 0.0357 | 0.0110 |
| 4 | 2.776 | 1.406 | 0.107 | 0.416 | 0.0257 | 0.0493 | 0.0220 |
| 5 | 2.492 | 1.153 | 0.105 | 0.409 | 0.0457 | 0.0778 | 0.0444 |
| **6** | **2.360** | **1.058** | **0.102** | **0.397** | **0.0631** | **0.1057** | **0.0635** |

За 6 эпох total loss упал 3.20→2.36, classification loss 1.64→1.06, а **mAP вырос в ~5.6×**
(0.011→0.063), mAP50 0.027→0.106 — монотонный рост, пайплайн обучается корректно.
Значения скромные по абсолюту (мало эпох, маленький сабсет, уменьшенный resize) —
это иллюстрация рабочего пайплайна, а не SOTA. Графики: `outputs/figures/loss_and_map.png`.

## Графики потерь

`outputs/figures/loss_and_map.png` — два сабплота:
1. компоненты loss по эпохам: total, classification (`loss_ce`),
   bbox regression (`loss_bbox` L1 + `loss_giou`);
2. валидационные mAP / mAP@0.5.

Те же кривые доступны по шагам в TensorBoard (`train/loss_*`, `val/mAP*`).

## Профайлер

`torch.profiler` снимает trace нескольких train-шагов:
- `outputs/profiler/detr_trace.json` — chrome trace (открыть в `chrome://tracing`
  или Perfetto);
- `outputs/profiler/summary.txt` — топ операций по CPU time.

На MPS видно, что время доминируют свёртки backbone (`mps_convolution_backward`),
`upsample_nearest2d` (интерполяция в backbone/маски) и `bmm` (attention в
энкодере/декодере). Часть операций уходит в CPU-fallback (`PYTORCH_ENABLE_MPS_FALLBACK`).

## Error analysis

`src/error_analysis.py` прогоняет лучшую модель по валидации, сопоставляет
предсказания с GT по IoU и раскладывает на категории (подход в духе TIDE):

| Категория | Условие |
|-----------|---------|
| **correct** | IoU≥0.5 и класс верный |
| **localization error** | класс верный, но 0.1≤IoU<0.5 (плохая локализация) |
| **classification error** | IoU≥0.5, но класс неверный |
| **duplicate** | дубль на уже найденный объект |
| **background FP** | предсказание без объекта (IoU<0.1) |
| **missed GT (FN)** | объект не найден |

Артефакты:
- `outputs/error_analysis.md` / `.json` — сводная таблица;
- `outputs/figures/error_analysis.png` — bar chart категорий;
- `outputs/figures/predictions.png` — примеры: **красные** боксы = предсказания
  (с классом и score), **зелёные пунктирные** = ground truth.

**Результаты на демо-модели** (val=64, threshold=0.1):

| Категория | Кол-во | % предсказаний |
|-----------|-------:|---------------:|
| correct (класс+локализация) | 55 | 5.7% |
| **localization error** (0.1≤IoU<0.5) | **440** | **45.3%** |
| classification error (IoU≥0.5, класс неверный) | 105 | 10.8% |
| duplicate | 79 | 8.1% |
| background FP (IoU<0.1) | 293 | 30.1% |
| missed GT (FN) | 152 | — |

**Вывод:** доминирует **ошибка локализации (45%)** — модель уже находит объекты, но
рисует боксы неточно (IoU<0.5). Это типично для недообученного DETR: regression-ветви
и Hungarian matching ещё не сошлись. Второй по величине источник — **background FP (30%)**
из-за низкого порога 0.1 (голова классификации недоучена, score размазаны). Ошибок
*классификации* сравнительно мало (11%) — когда бокс точный, класс чаще верный.
**Что улучшит метрики:** больше эпох, полный resize (800/1333), больше данных,
повышение порога уверенности по мере дообучения.

**Как читать:** преобладание *localization error* означает, что модель находит
объекты, но неточно их локализует (нужно дольше учить регрессию боксов / больше
эпох); преобладание *classification error* — путает классы (например, `Mask` vs
`Face_Shield`); много *background FP* — слишком низкий порог уверенности или
недоученная голова; много *missed GT* — низкий recall (редкие/мелкие объекты).

## Замечания / ограничения

- Прогон выполнен на CPU/MPS без CUDA, поэтому resize уменьшен и взят небольшой
  сабсет — это **демонстрация работоспособного пайплайна**, метрики намеренно
  скромные. Для полноценных чисел запустите полный прогон на GPU.
- Для перехода на **Deformable-DETR** достаточно заменить класс модели на
  `DeformableDetrForObjectDetection` и `--model-name SenseTime/deformable-detr`
  (остальной пайплайн совместим).
- Для реального **COCO-subset ≥10 классов** подставьте свой датасет в формате HF
  `datasets` с полем `objects` (`bbox` в COCO `[x,y,w,h]`, `category` как ClassLabel).
