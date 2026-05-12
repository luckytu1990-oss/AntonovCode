# Интеллектуальная система контроля доступа

Простой учебный проект: YOLO детектирует людей на видео или RTSP-потоке, проверяет попадание человека в запрещенную полигональную зону и отправляет кадр нарушения в Telegram.

## Возможности

- источник видео: файл, RTSP URL или веб-камера;
- запрещенная зона задается полигоном в YAML-конфиге;
- детекция людей через YOLO;
- трекинг ByteTrack для подавления повторных уведомлений по одному человеку;
- сохранение кадров нарушений в папку `events`;
- сохранение размеченного выходного видео по флагу в конфиге;
- отправка фото нарушения в Telegram-бота.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

При первом запуске Ultralytics автоматически скачает веса `yolov8n.pt`, если их нет локально.

Если хотите запускать командой `access-control`, установите проект в editable-режиме:

```bash
pip install -e .
```

## Настройка

Скопируйте пример конфига:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Основные поля:

```yaml
source: data/ped.mp4
model: yolov8n.pt
model_backend: ultralytics

zone:
  polygon:
    - [220, 120]
    - [560, 120]
    - [620, 420]
    - [180, 420]

telegram:
  enabled: true
  bot_token: "TOKEN"
  chat_id: "CHAT_ID"

output:
  photos_dir: events/photos
  videos_dir: events/videos
  save_video: true
  video_fps: null
  video_codec: mp4v
```

Для модели `YOLOv8-human` используйте отдельный backend с SORT-трекером:

```bash
git lfs install
git lfs pull
```

```yaml
model: weights/yolov8n-human-crowdhuman.pt
model_backend: yolo_human

detection:
  confidence: 0.45
  image_size: 640
  device: cpu
  nms_iou: 0.7

tracker:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

yolo_human:
  repo_path: third_party/YOLOv8-human
```

Для RTSP:

```yaml
source: rtsp://login:password@192.168.1.10:554/stream1
```

## Запуск

```bash
python -m access_control.app --config configs/config.yaml
```

Можно переопределить источник без изменения конфига:

```bash
python -m access_control.app --config configs/config.yaml --source data/ped.mp4
```

Для запуска из корня проекта без установки пакета используйте:

```bash
PYTHONPATH=src python -m access_control.app --config configs/config.yaml
```

После `pip install -e .` можно так:

```bash
access-control --config configs/config.yaml
```

## Как работает алгоритм

1. YOLO ищет объекты класса `person`.
2. ByteTrack присваивает человеку `track_id`.
3. Для каждого bounding box берется центральная точка.
4. Если точка попала внутрь запрещенного полигона, фиксируется нарушение.
5. Если по этому `track_id` недавно уже было уведомление, новое уведомление не отправляется.
6. Кадр сохраняется в `events/photos` и отправляется в Telegram.
7. Если `output.save_video: true`, размеченный видеопоток сохраняется в `events/videos`.

## Подбор полигона

Координаты полигона задаются в пикселях исходного кадра. Удобнее всего открыть кадр из видео, определить нужные точки и перенести их в `configs/config.yaml`.
