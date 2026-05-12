from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectionConfig:
    confidence: float
    image_size: int
    device: str | None
    nms_iou: float


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int
    min_hits: int
    iou_threshold: float


@dataclass(frozen=True)
class YoloHumanConfig:
    repo_path: Path


@dataclass(frozen=True)
class ZoneConfig:
    polygon: list[tuple[int, int]]


@dataclass(frozen=True)
class AlertsConfig:
    cooldown_seconds: int
    draw_preview: bool


@dataclass(frozen=True)
class OutputConfig:
    photos_dir: Path
    videos_dir: Path
    save_video: bool
    video_fps: float | None
    video_codec: str


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool
    bot_token: str
    chat_id: str


@dataclass(frozen=True)
class RuntimeConfig:
    show_window: bool
    window_name: str


@dataclass(frozen=True)
class AppConfig:
    source: str
    model: str
    model_backend: str
    detection: DetectionConfig
    tracker: TrackerConfig
    yolo_human: YoloHumanConfig
    zone: ZoneConfig
    alerts: AlertsConfig
    output: OutputConfig
    telegram: TelegramConfig
    runtime: RuntimeConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    polygon = raw.get("zone", {}).get("polygon", [])
    if len(polygon) < 3:
        raise ValueError("zone.polygon должен содержать минимум 3 точки")

    alerts_raw = raw.get("alerts", {})
    detection_raw = raw.get("detection", {})
    output_raw = raw.get("output", {})
    tracker_raw = raw.get("tracker", {})
    yolo_human_raw = raw.get("yolo_human", {})
    legacy_output_dir = Path(alerts_raw.get("output_dir", "events"))

    return AppConfig(
        source=str(raw.get("source", "0")),
        model=str(raw.get("model", "yolov8n.pt")),
        model_backend=str(raw.get("model_backend", "ultralytics")),
        detection=DetectionConfig(
            confidence=float(detection_raw.get("confidence", 0.45)),
            image_size=int(detection_raw.get("image_size", 640)),
            device=detection_raw.get("device"),
            nms_iou=float(detection_raw.get("nms_iou", 0.7)),
        ),
        tracker=TrackerConfig(
            max_age=int(tracker_raw.get("max_age", 30)),
            min_hits=int(tracker_raw.get("min_hits", 3)),
            iou_threshold=float(tracker_raw.get("iou_threshold", 0.3)),
        ),
        yolo_human=YoloHumanConfig(
            repo_path=Path(yolo_human_raw.get("repo_path", "third_party/YOLOv8-human")),
        ),
        zone=ZoneConfig(
            polygon=[(int(point[0]), int(point[1])) for point in polygon],
        ),
        alerts=AlertsConfig(
            cooldown_seconds=int(alerts_raw.get("cooldown_seconds", 30)),
            draw_preview=bool(alerts_raw.get("draw_preview", True)),
        ),
        output=OutputConfig(
            photos_dir=Path(output_raw.get("photos_dir", legacy_output_dir / "photos")),
            videos_dir=Path(output_raw.get("videos_dir", legacy_output_dir / "videos")),
            save_video=bool(output_raw.get("save_video", False)),
            video_fps=(
                None
                if output_raw.get("video_fps") is None
                else float(output_raw.get("video_fps"))
            ),
            video_codec=str(output_raw.get("video_codec", "mp4v")),
        ),
        telegram=TelegramConfig(
            enabled=bool(raw.get("telegram", {}).get("enabled", False)),
            bot_token=str(raw.get("telegram", {}).get("bot_token", "")),
            chat_id=str(raw.get("telegram", {}).get("chat_id", "")),
        ),
        runtime=RuntimeConfig(
            show_window=bool(raw.get("runtime", {}).get("show_window", True)),
            window_name=str(raw.get("runtime", {}).get("window_name", "Access Control")),
        ),
    )
