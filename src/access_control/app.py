from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from access_control.config import AppConfig, load_config
from access_control.sort_tracker import SortTracker
from access_control.telegram import TelegramNotifier
from access_control.yolo_human import YoloHumanDetector

PERSON_CLASS_ID = 0


class ViolationRegistry:
    def __init__(self, cooldown_seconds: int) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_by_track: dict[int, float] = {}

    def should_alert(self, track_id: int) -> bool:
        now = time.monotonic()
        last_alert = self.last_alert_by_track.get(track_id)
        if last_alert is not None and now - last_alert < self.cooldown_seconds:
            return False
        self.last_alert_by_track[track_id] = now
        return True


class VideoRecorder:
    def __init__(self, enabled: bool, output_dir: Path, fps: float, codec: str) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None

    def write(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = self.output_dir / f"output_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Не удалось открыть файл видео для записи: {self.output_path}")
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO access control system")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/config.example.yaml",
        help="Путь к YAML-конфигу",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Переопределить источник видео: путь к файлу, RTSP URL или индекс камеры",
    )
    return parser.parse_args()


def point_inside_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_zone(frame: np.ndarray, polygon: np.ndarray) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color=(0, 0, 255))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)


def draw_person(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track_id: int,
    in_zone: bool,
) -> None:
    x1, y1, x2, y2 = box
    color = (0, 0, 255) if in_zone else (0, 180, 0)
    label = f"person #{track_id}" if track_id >= 0 else "person"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def save_violation_frame(frame: np.ndarray, output_dir: Path, track_id: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"violation_track_{track_id}_{timestamp}.jpg"
    cv2.imwrite(str(image_path), frame)
    return image_path


def get_track_id(box, fallback_id: int) -> int:
    if box.id is None:
        return fallback_id
    return int(box.id.item())


def detect_source_fps(source: int | str, fallback_fps: float = 25.0) -> float:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return fallback_fps
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps is None or fps <= 1:
        return fallback_fps
    return float(fps)


def handle_track(
    frame: np.ndarray,
    original_frame: np.ndarray,
    polygon: np.ndarray,
    registry: ViolationRegistry,
    notifier: TelegramNotifier,
    config: AppConfig,
    box: tuple[int, int, int, int],
    track_id: int,
) -> None:
    x1, y1, x2, y2 = box
    center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
    in_zone = point_inside_polygon(center_point, polygon)

    draw_person(frame, box, track_id, in_zone)
    cv2.circle(frame, center_point, 4, (255, 255, 255), -1)

    if in_zone and registry.should_alert(track_id):
        alert_frame = frame if config.alerts.draw_preview else original_frame
        image_path = save_violation_frame(alert_frame, config.output.photos_dir, track_id)
        caption = f"Нарушение доступа: человек в запрещенной зоне. Track ID: {track_id}"
        print(f"[ALERT] {caption}. Кадр: {image_path}")
        try:
            notifier.send_photo(image_path, caption)
        except Exception as error:
            print(f"[WARN] Не удалось отправить уведомление в Telegram: {error}")


def create_recorder(config: AppConfig, source: int | str) -> VideoRecorder:
    return VideoRecorder(
        enabled=config.output.save_video,
        output_dir=config.output.videos_dir,
        fps=config.output.video_fps or detect_source_fps(source),
        codec=config.output.video_codec,
    )


def run_ultralytics(config: AppConfig, source: int | str) -> None:
    source = int(config.source) if config.source.isdigit() else config.source
    model = YOLO(config.model)
    polygon = np.array(config.zone.polygon, dtype=np.int32)
    notifier = TelegramNotifier(config.telegram)
    registry = ViolationRegistry(config.alerts.cooldown_seconds)
    recorder = create_recorder(config, source)

    stream = model.track(
        source=source,
        stream=True,
        persist=True,
        conf=config.detection.confidence,
        imgsz=config.detection.image_size,
        classes=[PERSON_CLASS_ID],
        device=config.detection.device,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    try:
        for frame_index, result in enumerate(stream):
            original_frame = result.orig_img.copy()
            frame = original_frame.copy()
            draw_zone(frame, polygon)

            boxes = result.boxes if result.boxes is not None else []
            for detection_index, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                track_id = get_track_id(box, fallback_id=-(frame_index * 1000 + detection_index + 1))
                handle_track(
                    frame=frame,
                    original_frame=original_frame,
                    polygon=polygon,
                    registry=registry,
                    notifier=notifier,
                    config=config,
                    box=(x1, y1, x2, y2),
                    track_id=track_id,
                )

            recorder.write(frame)

            if config.runtime.show_window:
                cv2.imshow(config.runtime.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        recorder.release()
        cv2.destroyAllWindows()
        if recorder.output_path is not None:
            print(f"[INFO] Выходное видео сохранено: {recorder.output_path}")


def run_yolo_human(config: AppConfig, source: int | str) -> None:
    detector = YoloHumanDetector(
        model_path=config.model,
        repo_path=config.yolo_human.repo_path,
        image_size=config.detection.image_size,
        confidence=config.detection.confidence,
        iou_threshold=config.detection.nms_iou,
        device=config.detection.device,
    )
    tracker = SortTracker(
        max_age=config.tracker.max_age,
        min_hits=config.tracker.min_hits,
        iou_threshold=config.tracker.iou_threshold,
    )
    polygon = np.array(config.zone.polygon, dtype=np.int32)
    notifier = TelegramNotifier(config.telegram)
    registry = ViolationRegistry(config.alerts.cooldown_seconds)
    recorder = create_recorder(config, source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source}")

    try:
        while True:
            ok, original_frame = capture.read()
            if not ok:
                break

            frame = original_frame.copy()
            draw_zone(frame, polygon)

            detections = detector.detect(original_frame)
            tracks = tracker.update(detections)
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                handle_track(
                    frame=frame,
                    original_frame=original_frame,
                    polygon=polygon,
                    registry=registry,
                    notifier=notifier,
                    config=config,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    track_id=int(track_id),
                )

            recorder.write(frame)

            if config.runtime.show_window:
                cv2.imshow(config.runtime.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        recorder.release()
        cv2.destroyAllWindows()
        if recorder.output_path is not None:
            print(f"[INFO] Выходное видео сохранено: {recorder.output_path}")


def run(config: AppConfig) -> None:
    source = int(config.source) if config.source.isdigit() else config.source
    if config.model_backend == "ultralytics":
        run_ultralytics(config, source)
    elif config.model_backend == "yolo_human":
        run_yolo_human(config, source)
    else:
        raise ValueError("model_backend должен быть 'ultralytics' или 'yolo_human'")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.source is not None:
        config = AppConfig(
            source=args.source,
            model=config.model,
            model_backend=config.model_backend,
            detection=config.detection,
            tracker=config.tracker,
            yolo_human=config.yolo_human,
            zone=config.zone,
            alerts=config.alerts,
            output=config.output,
            telegram=config.telegram,
            runtime=config.runtime,
        )
    run(config)


if __name__ == "__main__":
    main()
