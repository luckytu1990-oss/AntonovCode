from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


class YoloHumanDetector:
    def __init__(
        self,
        model_path: str,
        repo_path: Path,
        image_size: int,
        confidence: float,
        iou_threshold: float,
        device: str | None,
    ) -> None:
        if not repo_path.exists():
            raise FileNotFoundError(
                f"Не найден репозиторий YOLOv8-human: {repo_path}. "
                "Скачайте его в third_party/YOLOv8-human."
            )
        sys.path.insert(0, str(repo_path))

        from utils import util  # type: ignore
        from utils.dataset import resize  # type: ignore

        self.resize = resize
        self.non_max_suppression = util.non_max_suppression
        self.image_size = image_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = torch.device(device or "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = checkpoint["model"].float().to(self.device).eval()

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> np.ndarray:
        resized_frame, ratio, pad = self.resize(frame, self.image_size, False)
        sample = resized_frame.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)
        tensor = torch.from_numpy(sample).unsqueeze(0).float().to(self.device) / 255.0

        outputs = self.model(tensor)
        detections = self.non_max_suppression(outputs, self.confidence, self.iou_threshold)[0]
        if detections.numel() == 0:
            return np.empty((0, 5), dtype=np.float32)

        detections_np = detections.detach().cpu().numpy()
        boxes = []
        frame_height, frame_width = frame.shape[:2]
        for x1, y1, x2, y2, confidence, _class_id in detections_np:
            x1 = int((x1 - pad[0]) / ratio[0])
            x2 = int((x2 - pad[0]) / ratio[0])
            y1 = int((y1 - pad[1]) / ratio[1])
            y2 = int((y2 - pad[1]) / ratio[1])
            boxes.append(
                [
                    np.clip(x1, 0, frame_width - 1),
                    np.clip(y1, 0, frame_height - 1),
                    np.clip(x2, 0, frame_width - 1),
                    np.clip(y2, 0, frame_height - 1),
                    float(confidence),
                ]
            )
        return np.array(boxes, dtype=np.float32)
