from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes_a[:, :4], 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes_b[:, :4], 4, axis=1)

    x_a = np.maximum(x11, x21.T)
    y_a = np.maximum(y11, y21.T)
    x_b = np.minimum(x12, x22.T)
    y_b = np.minimum(y12, y22.T)

    intersection = np.maximum(0.0, x_b - x_a) * np.maximum(0.0, y_b - y_a)
    area_a = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    area_b = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)
    return intersection / (area_a + area_b.T - intersection + 1e-6)


def bbox_to_measurement(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox[:4]
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    scale = width * height
    ratio = width / max(height, 1e-6)
    return np.array([center_x, center_y, scale, ratio], dtype=np.float32).reshape((4, 1))


def state_to_bbox(state: np.ndarray) -> np.ndarray:
    center_x, center_y, scale, ratio = state[:4].reshape((-1,))
    width = np.sqrt(max(scale * ratio, 0.0))
    height = scale / max(width, 1e-6)
    return np.array(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        dtype=np.float32,
    )


class KalmanBoxTrack:
    next_id = 1

    def __init__(self, bbox: np.ndarray) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_measurement(bbox)

        self.id = KalmanBoxTrack.next_id
        KalmanBoxTrack.next_id += 1
        self.time_since_update = 0
        self.hit_streak = 0
        self.hits = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_measurement(bbox))

    @property
    def bbox(self) -> np.ndarray:
        return state_to_bbox(self.kf.x)


class SortTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[KalmanBoxTrack] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            detections = np.empty((0, 5), dtype=np.float32)
        self.frame_count += 1

        predicted_boxes = np.array([track.predict() for track in self.tracks], dtype=np.float32)
        if predicted_boxes.size == 0:
            predicted_boxes = np.empty((0, 4), dtype=np.float32)

        matched, unmatched_detections, unmatched_tracks = self._associate(
            detections,
            predicted_boxes,
        )

        for detection_index, track_index in matched:
            self.tracks[track_index].update(detections[detection_index])

        for detection_index in unmatched_detections:
            self.tracks.append(KalmanBoxTrack(detections[detection_index]))

        active_tracks = []
        retained_tracks = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                retained_tracks.append(track)
            if track.time_since_update > 0:
                continue
            if track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                active_tracks.append(np.append(track.bbox, track.id))

        self.tracks = retained_tracks
        if not active_tracks:
            return np.empty((0, 5), dtype=np.float32)
        return np.array(active_tracks, dtype=np.float32)

    def _associate(
        self,
        detections: np.ndarray,
        predicted_boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(predicted_boxes) == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.arange(len(detections)),
                np.empty((0,), dtype=np.int32),
            )
        if len(detections) == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.arange(len(predicted_boxes)),
            )

        iou_matrix = iou_batch(detections, predicted_boxes)
        detection_indices, track_indices = linear_sum_assignment(-iou_matrix)

        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(predicted_boxes)))
        for detection_index, track_index in zip(detection_indices, track_indices):
            if iou_matrix[detection_index, track_index] < self.iou_threshold:
                continue
            matches.append([detection_index, track_index])
            unmatched_detections.remove(detection_index)
            unmatched_tracks.remove(track_index)

        return (
            np.array(matches, dtype=np.int32),
            np.array(unmatched_detections, dtype=np.int32),
            np.array(unmatched_tracks, dtype=np.int32),
        )
