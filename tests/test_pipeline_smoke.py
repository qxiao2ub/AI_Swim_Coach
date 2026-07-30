"""Run with: python tests/test_pipeline_smoke.py"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ai_pipeline as pipeline


def synthetic_landmark_frame(frame_count: int = 120) -> pd.DataFrame:
    rows = []
    for index in range(frame_count):
        phase = 2.0 * math.pi * index / 30.0
        row = {
            "frame_index": index,
            "processed_frame_index": index,
            "timestamp_sec": index / 30.0,
            "frame_width": 640,
            "frame_height": 360,
            "pose_detected": 1,
        }
        base_points = {
            "nose": (0.30, 0.42),
            "left_shoulder": (0.38, 0.46),
            "right_shoulder": (0.42, 0.48),
            "left_elbow": (0.49 + 0.05 * math.sin(phase), 0.42),
            "right_elbow": (0.31 - 0.05 * math.sin(phase), 0.52),
            "left_wrist": (0.58 + 0.08 * math.sin(phase), 0.38),
            "right_wrist": (0.22 - 0.08 * math.sin(phase), 0.56),
            "left_hip": (0.56, 0.50),
            "right_hip": (0.60, 0.52),
            "left_knee": (0.71, 0.49 + 0.03 * math.sin(phase)),
            "right_knee": (0.74, 0.53 - 0.03 * math.sin(phase)),
            "left_ankle": (0.88, 0.47 + 0.05 * math.sin(phase)),
            "right_ankle": (0.90, 0.55 - 0.05 * math.sin(phase)),
        }
        for landmark_index, name in enumerate(pipeline.LANDMARK_NAMES):
            x_value, y_value = base_points.get(
                name,
                (0.45 + 0.001 * landmark_index, 0.50 + 0.001 * landmark_index),
            )
            row[f"{name}_x"] = x_value
            row[f"{name}_y"] = y_value
            row[f"{name}_z"] = 0.01 * math.sin(phase)
            row[f"{name}_visibility"] = 0.95
            row[f"{name}_px"] = x_value * 640
            row[f"{name}_py"] = y_value * 360
            row[f"{name}_world_x"] = x_value
            row[f"{name}_world_y"] = y_value
            row[f"{name}_world_z"] = 0.01 * math.sin(phase)
            row[f"{name}_world_visibility"] = 0.95
        rows.append(row)
    return pd.DataFrame(rows)


class FakeMP:
    class ImageFormat:
        SRGB = "SRGB"

    class Image:
        def __init__(self, image_format, data):
            self.image_format = image_format
            self.data = data


class FakePose:
    def __init__(self):
        self.counter = 0

    def detect_for_video(self, image, timestamp_ms):
        del image, timestamp_ms
        phase = 2.0 * math.pi * self.counter / 10.0
        self.counter += 1
        landmarks = []
        for index in range(33):
            landmarks.append(
                SimpleNamespace(
                    x=0.25 + 0.015 * index + 0.002 * math.sin(phase),
                    y=0.35 + 0.006 * index + 0.002 * math.cos(phase),
                    z=0.0,
                    visibility=0.9,
                )
            )
        return SimpleNamespace(
            pose_landmarks=[landmarks],
            pose_world_landmarks=[landmarks],
        )

    def close(self):
        return None


def test_fake_video_extraction(work_dir: Path) -> None:
    video_path = work_dir / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(12):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(frame, str(index), (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()

    original_factory = pipeline._create_pose_landmarker
    pipeline._create_pose_landmarker = lambda *args, **kwargs: (FakeMP, FakePose())
    try:
        raw, annotated, metadata = pipeline.extract_pose_timeseries(
            video_path,
            output_dir=work_dir / "video_outputs",
            max_frames=6,
            stride=2,
            resize_width=320,
            model_path=work_dir / "unused.task",
        )
    finally:
        pipeline._create_pose_landmarker = original_factory

    assert len(raw) == 6
    assert metadata["pose_detection_rate"] == 1.0
    assert annotated and Path(annotated).exists()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="swim_coach_test_") as temporary:
        work_dir = Path(temporary)
        test_fake_video_extraction(work_dir)

        raw = synthetic_landmark_frame()
        features = pipeline.add_kinematic_features(raw)
        metadata = {
            "video_name": "synthetic.mp4",
            "processed_frames": len(features),
            "pose_detected_frames": len(features),
            "pose_detection_rate": 1.0,
        }
        metrics = pipeline.summarize_video(features, metadata)
        recommendations = pipeline.generate_recommendations(metrics)
        assert recommendations
        assert "body_line_deviation" in features.columns
        assert "wrist_symmetry_gap" in features.columns

        labeled = pipeline.create_demo_training_labels(features, n_labels=3)
        assert labeled["demo_label"].nunique() >= 2

        training = pipeline.train_supervised_models(
            labeled,
            output_dir=work_dir / "models",
            include_optional_boosters=False,
        )
        assert training["models"]
        assert Path(training["model_bundle_path"]).exists()

        x_sequence, y_sequence, feature_columns = pipeline.make_sequence_windows(
            labeled,
            label_col="demo_label",
            window_size=30,
            step=10,
        )
        assert x_sequence.ndim == 3
        assert len(x_sequence) == len(y_sequence)
        assert x_sequence.shape[2] == len(feature_columns)

        saved = pipeline.save_outputs(
            work_dir / "outputs",
            "synthetic.mp4",
            raw,
            features,
            metrics,
            recommendations,
        )
        assert Path(saved["analysis_bundle_zip"]).exists()

        feedback = pipeline.append_feedback(
            work_dir / "data" / "feedback.csv",
            video_name="synthetic.mp4",
            action_id=recommendations[0]["action_id"],
            rating=4,
            helpful=True,
            comment="Smoke test",
            metrics=metrics,
        )
        state = pipeline.demo_contextual_bandit_update(feedback)
        assert state["model_updated"] is True

    print("AI Swimming Coach pipeline smoke test passed.")


if __name__ == "__main__":
    main()
