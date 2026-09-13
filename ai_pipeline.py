"""Core AI pipeline for the AI Swimming Coach Streamlit prototype.

Author: Jasper Ding
Advisor: Dr. Qingyang Xiao

The module is deliberately framework-independent so the same functions can be
used from Streamlit, Colab, scripts, and tests.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LANDMARK_NAMES: List[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
    "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

ANGLE_SPECS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
}

DEFAULT_FEATURE_COLUMNS: List[str] = [
    "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle",
    "right_shoulder_angle", "left_hip_angle", "right_hip_angle",
    "left_knee_angle", "right_knee_angle", "body_line_angle",
    "body_line_deviation", "head_alignment_score", "kick_amplitude",
    "left_wrist_speed", "right_wrist_speed", "left_ankle_speed",
    "right_ankle_speed", "wrist_symmetry_gap", "ankle_symmetry_gap",
    "elbow_angle_symmetry_gap", "knee_angle_symmetry_gap",
    "mean_landmark_visibility",
]

FEEDBACK_ACTIONS: List[Dict[str, str]] = [
    {
        "action_id": "streamline_body_line",
        "title": "Improve body line",
        "drill": "Use a six-kick streamline drill with a stable head position.",
    },
    {
        "action_id": "left_right_symmetry",
        "title": "Balance left-right symmetry",
        "drill": "Use a 3-3-3 drill and compare both sides at low speed.",
    },
    {
        "action_id": "breathing_alignment",
        "title": "Reduce head lift",
        "drill": "Use a one-goggle breathing drill and keep the neck long.",
    },
    {
        "action_id": "compact_kick",
        "title": "Use a more compact kick",
        "drill": "Practice vertical kicking, then transfer the same compact motion.",
    },
    {
        "action_id": "camera_quality",
        "title": "Improve recording quality",
        "drill": "Record a side view with the full body visible and minimal glare.",
    },
]

POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

ProgressCallback = Callable[[float, str], None]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_stem(name: str) -> str:
    value = Path(name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "swim_video"


def _safe_import_cv2():
    import cv2

    return cv2


def ensure_pose_model(model_path: str | Path) -> Path:
    """Download the official MediaPipe lite pose model when not cached."""
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 100_000:
        return path

    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        request = urllib.request.Request(
            POSE_MODEL_URL,
            headers={"User-Agent": "AI-Swim-Coach/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if temporary.stat().st_size <= 100_000:
            raise RuntimeError("Downloaded pose model is unexpectedly small.")
        temporary.replace(path)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "The MediaPipe pose model could not be downloaded. Check the app's "
            "internet access and restart the analysis."
        ) from exc
    return path


def _create_pose_landmarker(
    model_path: str | Path,
    min_detection_confidence: float,
    min_tracking_confidence: float,
):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model = ensure_pose_model(model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


def _clip_resize(frame: np.ndarray, resize_width: Optional[int]) -> np.ndarray:
    if not resize_width or resize_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame
    cv2 = _safe_import_cv2()
    new_height = int(round(height * resize_width / width))
    return cv2.resize(frame, (int(resize_width), new_height), interpolation=cv2.INTER_AREA)


def _point_from_row(row: pd.Series, landmark: str, use_world: bool = False) -> np.ndarray:
    prefix = f"{landmark}_world" if use_world else landmark
    return np.asarray(
        [row.get(f"{prefix}_x", np.nan), row.get(f"{prefix}_y", np.nan), row.get(f"{prefix}_z", np.nan)],
        dtype=float,
    )


def angle_degrees(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    vector_a = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    vector_c = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    if np.any(~np.isfinite(vector_a)) or np.any(~np.isfinite(vector_c)):
        return float("nan")
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_c))
    if denominator <= 1e-12:
        return float("nan")
    cosine = float(np.dot(vector_a, vector_c) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _midpoint(row: pd.Series, first: str, second: str) -> np.ndarray:
    a = _point_from_row(row, first)
    b = _point_from_row(row, second)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return np.full(3, np.nan)
    return (a + b) / 2.0


def _landmark_row(
    landmarks: Any,
    world_landmarks: Any,
    frame_index: int,
    processed_index: int,
    timestamp_sec: float,
    frame_width: int,
    frame_height: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "frame_index": int(frame_index),
        "processed_frame_index": int(processed_index),
        "timestamp_sec": float(timestamp_sec),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "pose_detected": 0,
    }
    for name in LANDMARK_NAMES:
        for suffix in ("x", "y", "z", "visibility"):
            row[f"{name}_{suffix}"] = np.nan
            row[f"{name}_world_{suffix}"] = np.nan
        row[f"{name}_px"] = np.nan
        row[f"{name}_py"] = np.nan

    if landmarks is None:
        return row

    row["pose_detected"] = 1
    for index, name in enumerate(LANDMARK_NAMES):
        if index >= len(landmarks):
            break
        landmark = landmarks[index]
        row[f"{name}_x"] = float(landmark.x)
        row[f"{name}_y"] = float(landmark.y)
        row[f"{name}_z"] = float(landmark.z)
        row[f"{name}_visibility"] = float(getattr(landmark, "visibility", 0.0) or 0.0)
        row[f"{name}_px"] = float(landmark.x * frame_width)
        row[f"{name}_py"] = float(landmark.y * frame_height)

    if world_landmarks is not None:
        for index, name in enumerate(LANDMARK_NAMES):
            if index >= len(world_landmarks):
                break
            landmark = world_landmarks[index]
            row[f"{name}_world_x"] = float(landmark.x)
            row[f"{name}_world_y"] = float(landmark.y)
            row[f"{name}_world_z"] = float(landmark.z)
            row[f"{name}_world_visibility"] = float(
                getattr(landmark, "visibility", 0.0) or 0.0
            )
    return row


def _draw_landmarks(frame: np.ndarray, landmarks: Any) -> None:
    cv2 = _safe_import_cv2()
    height, width = frame.shape[:2]
    points = [
        (int(round(landmark.x * width)), int(round(landmark.y * height)))
        for landmark in landmarks
    ]
    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (0, 220, 255), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 80, 255), -1, cv2.LINE_AA)


def get_ffmpeg_executable() -> Optional[str]:
    """Return a usable FFmpeg executable without requiring an apt package.

    Streamlit Community Cloud can have mixed Debian repositories during base
    image transitions. Installing ``ffmpeg`` through ``packages.txt`` may then
    fail before Python dependencies are processed. The ``imageio-ffmpeg``
    wheel ships a platform-specific executable, so it is preferred as the
    portable fallback.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg and Path(bundled_ffmpeg).is_file():
            return str(bundled_ffmpeg)
    except (ImportError, OSError, RuntimeError):
        return None
    return None


def _transcode_h264(input_path: Path, output_path: Path) -> Optional[Path]:
    ffmpeg_executable = get_ffmpeg_executable()
    if not ffmpeg_executable:
        return None
    command = [
        ffmpeg_executable,
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, timeout=300)
    except (subprocess.SubprocessError, OSError):
        output_path.unlink(missing_ok=True)
        return None
    return output_path if output_path.exists() and output_path.stat().st_size > 0 else None


def extract_pose_timeseries(
    video_path: str | Path,
    output_dir: str | Path = "outputs",
    max_frames: int = 600,
    stride: int = 2,
    resize_width: int = 720,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    create_annotated_video: bool = True,
    model_path: str | Path | None = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, Optional[str], Dict[str, Any]]:
    """Extract MediaPipe pose landmarks and create an annotated video.

    ``max_frames`` means the maximum number of processed frames after applying
    ``stride``. The callback receives a value in [0, 1] and a status message.
    """
    del model_complexity
    cv2 = _safe_import_cv2()
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(
            f"Could not open video '{video_path.name}'. Convert it to H.264 MP4 and try again."
        )

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(raw_fps) or raw_fps <= 1e-3:
        raw_fps = 30.0
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = total_source_frames / raw_fps if total_source_frames > 0 else 0.0
    estimated_processed = (
        min(max_frames, int(math.ceil(total_source_frames / stride)))
        if total_source_frames > 0 else max_frames
    )

    stem = safe_stem(video_path.name)
    raw_annotated_path = output_dir / f"{stem}_annotated_raw.mp4"
    h264_annotated_path = output_dir / f"{stem}_annotated.mp4"
    rows: List[Dict[str, Any]] = []
    writer = None
    raw_frame_index = 0

    if model_path is None:
        model_path = Path.home() / ".cache" / "ai_swim_coach" / "pose_landmarker_lite.task"

    if progress_callback:
        progress_callback(0.02, "Preparing the pose model")
    mp, pose = _create_pose_landmarker(
        model_path,
        min_detection_confidence,
        min_tracking_confidence,
    )

    try:
        while len(rows) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            current_index = raw_frame_index
            raw_frame_index += 1
            if current_index % stride != 0:
                continue

            frame = _clip_resize(frame, resize_width)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(round(current_index * 1000.0 / raw_fps))
            result = pose.detect_for_video(mp_image, timestamp_ms)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            world_landmarks = (
                result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            )
            rows.append(
                _landmark_row(
                    landmarks,
                    world_landmarks,
                    current_index,
                    len(rows),
                    current_index / raw_fps,
                    width,
                    height,
                )
            )

            if create_annotated_video:
                if writer is None:
                    output_fps = max(raw_fps / stride, 1.0)
                    writer = cv2.VideoWriter(
                        str(raw_annotated_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        writer = None
                        raise RuntimeError("OpenCV could not create the annotated video.")
                annotated_frame = frame.copy()
                if landmarks is not None:
                    _draw_landmarks(annotated_frame, landmarks)
                writer.write(annotated_frame)

            if progress_callback and len(rows) % 5 == 0:
                ratio = min(len(rows) / max(estimated_processed, 1), 1.0)
                progress_callback(0.05 + 0.83 * ratio, f"Processing frame {len(rows)}")
    finally:
        pose.close()
        capture.release()
        if writer is not None:
            writer.release()

    if not rows:
        raise ValueError("No decodable frames were found in the uploaded video.")

    annotated_path: Optional[Path] = None
    if create_annotated_video and raw_annotated_path.exists():
        if progress_callback:
            progress_callback(0.91, "Optimizing the annotated video for the browser")
        transcoded = _transcode_h264(raw_annotated_path, h264_annotated_path)
        annotated_path = transcoded or raw_annotated_path
        if transcoded:
            raw_annotated_path.unlink(missing_ok=True)

    dataframe = pd.DataFrame(rows)
    pose_detected_frames = int(dataframe["pose_detected"].sum())
    metadata: Dict[str, Any] = {
        "video_name": video_path.name,
        "source_fps": raw_fps,
        "source_width": source_width,
        "source_height": source_height,
        "source_frame_count": total_source_frames,
        "source_duration_sec": duration_sec,
        "processed_frames": int(len(dataframe)),
        "pose_detected_frames": pose_detected_frames,
        "pose_detection_rate": float(pose_detected_frames / len(dataframe)),
        "stride": int(stride),
        "resize_width": int(resize_width),
    }
    if progress_callback:
        progress_callback(0.96, "Calculating time-series features")
    return dataframe, str(annotated_path) if annotated_path else None, metadata


def _series_speed(out: pd.DataFrame, landmark: str) -> pd.Series:
    coordinate_columns = [f"{landmark}_x", f"{landmark}_y", f"{landmark}_z"]
    if not all(column in out.columns for column in coordinate_columns):
        return pd.Series(np.nan, index=out.index, dtype=float)
    coordinates = out[coordinate_columns].apply(pd.to_numeric, errors="coerce")
    time_values = pd.to_numeric(out["timestamp_sec"], errors="coerce")
    delta_position = coordinates.diff()
    delta_time = time_values.diff().replace(0, np.nan)
    return np.sqrt((delta_position ** 2).sum(axis=1)) / delta_time


def add_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()

    for name, (first, center, third) in ANGLE_SPECS.items():
        out[name] = out.apply(
            lambda row: angle_degrees(
                _point_from_row(row, first),
                _point_from_row(row, center),
                _point_from_row(row, third),
            ),
            axis=1,
        )

    body_angles: List[float] = []
    body_deviations: List[float] = []
    head_scores: List[float] = []
    kick_amplitudes: List[float] = []
    visibility_values: List[float] = []

    for _, row in out.iterrows():
        shoulder_mid = _midpoint(row, "left_shoulder", "right_shoulder")
        hip_mid = _midpoint(row, "left_hip", "right_hip")
        left_ankle = _point_from_row(row, "left_ankle")
        right_ankle = _point_from_row(row, "right_ankle")
        nose = _point_from_row(row, "nose")

        if np.all(np.isfinite(shoulder_mid)) and np.all(np.isfinite(hip_mid)):
            vector = shoulder_mid - hip_mid
            angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
            absolute = abs(angle) % 180.0
            deviation = min(absolute, 180.0 - absolute)
            torso_length = float(np.linalg.norm(vector[:2]))
        else:
            angle = float("nan")
            deviation = float("nan")
            torso_length = float("nan")
        body_angles.append(angle)
        body_deviations.append(deviation)

        if (
            np.all(np.isfinite(nose))
            and np.all(np.isfinite(shoulder_mid))
            and np.isfinite(torso_length)
            and torso_length > 1e-6
        ):
            head_offset = abs(float(nose[1] - shoulder_mid[1])) / torso_length
            head_score = 100.0 * (1.0 - min(head_offset / 1.25, 1.0))
        else:
            head_score = float("nan")
        head_scores.append(head_score)

        if (
            np.all(np.isfinite(left_ankle))
            and np.all(np.isfinite(right_ankle))
            and np.isfinite(torso_length)
            and torso_length > 1e-6
        ):
            kick_amplitude = abs(float(left_ankle[1] - right_ankle[1])) / torso_length
        else:
            kick_amplitude = float("nan")
        kick_amplitudes.append(kick_amplitude)

        visibility_columns = [
            f"{name}_visibility" for name in LANDMARK_NAMES
            if f"{name}_visibility" in out.columns
        ]
        visibility = pd.to_numeric(row[visibility_columns], errors="coerce").mean()
        visibility_values.append(float(visibility) if pd.notna(visibility) else float("nan"))

    out["body_line_angle"] = body_angles
    out["body_line_deviation"] = body_deviations
    out["head_alignment_score"] = head_scores
    out["kick_amplitude"] = kick_amplitudes
    out["mean_landmark_visibility"] = visibility_values

    for landmark in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        out[f"{landmark}_speed"] = _series_speed(out, landmark)

    out["wrist_symmetry_gap"] = (
        out["left_wrist_speed"] - out["right_wrist_speed"]
    ).abs()
    out["ankle_symmetry_gap"] = (
        out["left_ankle_speed"] - out["right_ankle_speed"]
    ).abs()
    out["elbow_angle_symmetry_gap"] = (
        out["left_elbow_angle"] - out["right_elbow_angle"]
    ).abs()
    out["knee_angle_symmetry_gap"] = (
        out["left_knee_angle"] - out["right_knee_angle"]
    ).abs()

    feature_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in out.columns]
    if feature_columns:
        out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
        out[feature_columns] = out[feature_columns].interpolate(
            limit_direction="both", limit=5
        )
    return out


def _finite_stat(series: pd.Series, statistic: str = "median") -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    if statistic == "mean":
        return float(values.mean())
    if statistic == "p90":
        return float(values.quantile(0.90))
    return float(values.median())


def summarize_video(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = dict(metadata)
    if df.empty:
        return metrics

    metric_map = {
        "median_body_line_deviation_deg": ("body_line_deviation", "median"),
        "mean_head_alignment_score": ("head_alignment_score", "mean"),
        "median_kick_amplitude": ("kick_amplitude", "median"),
        "median_wrist_symmetry_gap": ("wrist_symmetry_gap", "median"),
        "median_ankle_symmetry_gap": ("ankle_symmetry_gap", "median"),
        "median_elbow_angle_symmetry_gap_deg": ("elbow_angle_symmetry_gap", "median"),
        "median_knee_angle_symmetry_gap_deg": ("knee_angle_symmetry_gap", "median"),
        "mean_landmark_visibility": ("mean_landmark_visibility", "mean"),
        "p90_left_wrist_speed": ("left_wrist_speed", "p90"),
        "p90_right_wrist_speed": ("right_wrist_speed", "p90"),
    }
    for output_name, (column, statistic) in metric_map.items():
        metrics[output_name] = _finite_stat(df[column], statistic) if column in df else None

    valid_timestamps = pd.to_numeric(df.get("timestamp_sec"), errors="coerce").dropna()
    if len(valid_timestamps) >= 2:
        metrics["analyzed_duration_sec"] = float(valid_timestamps.iloc[-1] - valid_timestamps.iloc[0])
    else:
        metrics["analyzed_duration_sec"] = 0.0
    return metrics


def _confidence(metrics: Dict[str, Any], base: float = 0.82) -> float:
    detection = float(metrics.get("pose_detection_rate") or 0.0)
    visibility = metrics.get("mean_landmark_visibility")
    visibility_value = float(visibility) if visibility is not None else detection
    return float(np.clip(base * (0.55 + 0.25 * detection + 0.20 * visibility_value), 0.25, 0.95))


def generate_recommendations(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    detection_rate = float(metrics.get("pose_detection_rate") or 0.0)

    def add(
        action_id: str,
        title: str,
        priority: str,
        evidence: str,
        suggestion: str,
        drill: str,
        base_confidence: float,
    ) -> None:
        recommendations.append(
            {
                "action_id": action_id,
                "title": title,
                "priority": priority,
                "confidence": _confidence(metrics, base_confidence),
                "evidence": evidence,
                "suggestion": suggestion,
                "drill": drill,
            }
        )

    if detection_rate < 0.65:
        add(
            "camera_quality",
            "Improve recording quality before judging technique",
            "high",
            f"Pose landmarks were detected in {detection_rate:.0%} of analyzed frames.",
            "Keep the full swimmer in frame, reduce reflections, and use a side view whenever possible.",
            "Record a short 5-15 second side-view clip at 30-60 fps.",
            0.92,
        )

    body_deviation = metrics.get("median_body_line_deviation_deg")
    if body_deviation is not None and body_deviation > 18.0:
        add(
            "streamline_body_line",
            "Stabilize the body line",
            "high" if body_deviation > 28.0 else "medium",
            f"Median body-line deviation was {body_deviation:.1f} degrees in the camera plane.",
            "Reduce vertical oscillation and keep the head, trunk, and hips aligned.",
            "Six-kick streamline drill with slow exhalation.",
            0.80,
        )

    head_score = metrics.get("mean_head_alignment_score")
    if head_score is not None and head_score < 68.0:
        add(
            "breathing_alignment",
            "Keep the head closer to the body line",
            "medium",
            f"The prototype head-alignment score averaged {head_score:.0f}/100.",
            "Rotate to breathe instead of lifting the head upward.",
            "One-goggle breathing drill for four easy lengths.",
            0.76,
        )

    elbow_gap = metrics.get("median_elbow_angle_symmetry_gap_deg")
    wrist_gap = metrics.get("median_wrist_symmetry_gap")
    if (
        (elbow_gap is not None and elbow_gap > 24.0)
        or (wrist_gap is not None and wrist_gap > 0.45)
    ):
        evidence_parts = []
        if elbow_gap is not None:
            evidence_parts.append(f"elbow-angle gap {elbow_gap:.1f} degrees")
        if wrist_gap is not None:
            evidence_parts.append(f"wrist-speed gap {wrist_gap:.2f} body units/s")
        add(
            "left_right_symmetry",
            "Balance the left and right stroke pattern",
            "medium",
            "Median " + " and ".join(evidence_parts) + ".",
            "Compare the catch and recovery timing on both sides at an easy pace.",
            "3-3-3 drill followed by relaxed full-stroke swimming.",
            0.73,
        )

    kick_amplitude = metrics.get("median_kick_amplitude")
    ankle_gap = metrics.get("median_ankle_symmetry_gap")
    if (
        (kick_amplitude is not None and kick_amplitude > 0.70)
        or (ankle_gap is not None and ankle_gap > 0.55)
    ):
        evidence_parts = []
        if kick_amplitude is not None:
            evidence_parts.append(f"relative kick amplitude {kick_amplitude:.2f}")
        if ankle_gap is not None:
            evidence_parts.append(f"ankle-speed gap {ankle_gap:.2f}")
        add(
            "compact_kick",
            "Use a smaller, more even kick",
            "medium",
            "Median " + " and ".join(evidence_parts) + ".",
            "Initiate the kick from the hips and avoid excessive knee bend or scissoring.",
            "Vertical kick for 3 x 20 seconds, then easy streamline kicking.",
            0.70,
        )

    if not recommendations:
        add(
            "maintain_consistency",
            "Maintain the current pattern and collect more views",
            "low",
            "The prototype thresholds did not identify a dominant issue in this clip.",
            "Record side, front, and rear views before making a major technique change.",
            "Repeat the same pace for three short clips and compare the time series.",
            0.66,
        )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(
        key=lambda item: (priority_order.get(str(item["priority"]), 3), -float(item["confidence"]))
    )
    return recommendations


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_outputs(
    output_dir: str | Path,
    video_name: str,
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    metrics: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    annotated_video_path: str | Path | None = None,
) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)
    stem = safe_stem(video_name)
    raw_csv = output_dir / f"{stem}_raw_landmarks.csv"
    feature_csv = output_dir / f"{stem}_timeseries_features.csv"
    metrics_json = output_dir / f"{stem}_summary_metrics.json"
    recommendations_json = output_dir / f"{stem}_recommendations.json"
    bundle_zip = output_dir / f"{stem}_analysis_bundle.zip"

    df_raw.to_csv(raw_csv, index=False)
    df_features.to_csv(feature_csv, index=False)
    metrics_json.write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    recommendations_json.write_text(
        json.dumps(recommendations, indent=2, default=_json_default), encoding="utf-8"
    )

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (raw_csv, feature_csv, metrics_json, recommendations_json):
            archive.write(path, arcname=path.name)
        if annotated_video_path:
            video_path = Path(annotated_video_path)
            if video_path.exists():
                archive.write(video_path, arcname=video_path.name)

    return {
        "raw_landmarks_csv": str(raw_csv),
        "timeseries_features_csv": str(feature_csv),
        "summary_metrics_json": str(metrics_json),
        "recommendations_json": str(recommendations_json),
        "analysis_bundle_zip": str(bundle_zip),
    }


def create_demo_training_labels(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_labels: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create KMeans-derived demo labels.

    These labels prove the training pipeline works. They are not substitutes for
    labels supplied by qualified swimming coaches.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    out = df.copy()
    if feature_cols is None:
        feature_cols = [column for column in DEFAULT_FEATURE_COLUMNS if column in out.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns were found for demo labeling.")
    if len(out) < max(n_labels * 3, 12):
        raise ValueError("The clip is too short for demo labeling. Analyze more frames.")

    features = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    scaled = StandardScaler().fit_transform(imputed)
    distinct_rows = np.unique(np.round(scaled, 6), axis=0).shape[0]
    clusters = min(int(n_labels), int(distinct_rows), len(out))
    if clusters < 2:
        raise ValueError("The pose features contain only one distinct pattern.")

    labels = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(scaled)
    out["demo_label"] = [f"pattern_{label + 1}" for label in labels]
    return out


def train_supervised_models(
    df: pd.DataFrame,
    label_col: str = "demo_label",
    feature_cols: Optional[List[str]] = None,
    output_dir: str | Path = "models",
    test_size: float = 0.30,
    seed: int = 42,
    include_optional_boosters: bool = False,
) -> Dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' is missing.")
    if feature_cols is None:
        feature_cols = [column for column in DEFAULT_FEATURE_COLUMNS if column in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns were found for model training.")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    encoder = LabelEncoder()
    target = encoder.fit_transform(df[label_col].astype(str))
    if len(encoder.classes_) < 2:
        raise ValueError("At least two label classes are required.")

    counts = np.bincount(target)
    stratify = target if counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=float(test_size),
        random_state=seed,
        stratify=stratify,
    )

    candidates: Dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=240,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }

    optional_errors: Dict[str, str] = {}
    if include_optional_boosters:
        try:
            from xgboost import XGBClassifier

            candidates["xgboost"] = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=180,
                            max_depth=4,
                            learning_rate=0.06,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            random_state=seed,
                            n_jobs=2,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            optional_errors["xgboost"] = str(exc)

        try:
            from lightgbm import LGBMClassifier

            candidates["lightgbm"] = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=180,
                            learning_rate=0.06,
                            max_depth=-1,
                            random_state=seed,
                            verbosity=-1,
                            n_jobs=2,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            optional_errors["lightgbm"] = str(exc)

    output_dir = ensure_dir(output_dir)
    results: Dict[str, Dict[str, Any]] = {}
    for model_name, model in candidates.items():
        try:
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, prediction))
            matrix = confusion_matrix(y_test, prediction, labels=range(len(encoder.classes_)))
            model_path = output_dir / f"{model_name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "label_encoder": encoder,
                    "feature_columns": feature_cols,
                },
                model_path,
            )
            result: Dict[str, Any] = {
                "accuracy": accuracy,
                "model_path": str(model_path),
                "confusion_matrix": matrix.tolist(),
            }
            fitted_estimator = model.named_steps.get("model")
            if hasattr(fitted_estimator, "feature_importances_"):
                importances = np.asarray(fitted_estimator.feature_importances_, dtype=float)
                ranked = sorted(
                    zip(feature_cols, importances.tolist()),
                    key=lambda item: item[1],
                    reverse=True,
                )
                result["feature_importance"] = [
                    {"feature": feature, "importance": float(value)}
                    for feature, value in ranked[:12]
                ]
            results[model_name] = result
        except Exception as exc:
            optional_errors[model_name] = str(exc)

    if not results:
        raise RuntimeError("No supervised model completed training.")

    metadata_path = output_dir / "training_metadata.json"
    metadata = {
        "label_classes": list(encoder.classes_),
        "feature_columns": feature_cols,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "models": results,
        "skipped_models": optional_errors,
        "warning": "Demo labels are KMeans-derived and are not coach ground truth.",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )

    bundle_path = output_dir / "supervised_models_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(metadata_path, arcname=metadata_path.name)
        for result in results.values():
            path = Path(result["model_path"])
            archive.write(path, arcname=path.name)

    metadata["metadata_path"] = str(metadata_path)
    metadata["model_bundle_path"] = str(bundle_path)
    return metadata


def make_sequence_windows(
    df: pd.DataFrame,
    label_col: str,
    window_size: int = 45,
    step: int = 15,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' is missing.")
    if window_size < 2 or step < 1:
        raise ValueError("window_size must be >= 2 and step must be >= 1")
    if feature_cols is None:
        feature_cols = [column for column in DEFAULT_FEATURE_COLUMNS if column in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns were found for sequence windows.")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    labels = df[label_col].astype(str).to_numpy()

    windows: List[np.ndarray] = []
    window_labels: List[str] = []
    for start in range(0, len(features) - window_size + 1, step):
        end = start + window_size
        windows.append(features.iloc[start:end].to_numpy(dtype=np.float32))
        values, counts = np.unique(labels[start:end], return_counts=True)
        window_labels.append(str(values[np.argmax(counts)]))

    if not windows:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=str),
            feature_cols,
        )
    return np.stack(windows), np.asarray(window_labels), feature_cols


def build_deep_sequence_model(input_shape: Tuple[int, ...], n_classes: int):
    """Build the optional Conv1D plus BiLSTM model used in the Colab notebook."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.25),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def append_feedback(
    feedback_path: str | Path,
    video_name: str,
    action_id: str,
    rating: int,
    helpful: bool,
    comment: str,
    metrics: Dict[str, Any],
) -> pd.DataFrame:
    feedback_path = Path(feedback_path)
    if rating < 1 or rating > 5:
        raise ValueError("rating must be between 1 and 5")
    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path)
    else:
        feedback = pd.DataFrame(
            columns=[
                "timestamp_utc", "video_name", "action_id", "rating",
                "helpful", "comment", "metrics_json",
            ]
        )

    row = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "video_name": video_name,
        "action_id": action_id,
        "rating": int(rating),
        "helpful": bool(helpful),
        "comment": str(comment).strip(),
        "metrics_json": json.dumps(metrics, default=_json_default),
    }
    feedback = pd.concat([feedback, pd.DataFrame([row])], ignore_index=True)
    ensure_dir(feedback_path.parent)
    feedback.to_csv(feedback_path, index=False)
    return feedback


def demo_contextual_bandit_update(
    feedback_df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return a deterministic reward ranking from coach/user feedback.

    This is a transparent contextual-bandit demonstration, not an online RL
    production policy. A Bayesian-smoothed reward prevents one rating from
    completely dominating the ranking.
    """
    if feedback_df.empty:
        return {
            "status": "No feedback has been submitted.",
            "model_updated": False,
            "action_priorities": {},
        }

    rng = random.Random(seed)
    feedback = feedback_df.copy()
    feedback["rating"] = pd.to_numeric(feedback["rating"], errors="coerce").fillna(3.0)
    helpful_numeric = feedback["helpful"].astype(str).str.lower().isin(["true", "1", "yes"])
    feedback["reward"] = 0.75 * ((feedback["rating"] - 1.0) / 4.0) + 0.25 * helpful_numeric.astype(float)

    global_mean = float(feedback["reward"].mean())
    action_priorities: Dict[str, float] = {}
    action_details: List[Dict[str, Any]] = []
    action_catalog = {item["action_id"]: item for item in FEEDBACK_ACTIONS}
    for action_id in sorted(set(action_catalog) | set(feedback["action_id"].astype(str))):
        subset = feedback[feedback["action_id"].astype(str) == action_id]
        count = len(subset)
        posterior = (subset["reward"].sum() + 3.0 * global_mean) / (count + 3.0)
        exploration = 0.06 / math.sqrt(count + 1.0)
        priority = float(np.clip(posterior + exploration + rng.uniform(0.0, 0.005), 0.0, 1.0))
        action_priorities[action_id] = round(priority, 3)
        catalog_item = action_catalog.get(
            action_id,
            {"action_id": action_id, "title": action_id.replace("_", " ").title(), "drill": ""},
        )
        action_details.append(
            {
                **catalog_item,
                "feedback_count": count,
                "estimated_reward": round(float(posterior), 3),
                "priority": round(priority, 3),
            }
        )

    action_details.sort(key=lambda item: item["priority"], reverse=True)
    return {
        "status": "Feedback ranking updated.",
        "model_updated": True,
        "feedback_count": int(len(feedback)),
        "action_priorities": action_priorities,
        "ranked_actions": action_details,
    }
