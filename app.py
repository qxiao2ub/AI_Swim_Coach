"""Streamlit Community Cloud entrypoint for the AI Swimming Coach prototype.

Author: Jasper Ding
Advisor: Dr. Qingyang Xiao
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

APP_TITLE = "AI Swimming Coach"
APP_SUBTITLE = "Video pose analysis, time-series features, coaching insights, and trainable AI baselines"
AUTHOR_NAME = "Jasper Ding"
ADVISOR_NAME = "Dr. Qingyang Xiao"
SUPPORTED_EXTENSIONS = ["mp4", "mov", "m4v", "avi", "mkv", "wmv"]
REQUIRED_PYTHON = (3, 12)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="SW",
    layout="wide",
    initial_sidebar_state="expanded",
)

# MediaPipe's published Python support for this build ends at Python 3.12.
# Keep this check before pandas, NumPy, OpenCV, MediaPipe, and ai_pipeline imports
# so an accidentally deployed Python 3.13/3.14 app opens a repair screen instead
# of attempting long source builds during dependency installation.
if sys.version_info[:2] != REQUIRED_PYTHON:
    detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    st.title(APP_TITLE)
    st.error(
        f"This deployment is running Python {detected}. "
        "The full AI video-analysis stack requires Python 3.12."
    )
    st.markdown(
        """
        **One-time Streamlit Community Cloud repair**

        1. Open **Manage app** and record the repository, branch, app URL, and any secrets.
        2. Delete the current Community Cloud app. Python cannot be changed in place.
        3. Create the app again from this GitHub repository.
        4. Set the main file path to `app.py`.
        5. Open **Advanced settings** and select **Python 3.12**.
        6. Deploy. The same custom subdomain can be reused immediately.

        The repository also contains `.python-version` and `runtime.txt` as local-hosting hints,
        but Community Cloud's Python selector is the setting that controls the deployed runtime.
        """
    )
    st.code(
        "Required runtime: Python 3.12\n"
        "Streamlit entry point: app.py\n"
        f"Detected runtime: Python {detected}",
        language="text",
    )
    st.stop()

import pandas as pd

from ai_pipeline import (
    DEFAULT_FEATURE_COLUMNS,
    add_kinematic_features,
    append_feedback,
    create_demo_training_labels,
    demo_contextual_bandit_update,
    extract_pose_timeseries,
    generate_recommendations,
    get_ffmpeg_executable,
    make_sequence_windows,
    save_outputs,
    summarize_video,
    train_supervised_models,
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 1.4rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #062b46 0%, #0b6477 55%, #15919b 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 28px rgba(4, 48, 69, 0.18);
    }
    .hero h1 {font-size: 2.35rem; margin: 0 0 .35rem 0;}
    .hero p {font-size: 1.03rem; margin: 0; opacity: .92;}
    .hero .credits {font-size: .93rem; margin-top: .72rem; opacity: .96;}
    .team-card {
        border: 1px solid rgba(49, 70, 89, .16);
        border-radius: 12px;
        padding: .8rem .9rem;
        background: rgba(250, 252, 253, .78);
        margin-bottom: .8rem;
    }
    .small-note {font-size: .88rem; color: #5f6b76;}
    .rec-card {
        border: 1px solid rgba(49, 70, 89, .18);
        border-left: 5px solid #0b7285;
        border-radius: 12px;
        padding: .95rem 1rem;
        margin-bottom: .7rem;
        background: rgba(250, 252, 253, .9);
    }
    .priority-high {border-left-color: #c92a2a;}
    .priority-medium {border-left-color: #e67700;}
    .priority-low {border-left-color: #2b8a3e;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 70, 89, .14);
        border-radius: 12px;
        padding: .65rem .8rem;
        background: rgba(250, 252, 253, .75);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "analysis": None,
        "upload_fingerprint": None,
        "training": None,
        "feedback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_uploaded_name(name: str) -> str:
    filename = Path(name).name
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in filename)
    return cleaned or "uploaded_swim_video.mp4"


def _fingerprint(data: bytes, name: str) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8", errors="ignore"))
    digest.update(data[:2_000_000])
    digest.update(str(len(data)).encode("ascii"))
    return digest.hexdigest()


def _read_bytes(path: str | Path | None) -> bytes:
    if not path:
        return b""
    file_path = Path(path)
    return file_path.read_bytes() if file_path.exists() else b""


def _format_optional(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _reset_analysis() -> None:
    analysis = st.session_state.get("analysis")
    if analysis and analysis.get("work_dir"):
        shutil.rmtree(analysis["work_dir"], ignore_errors=True)
    st.session_state.analysis = None
    st.session_state.training = None
    st.session_state.feedback = None
    st.session_state.upload_fingerprint = None


def _display_recommendation(rec: Dict[str, Any], number: int) -> None:
    priority = str(rec.get("priority", "low")).lower()
    confidence = float(rec.get("confidence", 0.0))
    st.markdown(
        f"""
        <div class="rec-card priority-{priority}">
          <strong>{number}. {rec.get('title', 'Recommendation')}</strong><br>
          <span><b>Priority:</b> {priority.title()} &nbsp; | &nbsp; <b>Confidence:</b> {confidence:.0%}</span><br>
          <span><b>Evidence:</b> {rec.get('evidence', '')}</span><br>
          <span><b>Suggestion:</b> {rec.get('suggestion', '')}</span><br>
          <span><b>Training drill:</b> {rec.get('drill', '')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


_init_state()

st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
      <div class="credits"><strong>Author:</strong> {AUTHOR_NAME} &nbsp;&nbsp; | &nbsp;&nbsp; <strong>Advisor:</strong> {ADVISOR_NAME}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Prototype only: pose-derived scores and demo labels are not a substitute for an in-person qualified coach, medical advice, or injury assessment."
)

with st.sidebar:
    st.markdown("### Project team")
    st.markdown(f"**Author:** {AUTHOR_NAME}  \n**Advisor:** {ADVISOR_NAME}")
    st.divider()
    st.header("Analysis settings")
    max_frames = st.slider(
        "Maximum processed frames",
        min_value=90,
        max_value=1500,
        value=450,
        step=30,
        help="A higher value analyzes more of the clip but uses more time and memory.",
    )
    stride = st.select_slider(
        "Frame stride",
        options=[1, 2, 3, 4, 5],
        value=2,
        help="1 analyzes every frame; 2 analyzes every other frame.",
    )
    resize_width = st.select_slider(
        "Analysis width",
        options=[480, 640, 720, 960, 1280],
        value=720,
        help="Smaller widths run faster on Streamlit Community Cloud.",
    )
    detection_confidence = st.slider(
        "Pose detection confidence",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    tracking_confidence = st.slider(
        "Pose tracking confidence",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    st.divider()
    st.caption("Recommended input: a 5-20 second clip with the full swimmer visible. H.264 MP4 is the most reliable format.")
    with st.expander("Deployment diagnostics"):
        st.write(
            {
                "Python": sys.version.split()[0],
                "Required Python": "3.12",
                "XGBoost installed": importlib.util.find_spec("xgboost") is not None,
                "LightGBM installed": importlib.util.find_spec("lightgbm") is not None,
            }
        )
        if st.button(
            "Run computer-vision dependency check",
            key="run_dependency_diagnostics",
            use_container_width=True,
        ):
            try:
                import cv2
                import imageio_ffmpeg
                import mediapipe as mp

                st.success("Core computer-vision dependencies imported successfully.")
                st.write(
                    {
                        "OpenCV": cv2.__version__,
                        "MediaPipe": mp.__version__,
                        "Bundled FFmpeg": get_ffmpeg_executable() or "Unavailable",
                        "imageio-ffmpeg": getattr(imageio_ffmpeg, "__version__", "unknown"),
                    }
                )
            except Exception as diagnostic_error:
                st.warning(f"Dependency diagnostic failed: {diagnostic_error}")
    if st.session_state.analysis is not None:
        if st.button("Clear current analysis", use_container_width=True):
            _reset_analysis()
            st.rerun()

upload_col, guide_col = st.columns([1.5, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a swimming video",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=False,
        help="Supported containers depend on the codecs available to OpenCV and ffmpeg.",
    )
with guide_col:
    st.markdown("**Recording checklist**")
    st.markdown(
        "- Keep the entire swimmer in the frame.\n"
        "- Prefer a stable side view for body-line analysis.\n"
        "- Reduce glare, splashes, and obstructions.\n"
        "- Avoid uploading private or sensitive videos."
    )

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    current_fingerprint = _fingerprint(uploaded_bytes, uploaded_file.name)
    if (
        st.session_state.upload_fingerprint is not None
        and current_fingerprint != st.session_state.upload_fingerprint
    ):
        _reset_analysis()

    analyze_button = st.button(
        "Analyze swimming video",
        type="primary",
        use_container_width=True,
        disabled=len(uploaded_bytes) == 0,
    )

    if analyze_button:
        work_dir = Path(tempfile.mkdtemp(prefix="ai_swim_coach_"))
        upload_path = work_dir / _safe_uploaded_name(uploaded_file.name)
        output_dir = work_dir / "outputs"
        upload_path.write_bytes(uploaded_bytes)

        progress_bar = st.progress(0, text="Starting analysis")
        status_box = st.empty()

        def progress_callback(value: float, message: str) -> None:
            progress_bar.progress(min(max(int(value * 100), 0), 100), text=message)
            status_box.caption(message)

        try:
            raw_df, annotated_path, metadata = extract_pose_timeseries(
                upload_path,
                output_dir=output_dir,
                max_frames=max_frames,
                stride=stride,
                resize_width=resize_width,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                create_annotated_video=True,
                progress_callback=progress_callback,
            )
            feature_df = add_kinematic_features(raw_df)
            metrics = summarize_video(feature_df, metadata)
            recommendations = generate_recommendations(metrics)
            saved_paths = save_outputs(
                output_dir,
                uploaded_file.name,
                raw_df,
                feature_df,
                metrics,
                recommendations,
                annotated_video_path=annotated_path,
            )
            progress_callback(1.0, "Analysis complete")
            st.session_state.analysis = {
                "work_dir": str(work_dir),
                "video_name": uploaded_file.name,
                "upload_path": str(upload_path),
                "annotated_path": annotated_path,
                "raw_df": raw_df,
                "feature_df": feature_df,
                "metrics": metrics,
                "recommendations": recommendations,
                "saved_paths": saved_paths,
            }
            st.session_state.upload_fingerprint = current_fingerprint
            st.session_state.training = None
            st.session_state.feedback = None
            st.success("The video was processed successfully.")
            st.rerun()
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            progress_bar.empty()
            status_box.empty()
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)
else:
    st.caption("Upload a video to activate the analysis controls.")

analysis: Optional[Dict[str, Any]] = st.session_state.analysis
if analysis is None:
    st.markdown("### What this prototype produces")
    feature_columns = [
        ("Annotated video", "Pose landmarks and body connections overlaid on the uploaded clip."),
        ("Time-series data", "Frame-level landmarks, joint angles, movement speeds, symmetry gaps, and alignment features."),
        ("Coaching insights", "Transparent rule-based evidence, suggestions, and training drills."),
        ("Trainable AI", "Cloud-ready Logistic Regression, Random Forest, and Gradient Boosting; optional XGBoost, LightGBM, and Conv1D plus BiLSTM extensions."),
        ("Feedback learning", "A coach/user rating loop that updates recommendation priorities."),
    ]
    for left, right in feature_columns:
        st.markdown(f"**{left}:** {right}")
    st.stop()

metrics = analysis["metrics"]
feature_df: pd.DataFrame = analysis["feature_df"]
raw_df: pd.DataFrame = analysis["raw_df"]
recommendations = analysis["recommendations"]
saved_paths = analysis["saved_paths"]

overview_tab, video_tab, data_tab, ml_tab, deep_tab, feedback_tab, about_tab = st.tabs(
    [
        "Overview",
        "Annotated video",
        "Time-series data",
        "Supervised ML",
        "Deep learning",
        "Feedback learning",
        "About",
    ]
)

with overview_tab:
    st.subheader("Analysis summary")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Processed frames", int(metrics.get("processed_frames", 0)))
    metric_columns[1].metric("Pose detection", f"{float(metrics.get('pose_detection_rate', 0.0)):.0%}")
    metric_columns[2].metric(
        "Analyzed duration",
        _format_optional(metrics.get("analyzed_duration_sec"), 1, " s"),
    )
    metric_columns[3].metric(
        "Body-line deviation",
        _format_optional(metrics.get("median_body_line_deviation_deg"), 1, " deg"),
    )
    metric_columns[4].metric(
        "Head alignment",
        _format_optional(metrics.get("mean_head_alignment_score"), 0, "/100"),
    )

    st.subheader("Coach-style recommendations")
    for index, recommendation in enumerate(recommendations, start=1):
        _display_recommendation(recommendation, index)

    bundle_bytes = _read_bytes(saved_paths.get("analysis_bundle_zip"))
    st.download_button(
        "Download complete analysis bundle",
        data=bundle_bytes,
        file_name=Path(saved_paths["analysis_bundle_zip"]).name,
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

with video_tab:
    st.subheader("Pose-annotated video")
    annotated_path = analysis.get("annotated_path")
    if annotated_path and Path(annotated_path).exists():
        st.video(annotated_path)
        st.download_button(
            "Download annotated video",
            data=_read_bytes(annotated_path),
            file_name=Path(annotated_path).name,
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.warning("An annotated video was not created for this upload.")

    st.caption(
        "Landmarks can be unreliable underwater, during heavy splash, or when limbs overlap. Use the detection-rate metric when interpreting results."
    )

with data_tab:
    st.subheader("Machine-readable time series")
    available_plot_columns = [
        column
        for column in [
            "body_line_deviation",
            "head_alignment_score",
            "left_elbow_angle",
            "right_elbow_angle",
            "kick_amplitude",
            "wrist_symmetry_gap",
            "ankle_symmetry_gap",
        ]
        if column in feature_df.columns
    ]
    selected_columns = st.multiselect(
        "Features to plot",
        options=available_plot_columns,
        default=available_plot_columns[:4],
    )
    if selected_columns and "timestamp_sec" in feature_df:
        chart_data = feature_df[["timestamp_sec", *selected_columns]].copy()
        chart_data = chart_data.set_index("timestamp_sec")
        st.line_chart(chart_data, height=430)

    preview_columns = [
        column
        for column in ["frame_index", "timestamp_sec", "pose_detected", *DEFAULT_FEATURE_COLUMNS]
        if column in feature_df.columns
    ]
    st.dataframe(feature_df[preview_columns].head(250), use_container_width=True, height=380)

    left_download, right_download = st.columns(2)
    with left_download:
        st.download_button(
            "Download feature CSV",
            data=_read_bytes(saved_paths.get("timeseries_features_csv")),
            file_name=Path(saved_paths["timeseries_features_csv"]).name,
            mime="text/csv",
            use_container_width=True,
        )
    with right_download:
        st.download_button(
            "Download raw landmark CSV",
            data=_read_bytes(saved_paths.get("raw_landmarks_csv")),
            file_name=Path(saved_paths["raw_landmarks_csv"]).name,
            mime="text/csv",
            use_container_width=True,
        )

with ml_tab:
    st.subheader("Supervised machine-learning baseline")
    st.warning(
        "The current labels are KMeans-derived demo labels. Accuracy here validates the software pipeline, not swimming-coach accuracy. Replace them with coach-labeled stroke phases or technique issues."
    )
    labels_col, booster_col = st.columns(2)
    with labels_col:
        n_labels = st.slider("Number of demo patterns", 2, 5, 3)
    with booster_col:
        boosters_available = (
            importlib.util.find_spec("xgboost") is not None
            and importlib.util.find_spec("lightgbm") is not None
        )
        include_boosters = st.checkbox(
            "Include optional XGBoost and LightGBM",
            value=False,
            disabled=not boosters_available,
            help=(
                "These optional packages are intentionally excluded from the fast Community Cloud build. "
                "Install requirements-optional-boosters.txt locally or in Colab to enable them."
            ),
        )
        if not boosters_available:
            st.caption("Cloud-fast mode: Logistic Regression, Random Forest, and Gradient Boosting are enabled.")

    if st.button("Train supervised models", type="primary", use_container_width=True):
        try:
            with st.spinner("Creating demo labels and training models..."):
                training_df = create_demo_training_labels(
                    feature_df,
                    n_labels=n_labels,
                )
                model_dir = Path(analysis["work_dir"]) / "models"
                results = train_supervised_models(
                    training_df,
                    output_dir=model_dir,
                    include_optional_boosters=include_boosters,
                )
                st.session_state.training = {
                    "training_df": training_df,
                    "results": results,
                }
            st.success("Model training completed.")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    training_state = st.session_state.training
    if training_state:
        results = training_state["results"]
        result_rows = [
            {
                "Model": model_name.replace("_", " ").title(),
                "Demo accuracy": model_result["accuracy"],
            }
            for model_name, model_result in results["models"].items()
        ]
        result_frame = pd.DataFrame(result_rows).sort_values("Demo accuracy", ascending=False)
        st.dataframe(
            result_frame.style.format({"Demo accuracy": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        if results.get("skipped_models"):
            with st.expander("Skipped model details"):
                st.json(results["skipped_models"])

        best_model_name = max(
            results["models"],
            key=lambda name: results["models"][name]["accuracy"],
        )
        importance = results["models"][best_model_name].get("feature_importance")
        if importance:
            st.markdown(f"**Top features from {best_model_name.replace('_', ' ').title()}**")
            importance_df = pd.DataFrame(importance).set_index("feature")
            st.bar_chart(importance_df["importance"], horizontal=True, height=360)

        st.download_button(
            "Download trained model bundle",
            data=_read_bytes(results.get("model_bundle_path")),
            file_name="supervised_models_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

with deep_tab:
    st.subheader("Deep-learning sequence design")
    st.write(
        "The Colab notebook defines a Conv1D plus bidirectional LSTM network for stroke-phase or technique-issue classification. TensorFlow is intentionally not installed in the default Community Cloud environment to keep deployment lighter and more reliable."
    )

    architecture = pd.DataFrame(
        [
            ["Input", "window_size x feature_count", "Pose-feature sequence"],
            ["Conv1D", "64 filters, kernel 5", "Local motion patterns"],
            ["Batch normalization", "64 channels", "Stable optimization"],
            ["Max pooling", "pool size 2", "Temporal compression"],
            ["BiLSTM", "64 units", "Past and future context"],
            ["Dropout", "0.25", "Regularization"],
            ["BiLSTM", "32 units", "Sequence summary"],
            ["Dense", "64 ReLU", "Technique representation"],
            ["Softmax", "one unit per class", "Stroke phase or issue"],
        ],
        columns=["Layer", "Configuration", "Purpose"],
    )
    st.dataframe(architecture, use_container_width=True, hide_index=True)

    window_col, step_col = st.columns(2)
    with window_col:
        window_size = st.slider("Sequence window size", 15, 120, 45, 5)
    with step_col:
        window_step = st.slider("Window step", 5, 60, 15, 5)

    sequence_source = (
        st.session_state.training["training_df"]
        if st.session_state.training
        else None
    )
    if sequence_source is None:
        try:
            sequence_source = create_demo_training_labels(feature_df, n_labels=3)
        except Exception:
            sequence_source = None

    if sequence_source is not None:
        x_sequence, y_sequence, sequence_features = make_sequence_windows(
            sequence_source,
            label_col="demo_label",
            window_size=window_size,
            step=window_step,
        )
        sequence_metrics = st.columns(3)
        sequence_metrics[0].metric("Sequence windows", int(len(x_sequence)))
        sequence_metrics[1].metric("Frames per window", int(window_size))
        sequence_metrics[2].metric("Features per frame", int(len(sequence_features)))
        if len(y_sequence):
            st.caption("Window-label distribution")
            st.dataframe(
                pd.Series(y_sequence, name="demo_label").value_counts().rename_axis("label").reset_index(name="windows"),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Analyze a longer clip to create sequence windows.")

    template_path = Path(__file__).with_name("deep_learning_template.py")
    if template_path.exists():
        st.download_button(
            "Download optional TensorFlow training template",
            data=template_path.read_bytes(),
            file_name=template_path.name,
            mime="text/x-python",
            use_container_width=True,
        )

with feedback_tab:
    st.subheader("Coach and swimmer feedback loop")
    st.write(
        "Rate the recommendation quality. The prototype converts ratings into a transparent reward score and updates the recommendation ranking."
    )
    recommendation_options = {
        recommendation["action_id"]: recommendation["title"]
        for recommendation in recommendations
    }
    with st.form("feedback_form", clear_on_submit=True):
        selected_action = st.selectbox(
            "Recommendation",
            options=list(recommendation_options),
            format_func=lambda action_id: recommendation_options[action_id],
        )
        rating = st.slider("Rating", 1, 5, 4)
        helpful = st.checkbox("This recommendation was helpful", value=True)
        comment = st.text_area("Coach or swimmer comment", max_chars=1000)
        submitted = st.form_submit_button("Submit feedback", type="primary", use_container_width=True)

    if submitted:
        feedback_path = Path(analysis["work_dir"]) / "data" / "feedback_log.csv"
        current_feedback = append_feedback(
            feedback_path,
            video_name=analysis["video_name"],
            action_id=selected_action,
            rating=rating,
            helpful=helpful,
            comment=comment,
            metrics=metrics,
        )
        bandit_state = demo_contextual_bandit_update(current_feedback)
        st.session_state.feedback = {
            "dataframe": current_feedback,
            "state": bandit_state,
            "path": str(feedback_path),
        }
        st.success("Feedback saved for this session.")

    if st.session_state.feedback:
        feedback_state = st.session_state.feedback
        st.dataframe(feedback_state["dataframe"], use_container_width=True, hide_index=True)
        ranked_actions = feedback_state["state"].get("ranked_actions", [])
        if ranked_actions:
            rank_df = pd.DataFrame(ranked_actions)[
                ["title", "feedback_count", "estimated_reward", "priority"]
            ]
            st.markdown("**Updated action ranking**")
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download feedback CSV",
            data=_read_bytes(feedback_state["path"]),
            file_name="feedback_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Community Cloud storage is temporary. Download feedback before the app restarts or redeploys.")

with about_tab:
    st.subheader("Project team")
    team_left, team_right = st.columns(2)
    team_left.markdown(f"**Author**  \n{AUTHOR_NAME}")
    team_right.markdown(f"**Advisor**  \n{ADVISOR_NAME}")
    st.divider()
    st.subheader("Prototype architecture")
    st.markdown(
        "1. **Input:** uploaded swimming video.\n"
        "2. **Computer vision:** MediaPipe Pose Landmarker extracts 33 body landmarks.\n"
        "3. **Feature engineering:** joint angles, body alignment, speed, kick amplitude, and symmetry time series.\n"
        "4. **Recommendation engine:** transparent rules generate evidence, suggestions, and drills.\n"
        "5. **Supervised learning:** Logistic Regression, Random Forest, and Gradient Boosting run in the cloud build; XGBoost and LightGBM remain optional.\n"
        "6. **Deep learning:** optional Conv1D plus BiLSTM model in Colab.\n"
        "7. **Feedback learning:** ratings update action priorities through a reward-ranking demonstration."
    )
    st.markdown("**Limitations**")
    st.markdown(
        "- Underwater refraction and occlusion can reduce landmark accuracy.\n"
        "- Camera-plane angles are not full biomechanical measurements.\n"
        "- Demo labels are unsupervised clusters, not coach ground truth.\n"
        "- Real validation requires diverse, consented, coach-labeled swimming videos."
    )
    with st.expander("Current summary JSON"):
        st.code(json.dumps(metrics, indent=2, default=str), language="json")
