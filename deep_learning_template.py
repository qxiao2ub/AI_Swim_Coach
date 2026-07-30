"""Optional TensorFlow training template for the AI Swimming Coach project.

Run this in Google Colab after producing a feature CSV with coach-supplied labels.
The Streamlit Community Cloud deployment intentionally does not install
TensorFlow by default.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ai_pipeline import build_deep_sequence_model, make_sequence_windows

FEATURE_CSV = Path("outputs/swim_timeseries_features.csv")
LABEL_COLUMN = "coach_label"
WINDOW_SIZE = 45
WINDOW_STEP = 15
EPOCHS = 30


def main() -> None:
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURE_CSV}")

    frame = pd.read_csv(FEATURE_CSV)
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(
            f"Add a '{LABEL_COLUMN}' column containing coach-approved stroke "
            "phase or technique labels before training."
        )

    x_sequence, y_text, feature_columns = make_sequence_windows(
        frame,
        label_col=LABEL_COLUMN,
        window_size=WINDOW_SIZE,
        step=WINDOW_STEP,
    )
    if len(x_sequence) < 10:
        raise ValueError("Not enough sequence windows. Add more labeled video data.")

    encoder = LabelEncoder()
    y_sequence = encoder.fit_transform(y_text)
    model = build_deep_sequence_model(
        input_shape=x_sequence.shape[1:],
        n_classes=len(encoder.classes_),
    )
    model.summary()
    model.fit(
        x_sequence,
        y_sequence,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=16,
        verbose=1,
    )
    model.save("models/swim_sequence_model.keras")
    np.save("models/swim_sequence_label_classes.npy", encoder.classes_)
    Path("models/swim_sequence_features.txt").write_text(
        "\n".join(feature_columns), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
