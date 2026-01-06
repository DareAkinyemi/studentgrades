"""
Streamlit app: predict overall_score.
If model.joblib/model_meta.json are missing, it trains them automatically.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_DIR = Path(os.getenv("MODEL_DIR", "."))
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
DATA_PATH = MODEL_DIR / "Student_Performance.csv"   # keep CSV in repo
TRAIN_SCRIPT = MODEL_DIR / "train.py"


def ensure_model_exists(mode: str = "full") -> None:
    """
    Train model if artifacts are missing.
    """
    if MODEL_PATH.exists() and META_PATH.exists():
        return

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Add Student_Performance.csv to the repo "
            f"or switch to the 'download model' approach."
        )
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError("train.py not found beside app.py.")

    st.info("Model not found. Training model (first run only)...")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data", str(DATA_PATH),
        "--mode", mode,
        "--outdir", str(MODEL_DIR),
    ]
    # Run training and show errors in Streamlit if it fails
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("Training failed.")
        st.code(result.stdout + "\n" + result.stderr)
        raise RuntimeError("Training failed. See logs above.")

    st.success("Training complete. Model saved.")


@st.cache_resource
def load_artifacts(mode: str):
    ensure_model_exists(mode=mode)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def build_input_form(meta: dict) -> dict:
    st.sidebar.header("Student inputs")
    inputs = {}

    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])

    st.sidebar.subheader("Numeric")
    for col in num_cols:
        default = 0.0
        if col == "age":
            default = 16.0
        elif col == "study_hours":
            default = 2.0
        elif col == "attendance_percentage":
            default = 80.0
        inputs[col] = st.sidebar.number_input(col, value=float(default))

    st.sidebar.subheader("Categorical")
    common = {
        "gender": ["female", "male", "other"],
        "school_type": ["public", "private"],
        "internet_access": ["yes", "no"],
        "extra_activities": ["yes", "no"],
        "travel_time": ["<15 min", "15-30 min", "30-60 min", ">60 min"],
        "study_method": ["notes", "group", "online", "tutor", "self"],
        "parent_education": ["high school", "college", "graduate", "post graduate"],
    }
    for col in cat_cols:
        options = common.get(col, [])
        if options:
            inputs[col] = st.sidebar.selectbox(col, options=options, index=0)
        else:
            inputs[col] = st.sidebar.text_input(col, value="")

    return inputs


def main():
    st.set_page_config(page_title="Overall Score Predictor", layout="centered")
    st.title("Student Overall Score Predictor")

    # Let you switch modes from the UI if you want
    mode = st.sidebar.selectbox("Training mode", ["full", "early"], index=0)

    model, meta = load_artifacts(mode)

    st.caption(f"Model mode: **{meta.get('mode','?')}** · Best model: **{meta.get('best_model','?')}**")

    inputs = build_input_form(meta)

    feature_cols = meta["feature_cols"]
    input_df = pd.DataFrame([[inputs.get(c) for c in feature_cols]], columns=feature_cols)

    st.subheader("Prediction")
    if st.button("Predict overall score"):
        pred = float(model.predict(input_df)[0])
        pred = max(0.0, min(100.0, pred))
        st.metric("Predicted overall score", f"{pred:.1f} / 100")

        if pred >= 90:
            grade = "A+"
        elif pred >= 80:
            grade = "A"
        elif pred >= 70:
            grade = "B"
        elif pred >= 60:
            grade = "C"
        elif pred >= 50:
            grade = "D"
        else:
            grade = "F"
        st.write(f"Estimated grade band: **{grade}**")

    with st.expander("Show the input row sent to the model"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("Model metrics"):
        st.json(meta.get("metrics", {}))


if __name__ == "__main__":
    main()
