\
"""
Train a model to predict `overall_score` from Student_Performance.csv.

Usage:
  python train.py --data Student_Performance.csv --mode full
  python train.py --data Student_Performance.csv --mode early

Modes:
  - full  : Uses all non-target columns except obvious leakage columns.
  - early : Uses only "pre-exam" features (no subject scores).
Outputs:
  - model.joblib (sklearn Pipeline)
  - model_meta.json (feature lists + metrics)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


TARGET = "overall_score"


def select_features(df: pd.DataFrame, mode: str) -> list[str]:
    # Columns that should never be used as features
    always_drop = {"student_id", TARGET}

    # final_grade is usually derived from overall_score => leakage
    if "final_grade" in df.columns:
        always_drop.add("final_grade")

    # If mode=early, avoid using subject scores (often not known "ahead of time")
    if mode == "early":
        for col in ["math_score", "science_score", "english_score"]:
            if col in df.columns:
                always_drop.add(col)

    features = [c for c in df.columns if c not in always_drop]
    if not features:
        raise ValueError("No features left after filtering. Check column names.")
    return features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to Student_Performance.csv")
    ap.add_argument("--mode", type=str, default="full", choices=["full", "early"])
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for artifacts")
    args = ap.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data.")

    feature_cols = select_features(df, args.mode)
    X = df[feature_cols].copy()
    y = df[TARGET].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    # Two reasonable models (fast + strong baseline). We'll pick the best by MAE.
    candidates = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "rf": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name, best_mae, best_pipe = None, float("inf"), None

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        results[name] = {"mae": mae, "r2": r2}

        if mae < best_mae:
            best_name, best_mae, best_pipe = name, mae, pipe

    assert best_pipe is not None

    # Save artifacts
    joblib.dump(best_pipe, outdir / "model.joblib")

    meta = {
        "mode": args.mode,
        "best_model": best_name,
        "metrics": results,
        "feature_cols": feature_cols,
        "num_cols": X_train.select_dtypes(include=["number"]).columns.tolist(),
        "cat_cols": [c for c in feature_cols if c not in X_train.select_dtypes(include=["number"]).columns],
    }
    (outdir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    print("Saved:")
    print(f" - {outdir / 'model.joblib'}")
    print(f" - {outdir / 'model_meta.json'}")
    print("\nMetrics:")
    for k, v in results.items():
        print(f" - {k}: MAE={v['mae']:.3f}, R2={v['r2']:.3f}")
    print(f"\nBest: {best_name} (MAE={best_mae:.3f})")


if __name__ == "__main__":
    main()
