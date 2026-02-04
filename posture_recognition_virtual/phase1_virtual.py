import pandas as pd
import numpy as np
import os

from preprocessing.normalize import normalize_signal
from preprocessing.filter import low_pass_filter
from preprocessing.calibration import calibrate
from windowing.sliding_window import sliding_window
from features.extract_features import extract_features

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
POSTURES = ["standing", "sitting", "lying"]

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

X = []
y = []

for posture in POSTURES:
    print(f"\nProcessing posture: {posture}")

    file_path = os.path.join(RAW_DATA_PATH, f"{posture}.csv")
    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} samples")

    # Normalize
    df["Ax"] = normalize_signal(df["Ax"])
    df["Ay"] = normalize_signal(df["Ay"])
    df["Az"] = normalize_signal(df["Az"])

    # Filter
    df["Ax_f"] = low_pass_filter(df["Ax"])
    df["Ay_f"] = low_pass_filter(df["Ay"])
    df["Az_f"] = low_pass_filter(df["Az"])

    # Calibration
    if posture == "standing":
        calib_ref = calibrate(df)
        print("Calibration reference:", calib_ref)

    # Sliding window
    windows = sliding_window(df)
    print(f"Created {len(windows)} windows")

    # Feature extraction
    for w in windows:
        features = extract_features(w)
        X.append(features)
        y.append(posture)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(PROCESSED_DATA_PATH, "X.npy"), X)
np.save(os.path.join(PROCESSED_DATA_PATH, "y.npy"), y)

print("\n✅ Phase 1 completed successfully")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique labels:", np.unique(y))
