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

calib_ref = None

calib_ref = None

for posture in POSTURES:
    print(f"\nProcessing posture: {posture}")

    file_path = os.path.join(RAW_DATA_PATH, f"{posture}.csv")
    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} samples")

    # Normalize
    # Normalize
    df["Ax"] = normalize_signal(df["Ax"])
    df["Ay"] = normalize_signal(df["Ay"])
    df["Az"] = normalize_signal(df["Az"])

# Filter FIRST
    df["Ax_f"] = low_pass_filter(df["Ax"])
    df["Ay_f"] = low_pass_filter(df["Ay"])
    df["Az_f"] = low_pass_filter(df["Az"])

# Compute calibration using filtered signals
    if posture == "standing":
        calib_ref = calibrate(df)
        print("Calibration reference:", calib_ref)

# Apply calibration correction on filtered signals
    df["Ax_f"] = df["Ax_f"] - calib_ref["Ax_ref"]
    df["Ay_f"] = df["Ay_f"] - calib_ref["Ay_ref"]
    df["Az_f"] = df["Az_f"] - calib_ref["Az_ref"]


    # Sliding window
    windows = sliding_window(df)
    print(f"Created {len(windows)} windows")

    for w in windows:
        features = extract_features(w)
        X.append(features)
        y.append(posture)


    windows = sliding_window(df)
    print(f"Created {len(windows)} windows")

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
