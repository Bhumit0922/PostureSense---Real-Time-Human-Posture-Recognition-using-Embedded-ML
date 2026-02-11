import pandas as pd
import numpy as np
import time
import pickle

from preprocessing.normalize import normalize_signal
from preprocessing.filter import low_pass_filter
from preprocessing.calibration import calibrate
from features.extract_features import extract_features

# ==============================
# 1️⃣ Load Trained Model & Scaler
# ==============================

with open("models/trained/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/trained/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("✅ Trained Random Forest model loaded")
print("✅ Scaler loaded")

# ==============================
# 2️⃣ Load Sensor Data
# ==============================

df = pd.read_csv("data/raw/standing.csv")

# ==============================
# 3️⃣ Preprocessing (EXACT SAME ORDER AS PHASE 1)
# ==============================

# Normalize
df["Ax"] = normalize_signal(df["Ax"])
df["Ay"] = normalize_signal(df["Ay"])
df["Az"] = normalize_signal(df["Az"])

# Filter FIRST (VERY IMPORTANT)
df["Ax_f"] = low_pass_filter(df["Ax"])
df["Ay_f"] = low_pass_filter(df["Ay"])
df["Az_f"] = low_pass_filter(df["Az"])

# Now calibrate using filtered signals
calib_df = df.head(50).copy()
calib_ref = calibrate(calib_df)

# Apply calibration correction on filtered signals
df["Ax_f"] = df["Ax_f"] - calib_ref["Ax_ref"]
df["Ay_f"] = df["Ay_f"] - calib_ref["Ay_ref"]
df["Az_f"] = df["Az_f"] - calib_ref["Az_ref"]

# ==============================
# 4️⃣ Real-Time Sliding Window Simulation
# ==============================

buffer = []
WINDOW_SIZE = 18
STEP_TIME = 0.02
MAX_SAMPLES = int(3 / STEP_TIME)

print("\n🚀 Starting real-time posture simulation...\n")


for i in range(min(len(df), MAX_SAMPLES)):

    buffer.append(df.iloc[i])

    if len(buffer) > WINDOW_SIZE:
        buffer.pop(0)

    if len(buffer) == WINDOW_SIZE:

        window_df = pd.DataFrame(buffer)

        # Feature extraction
        features = extract_features(window_df)
        features = np.array(features).reshape(1, -1)

        # Apply scaler
        features = scaler.transform(features)

        # Predict
        posture = model.predict(features)[0]

        print(
            f"sample={i} | "
            f"t={df.iloc[i]['timestamp']:.2f}s → "
            f"POSTURE: {posture.upper()}"
        )

    time.sleep(STEP_TIME)

print("\n🛑 End of demo segment")
