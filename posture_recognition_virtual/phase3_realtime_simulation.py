import pandas as pd
import numpy as np
import time
import pickle

from preprocessing.normalize import normalize_signal
from preprocessing.filter import low_pass_filter
from features.extract_features import extract_features

# Load trained model
with open("models/trained/dt_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Trained model loaded")

# Load CSV as sensor data
df = pd.read_csv("data/raw/standing.csv")

# Preprocessing
df["Ax"] = normalize_signal(df["Ax"])
df["Ay"] = normalize_signal(df["Ay"])
df["Az"] = normalize_signal(df["Az"])

df["Ax_f"] = low_pass_filter(df["Ax"])
df["Ay_f"] = low_pass_filter(df["Ay"])
df["Az_f"] = low_pass_filter(df["Az"])

buffer = []
WINDOW_SIZE = 18
STEP_TIME = 0.02
MAX_SAMPLES = int(3 / STEP_TIME)  # 3 seconds demo

print("\n🚀 Starting real-time posture simulation...\n")

for i in range(min(len(df), MAX_SAMPLES)):
    buffer.append(df.iloc[i])

    if len(buffer) > WINDOW_SIZE:
        buffer.pop(0)

    if len(buffer) == WINDOW_SIZE:
        window_df = pd.DataFrame(buffer)

        features = extract_features(window_df)
        features = np.array(features).reshape(1, -1)

        posture = model.predict(features)[0]

        print(
            f"sample={i} | "
            f"t={df.iloc[i]['timestamp']:.2f}s → "
            f"POSTURE: {posture.upper()}"
        )

    time.sleep(STEP_TIME)

print("\n🛑 End of demo segment")
