import serial
import numpy as np
import pandas as pd
import pickle
import time

from preprocessing.normalize import normalize_signal
from preprocessing.filter import low_pass_filter
from features.extract_features import extract_features

PORT = "COM3"   # 🔥 CHANGE IF NEEDED
BAUD = 9600

# 🔹 CONNECT SERIAL SAFELY
while True:
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        print("✅ Connected to Arduino on", PORT)
        break
    except:
        print("⏳ Waiting for Arduino connection...")
        time.sleep(2)

# 🔹 LOAD MODEL
with open("models/trained/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/trained/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("✅ Model Loaded")

buffer = []
WINDOW_SIZE = 18

while True:
    try:
        # 🔹 READ SERIAL DATA
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        values = line.split(",")

        # Skip invalid data
        if len(values) != 3:
            continue

        try:
            x, y, z = map(float, values)
        except:
            continue

        # 🔹 STORE DATA
        buffer.append({"Ax": x, "Ay": y, "Az": z})

        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)

        # 🔹 ML PREDICTION
        if len(buffer) == WINDOW_SIZE:

            df = pd.DataFrame(buffer)

            df["Ax"] = normalize_signal(df["Ax"])
            df["Ay"] = normalize_signal(df["Ay"])
            df["Az"] = normalize_signal(df["Az"])

            df["Ax_f"] = low_pass_filter(df["Ax"])
            df["Ay_f"] = low_pass_filter(df["Ay"])
            df["Az_f"] = low_pass_filter(df["Az"])

            features = extract_features(df)
            features = np.array(features).reshape(1, -1)
            features = scaler.transform(features)

            posture = model.predict(features)[0]

            # 🔥 DECISION LOGIC
            result = "BAD" if posture == "sitting" else "GOOD"

            print(f"📊 {posture.upper()} → {result}")

            # 🔹 SEND BACK TO ARDUINO
            ser.write((result + "\r\n").encode())

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(1)