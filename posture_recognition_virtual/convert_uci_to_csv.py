import numpy as np
import pandas as pd
import os

# =========================
# Paths
# =========================
base_path = "data/uci_har/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/"
label_path = "data/uci_har/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"
output_path = "data/raw/"

os.makedirs(output_path, exist_ok=True)

# =========================
# Load accelerometer signals
# =========================
x = np.loadtxt(base_path + "body_acc_x_train.txt")
y = np.loadtxt(base_path + "body_acc_y_train.txt")
z = np.loadtxt(base_path + "body_acc_z_train.txt")

labels = np.loadtxt(label_path)

# =========================
# Activity mapping
# =========================
activity_map = {
    4: "sitting",
    5: "standing",
    6: "lying"
}

fs = 50               # Sampling frequency (Hz)
dt = 1 / fs

# =========================
# Convert to CSV
# =========================
for act_id, act_name in activity_map.items():
    indices = np.where(labels == act_id)[0][:10]  # take few samples only

    rows = []
    for idx in indices:
        for i in range(len(x[idx])):
            rows.append([
                i * dt,
                x[idx][i],
                y[idx][i],
                z[idx][i]
            ])

    df = pd.DataFrame(rows, columns=["timestamp", "Ax", "Ay", "Az"])
    df.to_csv(output_path + f"{act_name}.csv", index=False)

    print(f"{act_name}.csv created with {len(df)} rows")

print("\n✅ Conversion complete. CSV files are ready.")
