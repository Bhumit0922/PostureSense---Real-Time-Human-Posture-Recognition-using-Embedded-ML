import numpy as np

def extract_features(window):

    features = []

    for axis in ["Ax_f", "Ay_f", "Az_f"]:
        signal = window[axis].values

        # Basic statistics
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.var(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))

        # RMS
        features.append(np.sqrt(np.mean(signal**2)))

        # Energy
        features.append(np.sum(signal**2))

    # Signal Magnitude Area (SMA)
    sma = np.sum(
        np.abs(window["Ax_f"]) +
        np.abs(window["Ay_f"]) +
        np.abs(window["Az_f"])
    )
    features.append(sma)

    # Magnitude
    mag = np.sqrt(
        window["Ax_f"]**2 +
        window["Ay_f"]**2 +
        window["Az_f"]**2
    )

    features.append(np.mean(mag))
    features.append(np.std(mag))

    # Correlations
    features.append(np.corrcoef(window["Ax_f"], window["Ay_f"])[0,1])
    features.append(np.corrcoef(window["Ay_f"], window["Az_f"])[0,1])
    features.append(np.corrcoef(window["Ax_f"], window["Az_f"])[0,1])

    return features
