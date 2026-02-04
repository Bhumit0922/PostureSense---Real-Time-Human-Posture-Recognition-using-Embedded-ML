import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(window):
    """
    Extract time-domain features from a window.

    Parameters:
    window : pandas DataFrame
        Window containing Ax_f, Ay_f, Az_f

    Returns:
    List of extracted features
    """
    features = []

    for axis in ["Ax_f", "Ay_f", "Az_f"]:
        signal = window[axis].values

        features.append(np.mean(np.abs(signal)))      # Mean Absolute Value
        features.append(np.var(signal))                # Variance
        features.append(np.max(signal) - np.min(signal))  # Dynamic Acceleration Change
        features.append(skew(signal))                  # Skewness
        features.append(kurtosis(signal))              # Kurtosis

    return features
