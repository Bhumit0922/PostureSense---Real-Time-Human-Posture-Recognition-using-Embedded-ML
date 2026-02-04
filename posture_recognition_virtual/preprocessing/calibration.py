import numpy as np

def calibrate(df):
    """
    Compute calibration reference values.

    Parameters:
    df : pandas DataFrame
        DataFrame containing filtered Ax_f, Ay_f, Az_f

    Returns:
    Dictionary with reference mean values
    """
    reference = {
        "Ax_ref": np.mean(df["Ax_f"]),
        "Ay_ref": np.mean(df["Ay_f"]),
        "Az_ref": np.mean(df["Az_f"])
    }
    return reference
