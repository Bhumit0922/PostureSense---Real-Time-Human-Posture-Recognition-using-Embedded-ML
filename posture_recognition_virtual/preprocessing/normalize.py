def normalize_signal(signal, unit="g"):
    """
    Normalize acceleration signal.

    Parameters:
    signal : pandas Series
        Acceleration values
    unit : str
        'g'   -> already in g
        'ms2' -> convert from m/s^2 to g

    Returns:
    Normalized signal in g units
    """
    if unit == "ms2":
        return signal / 9.81
    return signal
