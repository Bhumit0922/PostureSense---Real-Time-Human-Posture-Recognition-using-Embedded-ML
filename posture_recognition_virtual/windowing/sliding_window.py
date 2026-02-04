def sliding_window(df, win_size=18, step=3):
    """
    Create sliding windows from signal data.

    Parameters:
    df : pandas DataFrame
        DataFrame containing signal data
    win_size : int
        Number of samples per window (350 ms → 18 samples)
    step : int
        Step size (50 ms → 3 samples)

    Returns:
    List of DataFrame windows
    """
    windows = []
    for start in range(0, len(df) - win_size, step):
        window = df.iloc[start:start + win_size]
        windows.append(window)
    return windows
