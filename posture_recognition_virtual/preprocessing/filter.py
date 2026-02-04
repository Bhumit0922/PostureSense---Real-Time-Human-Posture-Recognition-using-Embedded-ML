from scipy.signal import firwin, lfilter

def low_pass_filter(signal, fs=50, cutoff=10):
    """
    Apply FIR low-pass filter to signal.

    Parameters:
    signal : pandas Series
        Input signal
    fs : int
        Sampling frequency (Hz)
    cutoff : int
        Cutoff frequency (Hz)

    Returns:
    Filtered signal
    """
    nyquist = fs / 2
    taps = firwin(numtaps=9, cutoff=cutoff / nyquist)
    filtered_signal = lfilter(taps, 1.0, signal)
    return filtered_signal
