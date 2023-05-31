import numpy as np
import scipy.signal as signal


def xcorr(x, y, dt, scale="none"):
    """Obtain cross-correlation of signals x and y

    Args:
    x,y: signals
    dt: time step
    scale: 'none','biased','unbiased','coeff'

    Returns:
    lags: lags of correlation
    corr: coefficients of correlation
    """
    # Calculate cross-correlation
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")

    if scale == "biased":
        corr = corr / x.size
    elif scale == "unbiased":
        corr = corr / (x.size - abs(lags))
    elif scale == "coeff":
        corr = corr / np.sqrt(np.dot(x, x) * np.dot(y, y))

    return dt * lags, corr


def period(x, dt):
    """Determine the period of a signal using autocorrelation

    Args:
    x: signal
    dt: time step

    Returns:
    T: period
    """
    # Calculate autocorrelation
    lags, corr = xcorr(x, x, dt, "biased")
    # consider only the positive lags (>0)
    lags_pos = lags[lags > 0]
    corr_pos = corr[lags > 0]
    # Find the peaks of the autocorrelation
    pks = signal.find_peaks(corr_pos)
    pks_idx = pks[0]
    pks_corr = corr_pos[pks_idx]
    pks_lags = lags_pos[pks_idx]
    # Choose period as the lag with the highest peak
    T = pks_lags[np.argmax(pks_corr)]
    return T


def periodic_signal_peaks(x, T):
    """
    Return the indices of the first and last peaks of the signal

    Args:
    x: signal
    T: lower bound of distance between each peak, should be approximately the period
    Returns:
    (start_pk_idx, end_pk_idx): first and last peak indices

    """
    # Find the peaks of the signal
    # use the determined period as a distance lower bound
    # between the peaks
    pks = signal.find_peaks(x, distance=T)
    # Determine the first and last peaks
    pks_idx = pks[0]
    start_pk_idx = pks_idx[0]
    end_pk_idx = pks_idx[-1]
    return (start_pk_idx, end_pk_idx)


def power_spectral_density(x, dt):
    """
    Return the power spectral density of a signal.

    Args:
    x: signal
    dt: sampling time

    Returns:
    omega: fourier frequencies
    psd: one-sided power spectral density

    """
    # Get the signal length and number of signals
    signal_length = len(x)
    # Determine the fourier frequencies
    omega = 1 / dt * 2 * np.pi * np.fft.fftfreq(signal_length)
    # Take the fourier transform of the signal
    X_fft = np.fft.fft(x)

    # Calculate the one-sided spectrum
    if signal_length % 2 == 1:  # signal length odd
        # then we only have 0 frequency at index 0
        psd = 2 * np.abs(X_fft[1 : int((signal_length - 1) / 2) + 1] / signal_length)
        omega = omega[1 : int((signal_length - 1) / 2) + 1]
    elif signal_length % 2 == 0:  # signal length even
        # we have zero frequency at index 0, and pi/-pi frequency at index signal_length/2
        psd = 2 * np.abs(X_fft[1 : int(signal_length / 2)] / signal_length)
        omega = omega[1 : int(signal_length / 2)]
    return omega, psd
