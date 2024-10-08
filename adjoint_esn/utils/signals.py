import numpy as np
import scipy.signal as signal

from adjoint_esn.utils import preprocessing as pp


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


def amplitude_spectrum(x, dt):
    """
    Return the amplitude spectrum of a signal.

    Args:
    x: signal
    dt: sampling time

    Returns:
    omega: fourier frequencies
    asd: one-sided amplitude spectrum

    """
    # Get the signal length and number of signals
    N = len(x)
    # Determine the fourier frequencies
    omega = 1 / dt * 2 * np.pi * np.fft.fftfreq(N)
    # Take the fourier transform of the signal
    X_fft = np.fft.fft(x)

    # Calculate the one-sided spectrum
    if N % 2 == 1:  # signal length odd
        # then we only have 0 frequency at index 0, and nyquist frequency (pi, -pi) occurs twice
        X_fft_1 = X_fft[0 : int((N - 1) / 2) + 1]
        A = (1 / N) * np.abs(X_fft_1)
        A[1:] = 2 * A[1:]
        omega = omega[0 : int((N - 1) / 2) + 1]

    elif N % 2 == 0:  # signal length even
        # we have zero frequency at index 0, and nyquist frequency -pi at index signal_length/2
        X_fft_1 = X_fft[0 : int(N / 2) + 1]
        A = (1 / N) * np.abs(X_fft_1)
        A[1:-1] = (
            2 * A[1:-1]
        )  # zero frequency (DC) and the nyquist frequency do not occur twice
        omega = omega[0 : int(N / 2) + 1]
        omega[-1] = -omega[-1]  # change from -pi to pi
    return omega, A


def get_amp_spec(dt, y, remove_mean=True, periodic=False):
    # remove mean
    if remove_mean == True:
        y = y - np.mean(y)
    if periodic:
        T_period = period(y, dt)
        data_omega = 2 * np.pi / T_period
        print("Omega = ", data_omega)
        print("Period = ", T_period)
        # take the maximum number of periods
        # the real period isn't an exact multiple of the sampling time
        # therefore, the signal doesn't repeat itself at exact integer indices
        # so calculating the number of time steps in each period
        # does not work in order to cut the signal at the maximum number of periods
        # that's why we will cut between peaks, which is a more reliable option
        # though still not exact
        min_dist = pp.get_steps(T_period - 0.1, dt)
        (start_pk_idx, end_pk_idx) = periodic_signal_peaks(y, T=min_dist)
        y_pre_fft = y[
            start_pk_idx:end_pk_idx
        ]  # don't include end peak for continuous signal
    else:
        y_pre_fft = y

    # find asd
    omega, amp_spec = amplitude_spectrum(y_pre_fft, dt)
    return omega, amp_spec


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
    N = len(x)  # signal length
    fs = 1 / dt  # sampling frequency
    # Determine the fourier frequencies
    omega = fs * 2 * np.pi * np.fft.fftfreq(N)
    # Take the fourier transform of the signal
    X_fft = np.fft.fft(x)

    # Calculate the one-sided spectrum
    if N % 2 == 1:  # signal length odd
        # then we only have 0 frequency at index 0, and nyquist frequency (pi, -pi) occurs twice
        X_fft_1 = X_fft[0 : int((N - 1) / 2) + 1]
        psd = (1 / (fs * N)) * np.abs(X_fft_1) ** 2
        psd[1:] = 2 * psd[1:]
        omega = omega[0 : int((N - 1) / 2) + 1]

    elif N % 2 == 0:  # signal length even
        # we have zero frequency at index 0, and nyquist frequency -pi at index signal_length/2
        X_fft_1 = X_fft[0 : int(N / 2) + 1]
        psd = (1 / (fs * N)) * np.abs(X_fft_1) ** 2
        psd[1:-1] = (
            2 * psd[1:-1]
        )  # zero frequency (DC) and the nyquist frequency do not occur twice
        omega = omega[0 : int(N / 2) + 1]
        omega[-1] = -omega[-1]  # change from -pi to pi
    return omega, psd
