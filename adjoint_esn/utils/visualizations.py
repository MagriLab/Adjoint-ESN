import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats


def set_labels(xlabel, ylabel, legend, title):
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)
    if title:
        plt.title(title)
    return


def get_line_specs(line_specs, i):
    line_specs_ = {}
    [line_specs_.update({var: line_specs[var][i]}) for var in line_specs.keys()]
    return line_specs_


def plot_lines(x, *y, xlabel=None, ylabel=None, legend=None, title=None, **line_specs):
    for i, yy in enumerate(y):
        plt.plot(x, yy, **get_line_specs(line_specs, i))
    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_reverse_lines(
    *x, y, xlabel=None, ylabel=None, legend=None, title=None, **line_specs
):
    for i, xx in enumerate(x):
        plt.plot(xx, y, **get_line_specs(line_specs, i))
    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_phase_space(
    *y, idx_pair, xlabel=None, ylabel=None, legend=None, title=None, **line_specs
):
    for i, yy in enumerate(y):
        plt.plot(
            yy[:, idx_pair[0]], yy[:, idx_pair[1]], **get_line_specs(line_specs, i)
        )
    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_statistics(
    *y, xlabel=None, ylabel=None, legend=None, title=None, **line_specs
):
    # Histogram option
    n_bins = 100

    for i, yy in enumerate(y):
        density = stats.gaussian_kde(yy)
        n, x = np.histogram(yy, n_bins, density=True)
        plt.plot(x, density(x), **get_line_specs(line_specs, i))

    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_statistics_ensemble(
    *y, y_base, xlabel=None, ylabel=None, legend=None, title=None, **line_specs
):
    # Histogram option
    n_bins = 100

    density_base = stats.gaussian_kde(y_base)
    _, x = np.histogram(y_base, n_bins, density=True)
    plt.plot(x, density_base(x), **get_line_specs(line_specs, 0))

    density_arr = np.zeros((len(y), n_bins + 1))
    for i, yy in enumerate(y):
        density = stats.gaussian_kde(yy)
        density_arr[i] = density(x)[None, :]
    density_mean = np.mean(density_arr, axis=0)
    plt.plot(x, density_mean, **get_line_specs(line_specs, 1))

    density_std = np.std(density_arr, axis=0)
    plt.fill_between(
        x,
        density_mean - density_std,
        density_mean + density_std,
        alpha=0.2,
        color=line_specs["color"][1],
        antialiased=True,
        zorder=2,
    )
    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_bifurcation_diagram(x, y, **line_specs):
    pks = signal.find_peaks(y)[0]
    plt.plot(x * np.ones(len(pks)), y[pks], **line_specs)
    return y[pks]


def plot_asd_ensemble(
    *asd_y,
    asd_y_base,
    omega,
    range=10,
    xlabel=None,
    ylabel=None,
    legend=None,
    title=None,
    **line_specs
):
    plt.plot(omega, asd_y_base, **get_line_specs(line_specs, 0))

    asd_y_mean = np.mean(asd_y, axis=0)
    plt.plot(omega, asd_y_mean, **get_line_specs(line_specs, 1))

    asd_y_std = np.std(asd_y, axis=0)
    plt.fill_between(
        omega,
        asd_y_mean - asd_y_std,
        asd_y_mean + asd_y_std,
        alpha=0.2,
        color=line_specs["color"][1],
        antialiased=True,
        zorder=2,
    )
    max_freq = omega[np.argmax(asd_y_base)]
    min_xlim = max(0, max_freq - range / 2)
    max_xlim = min_xlim + range
    plt.xlim([min_xlim, max_xlim])

    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_asd(
    asd_y,
    omega_y,
    asd_y_base,
    omega_y_base,
    range=10,
    xlabel=None,
    ylabel=None,
    legend=None,
    title=None,
    **line_specs
):
    plt.plot(omega_y_base, asd_y_base, **get_line_specs(line_specs, 0))

    plt.plot(omega_y, asd_y, **get_line_specs(line_specs, 1))

    max_freq = omega_y_base[np.argmax(asd_y_base)]
    min_xlim = max(0, max_freq - range / 2)
    max_xlim = min_xlim + range
    plt.xlim([min_xlim, max_xlim])

    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return
