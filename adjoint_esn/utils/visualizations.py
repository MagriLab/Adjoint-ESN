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


def plot_bifurcation_diagram(x, y, **line_specs):
    pks = signal.find_peaks(y)[0]
    plt.plot(x * np.ones(len(pks)), y[pks], **line_specs)
    return y[pks]
