import matplotlib.animation as animation
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


def plot_lorenz63_attractor(fig, U, U_pred, length, colors, animate=False, legend=None):
    """A function to plot input data of Lorenz 63

    Args:
        U (array): 3-dim Input Time Series
        length (scalar): length of Plot 1

    Return:
        Plot 1 : 3d Plot of Attractor
        Plot 2 : Time Series wrt Number of Steps
        Plot 3 : Time Series wrt Lyapunov Time
        Plot 4 : Convection Current and Thermal Plots
    """

    def update(num, ax1, ax2, fig):
        ax1.view_init(azim=num)
        ax2.view_init(azim=num)
        return (fig,)

    # 3D PLOT OF LORENZ 63 ATTRACTOR
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_xlabel("$x$", labelpad=15)
    ax1.set_ylabel("$y$", labelpad=15)
    ax1.set_zlabel("$z$", labelpad=15)
    if isinstance(U, list):
        for i, UU in enumerate(U):
            ax1.plot(*UU[:length, :].T, lw=1.0, c=colors[0][i])
    else:
        ax1.plot(*U[:length, :].T, lw=1.0, c=colors[0])
    # ax.set_title("True")
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    zlims = ax1.get_zlim()
    ax1.dist = 10
    ax1.set_box_aspect([1, 1, 1.1])
    plt.tight_layout()
    plt.grid(False)
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax1.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax1.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax1.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_xlabel("$x$", labelpad=15)
    ax2.set_ylabel("$y$", labelpad=15)
    ax2.set_zlabel("$z$", labelpad=15)
    if isinstance(U_pred, list):
        for i, UU_pred in enumerate(U_pred):
            ax2.plot(*UU_pred[:length, :].T, lw=1.0, c=colors[1][i])
    else:
        ax2.plot(*U_pred[:length, :].T, lw=1.0, c=colors[1])
    # ax.set_title("Prediction")
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.set_zlim(zlims)
    ax2.dist = 10
    ax2.set_box_aspect([1, 1, 1.1])
    plt.tight_layout()
    plt.grid(False)
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax2.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax2.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax2.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    if legend:
        ax1.legend(legend, loc="upper right", fontsize=14)
        ax2.legend(legend, loc="upper right", fontsize=14)

    if animate:
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(0, 360, 2),
            fargs=(ax1, ax2, fig),
            interval=100,
            blit=False,
        )
    else:
        ani = None
    return ani


def plot_statistics(
    *y,
    n_bins=100,
    xlabel=None,
    ylabel=None,
    legend=None,
    title=None,
    **line_specs,
):
    # Histogram option
    for i, yy in enumerate(y):
        density = stats.gaussian_kde(yy)
        n, x = np.histogram(yy, n_bins, density=True)
        plt.plot(x, density(x), **get_line_specs(line_specs, i))

    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return


def plot_statistics_ensemble(
    *y,
    y_base,
    orientation="var_on_x",
    n_bins=100,
    xlabel=None,
    ylabel=None,
    legend=None,
    title=None,
    **line_specs,
):
    # Histogram option
    density_base = stats.gaussian_kde(y_base)
    x_base = np.linspace(np.min(y_base), np.max(y_base), n_bins)
    # check if normalized
    # pdf_sum = np.trapz(density_base(x_base), x_base)
    # print(pdf_sum)
    if orientation == "var_on_x":
        plt.plot(x_base, density_base(x_base), **get_line_specs(line_specs, 0))
    elif orientation == "var_on_y":
        plt.plot(density_base(x_base), x_base, **get_line_specs(line_specs, 0))

    if isinstance(y[0], list):
        for i in range(len(y)):
            x = np.linspace(np.min(y[i]), np.max(y[i]), n_bins)
            density_arr = np.zeros((len(y[i]), n_bins))
            for j, yy in enumerate(y[i]):
                density = stats.gaussian_kde(yy)
                density_arr[j] = density(x)[None, :]
            density_mean = np.mean(density_arr, axis=0)
            density_std = np.std(density_arr, axis=0)
            if orientation == "var_on_x":
                plt.plot(x, density_mean, **get_line_specs(line_specs, 1 + i))
                plt.fill_between(
                    x,
                    density_mean - density_std,
                    density_mean + density_std,
                    alpha=0.2,
                    color=line_specs["color"][1 + i],
                    antialiased=True,
                    zorder=2,
                )
            elif orientation == "var_on_y":
                plt.plot(density_mean, x, **get_line_specs(line_specs, 1 + i))
                plt.fill_betweenx(
                    x,
                    density_mean - density_std,
                    density_mean + density_std,
                    alpha=0.2,
                    color=line_specs["color"][1 + i],
                    antialiased=True,
                    zorder=2,
                )
    else:
        x = np.linspace(np.min(y), np.max(y), n_bins)
        density_arr = np.zeros((len(y), n_bins))
        for i, yy in enumerate(y):
            density = stats.gaussian_kde(yy)
            density_arr[i] = density(x)[None, :]
        density_mean = np.mean(density_arr, axis=0)
        density_std = np.std(density_arr, axis=0)
        if orientation == "var_on_x":
            plt.plot(x, density_mean, **get_line_specs(line_specs, 1))
            plt.fill_between(
                x,
                density_mean - density_std,
                density_mean + density_std,
                alpha=0.2,
                color=line_specs["color"][1],
                antialiased=True,
                zorder=2,
            )
        elif orientation == "var_on_y":
            plt.plot(density_mean, x, **get_line_specs(line_specs, 1))
            plt.fill_betweenx(
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
    **line_specs,
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
    alpha=1.0,
    **line_specs,
):
    plt.plot(omega_y_base, asd_y_base, **get_line_specs(line_specs, 0))
    if isinstance(asd_y, list):
        for i, (asd_yy, omega_yy) in enumerate(zip(asd_y, omega_y)):
            plt.plot(omega_yy, asd_yy, **get_line_specs(line_specs, 1 + i), alpha=alpha)
    else:
        plt.plot(omega_y, asd_y, **get_line_specs(line_specs, 1))

    max_freq = omega_y_base[np.argmax(asd_y_base)]
    min_xlim = max(0, max_freq - range / 2)
    max_xlim = min_xlim + range
    plt.xlim([min_xlim, max_xlim])

    min_ylim = -0.05 * np.max(asd_y_base)
    max_ylim = 1.2 * np.max(asd_y_base)
    plt.ylim([min_ylim, max_ylim])

    plt.grid()
    set_labels(xlabel, ylabel, legend, title)
    return
