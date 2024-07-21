import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp

plt.style.use("src/stylesheet.mplstyle")
plt.style.use("dark_background")

save_fig = True
plot_optimization = True

cmap1 = cm.create_custom_colormap(type="discrete", map_name="defne")
path_color = cmap1(2)  # "tab:red"
energy_color = cm.create_custom_colormap(type="continuous", map_name="aqua")

path_marker = "o"
lw = 2.5

# N_reservoir = 1200, connectivity = 20
model_path = Path("final_results/rijke/run_20231029_153121")

energy_path = "20231115_143329"

optimization_paths = [
    "20231129_221841",
    "20231129_182324",  # starts beta=4.2, tau=0.34
    "20231129_180629",  # starts beta=4.0, tau=0.25
    "20231129_220614",  # starts beta=5.2, tau=0.32
    "20231129_220654",  # starts beta=5.4, tau=0.12
]

cmap = cm.create_custom_colormap(
    type="continuous", colors=[path_color, "white"], N=len(optimization_paths)
)

# titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# figure_size = (15, 6)
figure_size = (10, 4)
fig = plt.figure(figsize=figure_size, constrained_layout=True)

# print model properties
ener_results = pp.unpickle_file(model_path / f"energy_results_{energy_path}.pickle")[0]
ener_mat_pred = ener_results["J"]
# ener_mat_mean = np.mean(ener_mat_pred, axis=0)  # ensembles x params x 2

beta_list = ener_results["beta_list"]
tau_list = ener_results["tau_list"]

# reshape as a grid rows increasing beta, columns increasing tau
J_mean_grid = ener_mat_pred.reshape(len(beta_list), len(tau_list))
J_mean_grid = J_mean_grid.T

beta, tau = np.meshgrid(beta_list, tau_list)

ax = fig.subplots(1, 2)
beta_increment = beta_list[1] - beta_list[0]
tau_increment = tau_list[1] - tau_list[0]
pos = ax[0].imshow(
    J_mean_grid,
    origin="lower",
    aspect=15,
    vmin=0,
    vmax=100,
    extent=(
        beta_list[0] - beta_increment / 2,
        beta_list[-1] + beta_increment / 2,
        tau_list[0] - tau_increment / 2,
        tau_list[-1] + tau_increment / 2,
    ),
    cmap=energy_color,
)

cbar = fig.colorbar(pos, ax=ax[0], shrink=1.0, ticks=[0, 20, 40, 60, 80, 100])
cbar.ax.set_yticklabels(["$0$", "$20$", "$40$", "$60$", "$80$", "$\geq 100$"])
cbar.ax.set_title("$J$")

ax[0].set_ylabel("$\\tau$")
ax[0].set_xlabel("$\\beta$")
ax[0].set_xticks(np.arange(0.5, 6.0, 0.5))
ax[0].set_yticks(np.arange(0.05, 0.4, 0.05))
ax[0].annotate("(a)", xy=(0.03, 0.95), xycoords="axes fraction")
# ax.set_title("$E_{ac}$")

# ax = fig.add_subplot(1, 1, 1, projection='3d')
# pos = ax.plot_surface(beta, tau, J_mean_grid, cmap=cm.viridis,
#                        linewidth=0, antialiased=False)
# cbar = fig.colorbar(pos, ax=ax, shrink=0.85)
# ax.view_init(azim=225)
# ax.set_ylabel("$\\tau$")
# ax.set_xlabel("$\\beta$")
if plot_optimization:
    for opt_idx, optimization_path in enumerate(optimization_paths):
        opt_results = pp.unpickle_file(
            model_path / f"optimization_results_{optimization_path}.pickle"
        )[0]
        # check energy and cut off iterations
        try:
            valid_iter_idx = np.where(np.array(opt_results["hist"]["J"]) < 1e-4)[0][0]
            p_hist = np.array(opt_results["hist"]["p"])[: valid_iter_idx + 1]
        except:
            p_hist = np.array(opt_results["hist"]["p"])
        ax[0].plot(
            p_hist[0, 0],
            p_hist[0, 1],
            linestyle="",
            marker="o",
            markersize=5,
            color=cmap(opt_idx),
        )
        ax[0].plot(
            p_hist[:-1, 0],
            p_hist[:-1, 1],
            linestyle="",
            marker="o",
            markersize=3,
            color=cmap(opt_idx),
        )
        ax[0].plot(p_hist[:, 0], p_hist[:, 1], linestyle=":", color=cmap(opt_idx))
        ax[0].plot(
            p_hist[-1, 0],
            p_hist[-1, 1],
            linestyle="",
            marker="*",
            markersize=7,
            color=cmap(opt_idx),
        )

        ax[0].quiver(
            p_hist[:-1, 0],
            p_hist[:-1, 1],
            p_hist[1:, 0] - p_hist[:-1, 0],
            p_hist[1:, 1] - p_hist[:-1, 1],
            scale_units="xy",
            angles="xy",
            ec=cmap(opt_idx),
            fc=cmap(opt_idx),
            scale=2.0,
            width=0.005,
            headwidth=5.0,
            zorder=2,
        )
        ax[0].annotate(
            f"$\mathbf{{{opt_idx+1}}}$",
            xy=(p_hist[0, 0] + 0.04, p_hist[0, 1] + 0.004),
            color="white",
        )

        ax[1].plot(
            opt_results["hist"]["J"],
            color=cmap(opt_idx),
            linewidth=lw,
            marker=path_marker,
        )

ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("$J$")
ax[1].legend([f"Path {opt_idx+1}" for opt_idx in range(len(optimization_paths))])
ax[1].annotate("(b)", xy=(-0.1, 0.95), xycoords="axes fraction")
ax[1].grid()

if save_fig:
    fig.savefig(f"paper/graphics_ppt/figure_energy_grid_opt_black.png")
    # fig.savefig("paper/graphics/figure_energy_grid.pdf")

plt.show()
