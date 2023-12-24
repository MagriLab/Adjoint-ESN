import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches, rc

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)
save_fig = True
plot_optimization = True

# N_reservoir = 1200, connectivity = 20
model_path = Path("local_results/rijke/run_20231029_153121")

energy_path = "20231115_143329"

optimization_paths = [
    "20231129_180629",  # starts beta=4.0, tau=0.25
    "20231129_182324",  # starts beta=4.2, tau=0.34
    # "20231129_185408",
    # "20231129_185415",
    "20231129_220614",  # starts beta=5.2, tau=0.32
    "20231129_220654",  # starts beta=5.4, tau=0.12
    "20231129_221841",
]

# titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

fig = plt.figure(figsize=(5, 5), constrained_layout=True)

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

ax = fig.add_subplot(1, 1, 1)
beta_increment = beta_list[1] - beta_list[0]
tau_increment = tau_list[1] - tau_list[0]
pos = ax.imshow(
    J_mean_grid,
    origin="lower",
    aspect=15,
    vmin=0,
    vmax=60,
    extent=(
        beta_list[0] - beta_increment / 2,
        beta_list[-1] + beta_increment / 2,
        tau_list[0] - tau_increment / 2,
        tau_list[-1] + tau_increment / 2,
    ),
)

cbar = fig.colorbar(pos, ax=ax, shrink=0.75, ticks=[0, 10, 20, 30, 40, 50, 60])
cbar.ax.set_yticklabels(["$0$", "$10$", "$20$", "$30$", "$40$", "$50$", "$\geq 60$"])

ax.set_ylabel("$\\tau$")
ax.set_xlabel("$\\beta$")
ax.set_xticks(np.arange(0.5, 6.0, 0.5))
ax.set_yticks(np.arange(0.05, 0.4, 0.05))
ax.set_title("$E_{ac}$")

# ax = fig.add_subplot(1, 1, 1, projection='3d')
# pos = ax.plot_surface(beta, tau, J_mean_grid, cmap=cm.viridis,
#                        linewidth=0, antialiased=False)
# cbar = fig.colorbar(pos, ax=ax, shrink=0.85)
# ax.view_init(azim=225)
# ax.set_ylabel("$\\tau$")
# ax.set_xlabel("$\\beta$")
path_color = "tab:red"
if plot_optimization:
    for optimization_path in optimization_paths:
        opt_results = pp.unpickle_file(
            model_path / f"optimization_results_{optimization_path}.pickle"
        )[0]
        # check energy and cut off iterations
        try:
            valid_iter_idx = np.where(np.array(opt_results["hist"]["J"]) < 1e-4)[0][0]
            p_hist = np.array(opt_results["hist"]["p"])[: valid_iter_idx + 1]
        except:
            p_hist = np.array(opt_results["hist"]["p"])

        ax.plot(
            p_hist[0, 0],
            p_hist[0, 1],
            linestyle="",
            marker="o",
            markersize=5,
            color=path_color,
        )
        ax.plot(
            p_hist[:-1, 0],
            p_hist[:-1, 1],
            linestyle="",
            marker="o",
            markersize=3,
            color=path_color,
        )
        ax.plot(p_hist[:, 0], p_hist[:, 1], linestyle=":", color=path_color)
        ax.plot(
            p_hist[-1, 0],
            p_hist[-1, 1],
            linestyle="",
            marker="*",
            markersize=7,
            color=path_color,
        )

        ax.quiver(
            p_hist[:-1, 0],
            p_hist[:-1, 1],
            p_hist[1:, 0] - p_hist[:-1, 0],
            p_hist[1:, 1] - p_hist[:-1, 1],
            scale_units="xy",
            angles="xy",
            ec=path_color,
            fc=path_color,
            scale=2.0,
            width=0.005,
            headwidth=5.0,
            zorder=2,
        )
ax.annotate("1", xy=(4.0, 0.26))
ax.annotate("2", xy=(5.2, 0.33))
if save_fig:
    fig.savefig("paper/graphics/figure_energy_grid.png", bbox_inches="tight")

plt.show()
