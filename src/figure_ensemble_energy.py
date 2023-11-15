import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, rc

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)
save_fig = True

# N_reservoir = 1200, connectivity = 20
model_path = Path("local_results/rijke/run_20231029_153121")

save_path = "20231115_002644"

# titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

fig = plt.figure(figsize=(5, 5), constrained_layout=True)

# print model properties
ener_results = pp.unpickle_file(model_path / f"energy_results_{save_path}.pickle")[0]
ener_mat_pred = ener_results["J"]
ener_mat_mean = np.mean(ener_mat_pred, axis=0)  # ensembles x params x 2


beta_list = ener_results["beta_list"]
tau_list = ener_results["tau_list"]

beta_step = 5
beta_ticks = np.arange(len(beta_list))
beta_ticks = beta_ticks[::beta_step]
beta_ticklabels = [f"{beta_name:.1f}" for beta_name in beta_list[::beta_step]]

tau_step = 5
tau_ticks = np.arange(len(tau_list))
tau_ticks = tau_ticks[::tau_step]
tau_ticklabels = [f"{tau_name:.2f}" for tau_name in tau_list[::tau_step]]

# reshape as a grid rows increasing beta, columns increasing tau
J_mean_grid = ener_mat_mean.reshape(len(beta_list), len(tau_list))
J_mean_grid = J_mean_grid.T

beta, tau = np.meshgrid(beta_list, tau_list)

ax = fig.add_subplot(1, 1, 1)
pos = ax.imshow(J_mean_grid, origin="lower", aspect=1.5)
cbar = fig.colorbar(pos, ax=ax, shrink=0.75, ticks=np.arange(5, 95, 10))
ax.set_yticks(tau_ticks)
ax.set_yticklabels(tau_ticklabels)
ax.set_ylabel("$\\tau$")

# set beta labels
ax.set_xticks(beta_ticks)
ax.set_xticklabels(beta_ticklabels)
ax.set_xlabel("$\\beta$")

ax.set_title("$J$")

# ax = fig.add_subplot(1, 1, 1, projection='3d')
# pos = ax.plot_surface(beta, tau, J_mean_grid, cmap=cm.viridis,
#                        linewidth=0, antialiased=False)
# cbar = fig.colorbar(pos, ax=ax, shrink=0.85)
# ax.view_init(azim=225)
# ax.set_ylabel("$\\tau$")
# ax.set_xlabel("$\\beta$")

if save_fig:
    fig.savefig("paper/graphics/figure_energy_grid.png", bbox_inches="tight")

plt.show()
