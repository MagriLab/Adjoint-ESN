import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from adjoint_esn.utils import preprocessing as pp

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)

model_paths = [
    Path("local_results/standard/run_20230914_215555"),
    Path("local_results/standard/run_20231021_170947"),
    Path("local_results/rijke/run_20231009_113025"),
]

beta_list = np.arange(1.0, 6.0, 1.0)
tau_list = np.arange(0.1, 0.35, 0.05)

beta_step = 1
beta_ticks = np.arange(len(beta_list))
beta_ticks = beta_ticks[::beta_step]
beta_ticklabels = [f"{beta_name:.2f}" for beta_name in beta_list[::beta_step]]

tau_step = 1
tau_ticks = np.arange(len(tau_list))
tau_ticks = tau_ticks[::tau_step]
tau_ticklabels = [f"{tau_name:.2f}" for tau_name in tau_list[::tau_step]]

titles = ["(a)", "(b)", "(c)"]
fig = plt.figure(figsize=(15, 5), constrained_layout=True)

for model_idx, model_path in enumerate(model_paths):
    error_results = pp.unpickle_file(model_path / "error_results.pickle")[0]
    error_mat = error_results["error"]  # params x folds x ensemble
    error_mat_mean = np.mean(error_mat, axis=2)  # take the mean over ensembles
    error_mat_mean = np.mean(error_mat_mean, axis=1)  # take mean over folds

    # reshape as a grid rows increasing beta, columns increasing tau
    error_mat_mean_grid = error_mat_mean.reshape(len(beta_list), len(tau_list))

    # plot on the logarithmic scale
    plot_mat = np.log10(error_mat_mean_grid)
    plot_mat = plot_mat.T  # take transpose, now beta are columns, tau are rows

    ax = fig.add_subplot(1, 3, model_idx + 1)
    pos = ax.imshow(plot_mat, vmin=-3, vmax=0, origin="lower")

    # set tau labels
    ax.set_yticks(tau_ticks)
    ax.set_yticklabels(tau_ticklabels)
    ax.set_ylabel("$\\tau$")

    # set beta labels
    ax.set_xticks(beta_ticks)
    ax.set_xticklabels(beta_ticklabels)
    ax.set_xlabel("$\\beta$")

    ax.set_title(titles[model_idx], loc="left")
    cbar = fig.colorbar(pos, ax=ax, shrink=0.75)

plt.show()
