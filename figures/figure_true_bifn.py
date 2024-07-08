import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

plt.style.use("src/stylesheet2.mplstyle")
cmap = cm.create_custom_colormap(type="discrete")

figure_size = (15, 7)

save_fig = True

true_color = cmap(0)
marker = "o"
markersize = 3

res = pp.unpickle_file(
    Path("final_results/rijke/run_20240307_175258/bifn_results_20240402_221857.pickle")
)[0]
beta_list = res["beta_list"]
tau_list = res["tau_list"]

if len(beta_list) > 1 and len(tau_list) == 1:
    vary_param = "beta"
    p_list = beta_list
elif len(beta_list) == 1 and len(tau_list) > 1:
    vary_param = "tau"
    p_list = tau_list

plot_names = ["mu_1"]
plot_names_ltx = ["\\mu_1", "E_{ac}"]
plt_idx_arr = [
    np.where(np.array(res["plot_names"]) == plot_name)[0][0] for plot_name in plot_names
]

# track minimum and maximum values of the peaks
pks_min = np.inf * np.ones((len(p_list), len(plt_idx_arr)))
pks_max = -np.inf * np.ones((len(p_list), len(plt_idx_arr)))

fig = plt.figure(figsize=figure_size)
ax = fig.subplots(len(plt_idx_arr), 1)
for p_idx, p in enumerate(p_list):
    for j, plt_idx in enumerate(plt_idx_arr):
        if len(plt_idx_arr) == 1:
            ax0 = ax
        else:
            ax0 = ax[j]
        pks = res["true_peaks"][p_idx][plt_idx]  # [n_params,n_plots]
        min_pks = np.min(pks)
        max_pks = np.max(pks)

        ax0.plot(
            p * np.ones(len(pks)),
            pks,
            color=true_color,
            marker=marker,
            markersize=markersize,
            linestyle="None",
        )
        ax0.set_ylabel(f"Max(${plot_names_ltx[j]}$)")
        ax0.set_xlabel(f"$\\{vary_param}$")
        pks_min[p_idx, j] = min(pks_min[p_idx, j], min_pks)
        pks_max[p_idx, j] = max(pks_max[p_idx, j], max_pks)

for j, k in enumerate(plt_idx_arr):
    if len(plt_idx_arr) == 1:
        ax0 = ax
    else:
        ax0 = ax[j]
    ylims = [
        np.min(pks_min[:, j]) - 0.05 * np.max(pks_max[:, j]),
        1.2 * np.max(pks_max[:, j]),
    ]
    ax0.set_ylim(ylims)
    ax0.grid()
if save_fig:
    plt.savefig(f"paper/graphics_ppt/true_bifn_diagram.png", dpi=300)
plt.show()
