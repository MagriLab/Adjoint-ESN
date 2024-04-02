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

plt.style.use("src/stylesheet.mplstyle")
cmap = cm.create_custom_colormap(type="discrete")

figure_size = (15, 4)

save_fig = False

true_color = cmap(0)
pred_color = cmap(1)
pred_color2 = cmap(2)

marker = "o"
markersize = 1.5

model_paths = [
    Path(
        "local_results/rijke/run_20231029_153121"
    ),  # rijke with reservoir, trained on beta = 1,2,3,4,5
    Path(
        "local_results/rijke/run_20240307_175258"
    ),  # rijke with reservoir, trained on beta = 6,6.5,7,7.5,8
]
path_str = "bifn_results"
beta_paths = ["20240401_163839", "20240401_163950"]
tau_paths = ["20240401_164121", "20240401_164506"]
res_paths = [beta_paths, tau_paths]
res_list = [[None] * len(res_paths) for _ in range(len(model_paths))]
for model_idx in range(len(model_paths)):
    for res_idx in range(len(res_paths)):
        res_list[model_idx][res_idx] = pp.unpickle_file(
            model_paths[model_idx]
            / f"{path_str}_{res_paths[res_idx][model_idx]}.pickle"
        )[0]

# # res_beta = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240325_154448.pickle'))[0] # tau = 0.2
# # res_beta = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240325_124140.pickle'))[0] # tau = 0.22
# # res_beta = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240325_141144.pickle'))[0] # tau = 0.23
# res_beta = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240327_034450.pickle'))[0] # tau = 0.22

# # res_tau = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240325_152355.pickle'))[0] # beta = 7.6
# # res_tau = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240325_172349.pickle'))[0] # beta = 8.7
# res_tau = pp.unpickle_file(Path('local_results/rijke/bifn_results_20240327_012236.pickle'))[0] # beta = 7.6

# res_list = [res_beta, res_tau]

# plot_names = ['mu_1','mu_2','E_ac']
# plot_names_ltx = ['\\mu_1','\\mu_2','E_{ac}']

plot_names = ["mu_1", "E_ac"]
plot_names_ltx = ["\\mu_1", "E_{ac}"]
esn_idx = 0

titles = [f"({chr(i)})" for i in range(ord("a"), ord("z") + 1)]
tt = 0
fig = plt.figure(figsize=figure_size, constrained_layout=True)
subfigs = fig.subfigures(nrows=1, ncols=len(res_list))

for res_idx in range(len(res_paths)):
    beta_list = res_list[0][res_idx]["beta_list"]
    tau_list = res_list[0][res_idx]["tau_list"]

    if len(beta_list) > 1 and len(tau_list) == 1:
        vary_param = "beta"
        p_list = beta_list
    elif len(beta_list) == 1 and len(tau_list) > 1:
        vary_param = "tau"
        p_list = tau_list

    plt_idx_arr = [
        np.where(np.array(res_list[0][res_idx]["plot_names"]) == plot_name)[0][0]
        for plot_name in plot_names
    ]

    ax = subfigs[res_idx].subplots(nrows=3, ncols=len(plot_names))

    # track minimum and maximum values of the peaks
    pks_min = np.inf * np.ones((len(p_list), len(plt_idx_arr)))
    pks_max = -np.inf * np.ones((len(p_list), len(plt_idx_arr)))

    for p_idx, p in enumerate(p_list):
        for j, plt_idx in enumerate(plt_idx_arr):
            if len(plt_idx_arr) == 1:
                ax0 = ax[0]
            else:
                ax0 = ax[0, j]
            plt.sca(ax0)
            pks = res_list[0][res_idx]["true_peaks"][p_idx][
                plt_idx
            ]  # [n_params,n_plots]
            min_pks_true = np.min(pks)
            max_pks_true = np.max(pks)

            plt.plot(
                p * np.ones(len(pks)),
                pks,
                color=true_color,
                marker=marker,
                markersize=markersize,
                linestyle="None",
            )
            plt.ylabel(f"Max(${plot_names_ltx[j]}$)")
            ax0.set_xticklabels([])

            if len(plt_idx_arr) == 1:
                ax1 = ax[1]
            else:
                ax1 = ax[1, j]
            plt.sca(ax1)
            pks_pred = res_list[0][res_idx]["pred_peaks"][p_idx][plt_idx][
                esn_idx
            ]  # [n_params,n_plots,n_models,n_ensemble]
            plt.plot(
                p * np.ones(len(pks_pred)),
                pks_pred,
                color=pred_color,
                marker=marker,
                markersize=markersize,
                linestyle="None",
            )
            plt.ylabel(f"Max(${plot_names_ltx[j]}$)")
            min_pks_pred = np.min(pks_pred)
            max_pks_pred = np.max(pks_pred)
            ax1.set_xticklabels([])

            if len(plt_idx_arr) == 1:
                ax2 = ax[2]
            else:
                ax2 = ax[2, j]
            plt.sca(ax2)
            pks_pred2 = res_list[1][res_idx]["pred_peaks"][p_idx][plt_idx][
                esn_idx
            ]  # [n_params,n_plots,n_models,n_ensemble]
            plt.plot(
                p * np.ones(len(pks_pred2)),
                pks_pred2,
                color=pred_color2,
                marker=marker,
                markersize=markersize,
                linestyle="None",
            )
            plt.xlabel(f"$\\{vary_param}$")
            plt.ylabel(f"Max(${plot_names_ltx[j]}$)")
            min_pks_pred2 = np.min(pks_pred2)
            max_pks_pred2 = np.max(pks_pred2)

            pks_min[p_idx, j] = min(
                pks_min[p_idx, j], min([min_pks_true, min_pks_pred, min_pks_pred2])
            )
            pks_max[p_idx, j] = max(
                pks_max[p_idx, j], max([max_pks_true, max_pks_pred, max_pks_pred2])
            )

    for j, k in enumerate(plt_idx_arr):
        for i in range(ax.shape[0]):
            ylims = [
                np.min(pks_min[:, j]) - 0.05 * np.max(pks_max[:, j]),
                1.2 * np.max(pks_max[:, j]),
            ]
            if len(plt_idx_arr) == 1:
                ax[i].set_ylim(ylims)
                ax[i].grid()
                ax[i].annotate(titles[tt], xy=(0.03, 0.85), xycoords="axes fraction")
            else:
                ax[i, j].set_ylim(ylims)
                ax[i, j].grid()
                ax[i, j].annotate(titles[tt], xy=(0.03, 0.85), xycoords="axes fraction")
            tt += 1
if save_fig:
    fig.savefig(f"paper/graphics/figure_bifn_diagram2.png", bbox_inches="tight")
plt.show()
