import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import visualizations as vis
from adjoint_esn.utils.enums import get_eVar

plt.style.use("src/stylesheet.mplstyle")

save_fig = True
figsize = (15, 6)
fig_name = 2
model_paths = [
    Path("local_results/rijke/run_20231029_153121"),
    Path("local_results/rijke/run_20231029_153121"),
    Path("local_results/rijke/run_20240307_175258"),
    Path("local_results/rijke/run_20240307_175258"),
    Path("local_results/rijke/run_20240307_175258"),
]
res_cont_names = [
    "20240722_152836",  # integration time 10, lco beta = 2.0, tau = 0.25
    "20240722_122711",  # integration time 10, lco beta = 4.5, tau = 0.12
    "20240722_121829",  # integration time 10, quasiperiodic beta = 6.1, tau = 0.2
    "20240722_061925",  # integration time 2 LT, chaotic beta = 7.6, tau = 0.22
    "20240722_122720",  # integration time 2 LT, chaotic beta = 8.7, tau = 0.23
]
LTs = [1.0, 1.0, 1.0, 8.5, 3.9]

config = post.load_config(model_paths[0])

input_vars = config.model.input_vars
output_vars = config.model.output_vars
N_g = config.simulation.N_g
eInputVar = get_eVar(input_vars, N_g)
eOutputVar = get_eVar(output_vars, N_g)

plt_idx_arr = [eOutputVar.mu_1, eOutputVar.eta_1]

cmap = cm.create_custom_colormap(type="continuous")
true_color = cmap(0.0)  # teal
pred_color = cmap(0.5)  # orange
pred_x0_color = cmap(1.0)  # dark purple

true_lw = 5.0
pred_lw = 2.5
pred_x0_lw = 1.5

true_ls = "-"
pred_ls = "-"
pred_x0_ls = "--"

titles = [f"({chr(i)})" for i in range(ord("a"), ord("z") + 1)]
tt = 0

fig = plt.figure(figsize=figsize, constrained_layout=True)
# subfigs = fig.subfigures(nrows=1, ncols=1)

ax_ = fig.subplots(len(plt_idx_arr), len(model_paths))

for j, (model_path, res_cont_name, LT) in enumerate(
    zip(model_paths, res_cont_names, LTs)
):
    cont_path = model_path / f"init_sensitivity_results_{res_cont_name}.pickle"
    res_cont = pp.unpickle_file(cont_path)[0]

    dJdu0_true = res_cont["dJdu0"]["true"]["adjoint"]
    dJdu0_esn = res_cont["dJdu0"]["esn"]["adjoint"]
    dJdu0_esn_x0 = res_cont["dJdu0"]["esn_from_x0"]["adjoint"]
    beta = res_cont["beta_list"][0]
    tau = res_cont["tau_list"][0]
    if len(model_paths) > 1:
        ax = ax_[:, j]
    else:
        ax = ax_
    for i, plt_idx in enumerate(plt_idx_arr):
        (h0,) = ax[i].plot(
            res_cont["loop_times"] / LT,
            dJdu0_true[0, plt_idx, 0, :],
            color=true_color,
            linestyle=true_ls,
            linewidth=true_lw,
        )

        dJdu0_esn_mean = np.mean(dJdu0_esn[:, 0, plt_idx, 0, :], axis=0)
        dJdu0_esn_std = np.std(dJdu0_esn[:, 0, plt_idx, 0, :], axis=0)
        (h1,) = ax[i].plot(
            res_cont["loop_times"] / LT,
            dJdu0_esn_mean,
            color=pred_color,
            linestyle=pred_ls,
            linewidth=pred_lw,
        )
        ax[i].fill_between(
            res_cont["loop_times"] / LT,
            dJdu0_esn_mean + dJdu0_esn_std,
            dJdu0_esn_mean - dJdu0_esn_std,
            alpha=0.2,
            color=pred_color,
            antialiased=True,
            zorder=2,
        )

        dJdu0_esn_x0_mean = np.mean(dJdu0_esn_x0[:, 0, plt_idx, 0, :], axis=0)
        dJdu0_esn_x0_std = np.std(dJdu0_esn_x0[:, 0, plt_idx, 0, :], axis=0)
        (h2,) = ax[i].plot(
            res_cont["loop_times"] / LT,
            dJdu0_esn_x0_mean,
            color=pred_x0_color,
            linestyle=pred_x0_ls,
            linewidth=pred_x0_lw,
        )
        ax[i].fill_between(
            res_cont["loop_times"] / LT,
            dJdu0_esn_x0_mean + dJdu0_esn_x0_std,
            dJdu0_esn_x0_mean - dJdu0_esn_x0_std,
            alpha=0.2,
            color=pred_x0_color,
            antialiased=True,
            zorder=2,
        )

        if j == 0:
            ax[i].set_ylabel(f"$dJ/d\\{plt_idx.name}$")

        ax[i].annotate(titles[tt], xy=(0.03, 0.85), xycoords="axes fraction")
        tt += 1

        ax[i].grid()
        if i < len(plt_idx_arr) - 1:
            ax[i].set_xticklabels([])

    ax[0].set_title(f"$\\beta = {beta:.2f}, \; \\tau = {tau:.2f}$")
    if LT == 1.0:
        ax[-1].set_xlabel("Integration time")
    else:
        ax[-1].set_xlabel("Integration time [LT]")

plt.figlegend(
    [h0, h1, h2],
    [
        "True",
        "ESN $dJ/d\mathbf{y}_{in}(0)$",
        "ESN $(dJ/d\mathbf{x}(0))\mathbf{W}_{out}^{-1}$",
    ],
    loc="upper center",
    ncols=3,
    bbox_to_anchor=(0.5, 1.1),
)
if save_fig:
    fig.savefig(
        f"paper/graphics/figure_init_cont_sensitivity_{fig_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
plt.show()
