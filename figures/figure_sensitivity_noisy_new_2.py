import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import adjoint_esn.utils.postprocessing as post
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam

plt.style.use("src/stylesheet.mplstyle")
cmap = cm.create_custom_colormap(type="discrete")

save_fig = True
fig_name = "sensitivity_noisy_beta"
# N_reservoir = 1200, connectivity = 20
which_idx = [0, 3, 5]
order_idx = [0, 2, 1]

model_paths = [
    Path("local_results/rijke/run_20231029_153121_noise_5_new_2"),  # 0
    Path("local_results/rijke/run_20231029_153121_noise_5_tikh_5_new_2"),  # 1
    Path("local_results/rijke/run_20231029_153121_noise_5_tikh_4_new_2"),  # 2
    Path("local_results/rijke/run_20231029_153121_noise_5_tikh_3_new_2"),  # 3
    Path("local_results/rijke/run_20231029_153121_noise_5_tikh_2_new_2"),  # 4
    Path("local_results/rijke/run_20231029_153121_noise_5_tikh_1_new_2"),  # 5
]
model_paths = [model_paths[idx] for idx in which_idx]

# not same washout, eta_1_init = 1.5
save_paths_beta = [
    [
        "20240725_015053",
        "",
        "",
        "20240724_100945",
        "20240724_103122",
        "20240724_224900",
    ],  # tau = 0.07
    [
        "20240724_205245",
        "",
        "",
        "20240724_061206",
        "20240724_053634",
        "20240724_180642",
    ],  # tau = 0.12
    [
        "20240724_211354",
        "",
        "",
        "20240724_103808",
        "20240724_084505",
        "20240724_211938",
    ],  # tau = 0.22
    [
        "20240725_051720",
        "",
        "",
        "20240724_120601",
        "20240724_125506",
        "20240724_203629",
    ],  # tau = 0.32
]
save_paths_tau = [
    [
        "20240724_190921",
        "",
        "",
        "20240724_030806",
        "20240724_023958",
        "20240724_145803",
    ],  # beta = 1.25
    [
        "20240724_182351",
        "",
        "",
        "20240724_063151",
        "20240724_062848",
        "20240724_183920",
    ],  # beta = 2.5
    [
        "20240724_181907",
        "",
        "",
        "20240724_033734",
        "20240724_031217",
        "20240724_154259",
    ],  # beta = 3.75
    [
        "20240724_180653",
        "",
        "",
        "20240724_074711",
        "20240724_071006",
        "20240724_202729",
    ],  # beta = 4.5
]
save_paths_beta = [
    [save_paths_beta[res_idx][idx] for idx in which_idx] for res_idx in range(4)
]
save_paths_tau = [
    [save_paths_tau[res_idx][idx] for idx in which_idx] for res_idx in range(4)
]

# reorder
model_paths = [model_paths[idx] for idx in order_idx]
save_paths_beta = [
    [save_paths_beta[res_idx][idx] for idx in order_idx] for res_idx in range(4)
]
save_paths_tau = [
    [save_paths_tau[res_idx][idx] for idx in order_idx] for res_idx in range(4)
]

# add legend
legend_str = ["True"]
legend_str_lambda = [
    "$\lambda = 10^{-6}$",
    "$\lambda = 10^{-5}$",
    "$\lambda = 10^{-4}$",
    "$\lambda = 10^{-3}$",
    "$\lambda = 10^{-2}$",
    "$\lambda = 10^{-1}$",
]
legend_str_lambda = [legend_str_lambda[idx] for idx in which_idx]
[legend_str.append(legend_str_lambda[idx]) for idx in order_idx]

# colors
true_color = cmap(0)
true_lw = 5.0
true_ls = "-"
pred_colors = [cmap(4), cmap(1), cmap(2)]
pred_lws = [2.5, 2.5, 2.5]
pred_lss = ["-", "--", "-."]

true_marker = "none"
pred_markers = ["none", "none", "none"]
true_ms = 6
pred_mss = [8, 8, 8]

titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

fig = plt.figure(figsize=(15, 3), constrained_layout=True)

for k, plot_name in enumerate(["varying_beta"]):
    if plot_name == "varying_beta":
        save_paths = save_paths_beta
        # param_list = eParam
        vary_param = "beta"
        not_vary_param = "tau"
    elif plot_name == "varying_tau":
        save_paths = save_paths_tau
        # param_list = eParam
        vary_param = "tau"
        not_vary_param = "beta"
    for j, save_path in enumerate(save_paths):
        i = eParam[vary_param]
        ax = plt.subplot(1, len(save_paths), j + 1 + len(save_paths) * k)
        ax.set_title(titles[j + len(save_paths) * k], loc="left")

        dJdp_mean = [None] * len(model_paths)
        dJdp_std = [None] * len(model_paths)
        for model_idx, model_path in enumerate(model_paths):
            if save_path[model_idx] != "":
                sens_results = pp.unpickle_file(
                    model_path / f"sensitivity_results_{save_path[model_idx]}.pickle"
                )[0]
                dJdp_esn = sens_results["dJdp"]["esn"]["adjoint"]
                dJdp_mean[model_idx] = np.mean(dJdp_esn, axis=0)
                dJdp_std[model_idx] = np.std(dJdp_esn, axis=0)
                p_list = pp.make_param_mesh(
                    [sens_results["beta_list"], sens_results["tau_list"]]
                )
                if dJdp_esn.ndim == 5:  # flatten the last columns (loop_time, loop_idx)
                    dJdp_mean[model_idx] = dJdp_mean[model_idx][:, :, 0, 0]
                    dJdp_std[model_idx] = dJdp_std[model_idx][:, :, 0, 0]
                # for i in param_list:
                #     ax=plt.subplot(2,2,2*k+1+i)
        if save_path[model_idx] != "":
            dJdp_true = sens_results["dJdp"]["true"]["adjoint"][:, i]
            if dJdp_true.ndim == 3:
                dJdp_true = dJdp_true[:, 0, 0]
            vis.plot_lines(
                p_list[:, eParam[vary_param]],
                dJdp_true,
                *[dJdp_mean[model_idx][:, i] for model_idx in range(len(model_paths))],
                linestyle=[true_ls, *pred_lss],
                color=[true_color, *pred_colors],
                marker=[true_marker, *pred_markers],
                markersize=[true_ms, *pred_mss],
                linewidth=[true_lw, *pred_lws],
            )
            for model_idx in range(len(model_paths)):
                plt.fill_between(
                    p_list[:, eParam[vary_param]],
                    dJdp_mean[model_idx][:, i] - dJdp_std[model_idx][:, i],
                    dJdp_mean[model_idx][:, i] + dJdp_std[model_idx][:, i],
                    alpha=0.2,
                    antialiased=True,
                    color=pred_colors[model_idx],
                    zorder=2,
                )
            plt.xlabel(f"$\\{vary_param}$")
            plt.title(f"$\\{not_vary_param} = {p_list[0, eParam[not_vary_param]]}$")

            if j == 0:
                plt.ylabel(f"$dJ/d\\{i.name}$")
            # if j == 0 and k == 0:
            #     plt.legend(legend_str, loc="upper left",
            #                ncols=2,
            #                handlelength=1.5,
            #                handletextpad=0.5,
            #                columnspacing=0.1)

            # if j == 0 and k == 1:
            #     plt.legend(legend_str, loc="upper right")

            if plot_name == "varying_beta":
                ax.xaxis.set_major_locator(MultipleLocator(1))
            elif plot_name == "varying_tau":
                ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(visible=True)
            min_ylim = np.min(dJdp_true)
            max_ylim = np.max(dJdp_true)
            range_ylim = max_ylim - min_ylim
            ax.set_ylim([min_ylim - 0.15 * range_ylim, max_ylim + 0.15 * range_ylim])

legend = plt.figlegend(
    legend_str, loc="upper right", ncols=len(legend_str), bbox_to_anchor=(1.0, 1.2)
)
handles = legend.get_lines()
handles = [handles[idx] for idx in [0, 1, 3, 2]]
labels = [legend_str[idx] for idx in [0, 1, 3, 2]]
# reorder legend
plt.figlegend(
    handles, labels, loc="upper right", ncols=len(legend_str), bbox_to_anchor=(1.0, 1.2)
)
if save_fig:
    fig.savefig(f"paper/graphics/figure_{fig_name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"paper/graphics/figure_{fig_name}.pdf", bbox_inches="tight")
plt.show()
