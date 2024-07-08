import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import LinearLocator, MultipleLocator

import adjoint_esn.utils.postprocessing as post
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)
save_fig = False
same_washout = False
model_path = Path("local_results/lorenz63/run_20240208_121804")

fig_name = "2"

titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# run_20240204_173928
# sens_results_list = [["20240205_005708"],
#                      ["20240205_022500"],
#                      ["20240205_031448"],
#                      ]
# sens_results_list = [["20240205_130051"],
#                      ["20240205_141510"],
#                      ["20240205_152601"],
#                      ]
# sens_results_list = [["20240206_085311"],
#                      ["20240206_085311"]]

# run_20240105_140530
# sens_results_list = [["20240122_072115","20240122_050248"],#,
#                      ["20240122_061243","20240122_041929"]] #,

# # different initial conditions
# 1.0 LT
# sens_results_list = [["20240124_103013"],
#                      ["20240124_110838"],
#                      ["20240124_111032"],
#                      ["20240124_113630"]]

# O.5 LT
# sens_results_list = [["20240124_172710"],
#                      ["20240124_172732"],
#                      ["20240124_174010"],
#                      ["20240124_180734"]]

# 0.8 LT
# sens_results_list = [["20240125_221545"],
#                      ["20240125_235107"],
#                      ["20240125_235544"],
#                      ["20240126_022957"]]

# sens_results_list = [["20240124_180734"],
#                      ["20240125_235107"],
#                      ["20240124_113630"],
#                      ]
# sens_results_name = "20240201_164536" # beta
# sens_results_name = "20240201_033758" # rho
# sens_results_name = "20240201_193724" # rho
# sens_results_name = "20240201_144530" # sigma

# changed parameter mean
# sens_results_list = [["20240207_041304"],
#                      ["20240207_022338"],
#                      ["20240207_044032"],
#                      ]
# sens_results_name = "20240207_041304" # rho

# run_20240206_181642
# sens_results_name = "20240208_015144" #beta
# sens_results_name = "20240208_000937" #rho
# sens_results_name = "20240208_001827" #sigma

# sens_results_list = [["20240207_222433"],
#                      ["20240207_225205"],
#                      ["20240207_234943"],
#                      ]

# run_20240208_121804
# sens_results_name = "20240208_201311" #beta
# sens_results_name = "20240208_193744" #rho
# sens_results_name = "20240208_201509"  # sigma
sens_results_list = [
    ["20240208_185946"],
    ["20240208_191537"],
    ["20240208_200922"],
]
direct_estimate = [-1.68, 1.01, 0.16]

# sens_results_name = "20240212_050609" #beta
# sens_results_name = "20240212_053848" #rho
# sens_results_name = "20240212_071830" # sigma

# sens_results_name = "20240214_090024" #beta
# sens_results_name = "20240213_224526" #rho
# sens_results_name = "20240214_054527" # sigma
# sens_param_name = "beta"
# sens_results_name = "20240218_152612"
esn_idx = 1
figure_size = (15, 3)
# true_color = ["dimgrey","silver"]
# pred_color = ["darkred","tab:red"]

true_color = ["silver"]
pred_color = ["tab:red"]

true_lw = [5, 2]
pred_lw = [3, 2]
true_ls = ["-", "-."]
pred_ls = ["--", "-."]

# ylims = [[[-3,0],[0.5,1.5],[0.0,0.5]],
#          [[-3,0],[0.25,1.25],[-0.2,0.2]]]

ylims = [
    [[-2.0, -0.5], [0.5, 1.5], [0.0, 0.2]],
    [[-2.0, -0.5], [0.5, 1.5], [0.0, 0.2]],
    [[-2.0, -1.0], [0.82, 0.9], [0.0, 0.2]],
    [[-2.0, -1.0], [0.82, 0.9], [0.0, 0.2]],
    [[-2.0, -1.0], [0.82, 0.9], [0.0, 0.2]],
]


# config = post.load_config(model_path)
# results = pp.unpickle_file(model_path / "results.pickle")[0]


param_vars = ["beta", "rho", "sigma"]
eParam = Lorenz63.get_eParamVar()

fig, axs = plt.subplots(
    len(sens_results_list),
    len(param_vars),
    figsize=figure_size,
    constrained_layout=True,
)
for p_idx in range(len(sens_results_list)):
    for loop_time_idx in range(len(sens_results_list[0])):
        sens_results = pp.unpickle_file(
            model_path
            / f"sensitivity_results_{sens_results_list[p_idx][loop_time_idx]}.pickle"
        )[0]
        print("Initial condition:", sens_results["y_init"])
        print("Loop times:", sens_results["loop_times"])
        for param_var_idx, param in enumerate(param_vars):
            dJdp_true = sens_results["dJdp"]["true"]["adjoint"][0, eParam[param], :, 0]
            n_loops = len(dJdp_true)
            cum_mean = 1 / np.arange(1, n_loops + 1) * np.cumsum(dJdp_true)

            dJdp_esn = sens_results["dJdp"]["esn"]["adjoint"][0, 0, eParam[param], :, 0]
            cum_mean_pred = 1 / np.arange(1, n_loops + 1) * np.cumsum(dJdp_esn)

            ax = axs[p_idx, param_var_idx]
            plt.sca(ax)
            vis.plot_lines(
                np.arange(1, n_loops + 1),
                cum_mean,
                cum_mean_pred,
                linestyle=[true_ls[loop_time_idx], pred_ls[loop_time_idx]],
                linewidth=[true_lw[loop_time_idx], pred_lw[loop_time_idx]],
                color=[true_color[loop_time_idx], pred_color[loop_time_idx]],
            )
            # ax.xaxis.set_major_locator(MultipleLocator(2000))
            ax.yaxis.set_major_formatter("{x:.2f}")
            ax.yaxis.set_major_locator(LinearLocator(4))
            ax.set_ylim(ylims[p_idx][param_var_idx])
            if p_idx == 0:
                ax.set_title(f"$d\\bar{{z}}/d\\{param}$")
            # if p_idx < len(sens_results_list)-1:
            #     ax.set_xticklabels([])
            if p_idx == len(sens_results_list) - 1:
                ax.set_xlabel("$N_{ensemble}$")
            if param_var_idx == 0:
                ax.set_ylabel(f"$\\tau = {sens_results['loop_times'][0]}$")
            # plt.grid(visible=True)
            # ax.set_xlim([-100,1000])
            ax.hlines(
                direct_estimate[param_var_idx], 0, 100000, color="black", linestyle="-."
            )

# sens_results = pp.unpickle_file(
#             model_path / f"sensitivity_results_{sens_results_name}.pickle"
#         )[0]
# print("Initial condition:", sens_results["y_init"])
# print("Loop times:", sens_results["loop_times"])
# n_ensemble = len(sens_results["dJdp"]["esn"]["adjoint"])
# fig, axs = plt.subplots(n_ensemble,len(param_vars),figsize=figure_size, constrained_layout=True)
# for param_var_idx, param in enumerate(param_vars):
#     dJdp_true = sens_results["dJdp"]["true"]["adjoint"][0,eParam[param],:,0] #p_idx, param_var_idx, loop_idx, loop_time_idx
#     n_loops = len(dJdp_true)
#     cum_mean = 1/np.arange(1,n_loops+1) * np.cumsum(dJdp_true)
#     for e_idx in range(n_ensemble):
#         dJdp_esn = sens_results["dJdp"]["esn"]["adjoint"][e_idx,0,eParam[param],:,0] #e_idx, p_idx, param_var_idx, loop_idx, loop_time_idx
#         cum_mean_pred = 1/np.arange(1,n_loops+1) * np.cumsum(dJdp_esn)

#         ax = axs[e_idx,param_var_idx]
#         plt.sca(ax)
#         vis.plot_lines(
#         np.arange(1,n_loops+1),
#         cum_mean,
#         cum_mean_pred,
#         linestyle=[true_ls[0], pred_ls[0]],
#         linewidth=[true_lw[0], pred_lw[0]],
#         color=[true_color[0], pred_color[0]],
#         )
#         #ax.xaxis.set_major_locator(MultipleLocator(2000))
#         ax.yaxis.set_major_formatter('{x:.2f}')
#         ax.set_ylim(ylims[e_idx][param_var_idx])
#         if e_idx == 0:
#             ax.set_title(f"$d\\bar{{z}}/d\\{param}$")
#         if e_idx < n_ensemble-1:
#             ax.set_xticklabels([])
#         if e_idx == n_ensemble-1:
#             ax.set_xlabel("$N_{ensemble}$")
#         # plt.grid(visible=True)

# sens_results = pp.unpickle_file(
#     model_path / f"sensitivity_results_{sens_results_name}.pickle"
# )[0]
# print("Initial condition:", sens_results["y_init"])
# print("Loop times:", sens_results["loop_times"])
# dJdp_true_mean = np.mean(sens_results["dJdp"]["true"]["adjoint"], axis=2)
# dJdp_esn_mean = np.mean(sens_results["dJdp"]["esn"]["adjoint"][0], axis=2)
# fig, axs = plt.subplots(
#     1, len(param_vars), figsize=figure_size, constrained_layout=True
# )
# for i, param in enumerate(param_vars):
#     plt.subplot(1, len(param_vars), i + 1)
#     vis.plot_lines(
#         sens_results[f"{sens_param_name}_list"],
#         dJdp_true_mean[:, eParam[param], 0],
#         # dJdp_esn_mean[:, eParam[param], 0],
#         linestyle=[true_ls[0], pred_ls[0]],
#         linewidth=[true_lw[0], pred_lw[0]],
#         color=[true_color[0], pred_color[0]],
#         xlabel=f"$\\{sens_param_name}$",  # "$\\beta$", #"$\\sigma$", #"$\\rho$",
#         ylabel=f"$d\\bar{{z}}/d\\{param}$",
#     )

# legend_str=["$\\tau = 1.5 \; LT$","$\\tau = 1.5 \; LT$, ESN","$\\tau = 1.0 \; LT$","$\\tau = 1.0 \; LT$, ESN"]
legend_str = ["True Adjoint", "ESN Adjoint"]
plt.figlegend(
    legend_str, loc="upper center", ncol=len(legend_str), bbox_to_anchor=(0.5, 1.2)
)
if save_fig:
    fig.savefig(
        f"paper_chaotic/graphics/figure_ESN_4_{sens_param_name}_fine_2.png",
        bbox_inches="tight",
    )
    # fig.savefig(f"paper_chaotic/graphics/figure_{fig_name}_sigma.pdf", bbox_inches="tight")
plt.show()
