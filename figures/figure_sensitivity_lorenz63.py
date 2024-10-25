import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import MultipleLocator

import adjoint_esn.utils.visualizations as vis
from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63

# figure options
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 24})
rc("text", usetex=True)
# plt.style.use("dark_background")
save_fig = False
use_cv = False
figure_name = "2"
figure_size = (15, 4)

cmap = cm.create_custom_colormap(type="discrete")
true_color = cmap(0)
pred_color = cmap(2)
direct_color = cmap(1)
# true_color = "silver"
# pred_color = "tab:red"
# direct_color = "black"
true_lw = 10
pred_lw = 6
direct_lw = 3.5
true_ls = "-"
pred_ls = "--"
direct_ls = ":"
true_marker = "None"
pred_marker = "None"
direct_marker = "None"

titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

# results
model_path = Path("local_results/lorenz63/run_20240208_121804")
eParam = Lorenz63.get_eParamVar()
param_vars = [my_eParam.name for my_eParam in eParam]
param_vars = ["sigma", "rho", "beta"]

sens_results_list = [None] * 3
xlims = [None] * 3

xticks = [None] * 3
xticks[eParam.beta] = 0.2
xticks[eParam.rho] = 5.0
xticks[eParam.sigma] = 2.0

ylims_obj = [None] * 3
ylims_sens = [None] * 3
degree_list = [None] * 3
my_param = [None] * 3

if figure_name == "2":
    # beta = 8/3, rho = 28, sigma = 10
    sens_results_list[eParam.beta] = "20240217_221457"  # beta
    sens_results_list[eParam.rho] = "20240213_224526"  # rho
    sens_results_list[eParam.sigma] = "20240214_054527"  # sigma

    xlims[eParam.beta] = (2.0, 3.0)
    xlims[eParam.rho] = (25.0, 55.0)
    xlims[eParam.sigma] = (6.0, 18.0)

    ylims_obj[eParam.beta] = (22.5, 25)
    ylims_obj[eParam.rho] = (20.0, 55.0)
    ylims_obj[eParam.sigma] = (21.5, 24.5)

    ylims_sens[eParam.beta] = (-2.5, -0.5)
    ylims_sens[eParam.rho] = (0.7, 1.1)
    ylims_sens[eParam.sigma] = (-0.1, 0.8)

    degree_list[eParam.beta] = 3
    degree_list[eParam.rho] = 2
    degree_list[eParam.sigma] = 4

    my_param[eParam.beta] = 8 / 3
    my_param[eParam.rho] = 28.0
    my_param[eParam.sigma] = 10.0
elif figure_name == "3":
    # beta = 2, rho = 52, sigma = 13
    sens_results_list[eParam.beta] = "20240219_122702"  # beta
    sens_results_list[eParam.rho] = "20240219_213318"  # rho
    sens_results_list[eParam.sigma] = "20240220_112449"  # sigma

    xlims[eParam.beta] = (2.0, 3.0)
    xlims[eParam.rho] = (25.0, 55.0)
    xlims[eParam.sigma] = (6.0, 15.0)

    ylims_obj[eParam.beta] = (47.0, 49.0)
    ylims_obj[eParam.rho] = (20.0, 55.0)
    ylims_obj[eParam.sigma] = (45.5, 49.5)

    ylims_sens[eParam.beta] = (-3.0, 0.0)
    ylims_sens[eParam.rho] = (0.7, 1.1)
    ylims_sens[eParam.sigma] = (-0.2, 1.0)

    degree_list[eParam.beta] = 3
    degree_list[eParam.rho] = 3
    degree_list[eParam.sigma] = 4

    my_param[eParam.beta] = 2.0
    my_param[eParam.rho] = 52.0
    my_param[eParam.sigma] = 13.0

# direct_results_list = [None]*3
# 5000 LT
# direct_results_list[eParam.beta] = "20240214_014534" #beta
# direct_results_list[eParam.rho] = "20240214_003346" #rho
# direct_results_list[eParam.sigma] = "20240214_012947" #sigma

# 10000 LT
# direct_results_list[eParam.beta] = "20240214_030048" #beta
# direct_results_list[eParam.rho] = "20240214_022116" #rho
# direct_results_list[eParam.sigma] = "20240214_030036" #sigma


fig, axs = plt.subplots(
    2, len(param_vars), figsize=figure_size, constrained_layout=True
)

for i, param in enumerate(param_vars):
    sens_results = pp.unpickle_file(
        model_path / f"sensitivity_results_{sens_results_list[eParam[param]]}.pickle"
    )[0]
    J_true_mean = np.mean(sens_results["J"]["true"], axis=1)
    J_esn_mean = np.mean(sens_results["J"]["esn"][0], axis=1)
    dJdp_true_mean = np.mean(sens_results["dJdp"]["true"]["adjoint"], axis=2)
    dJdp_esn_mean = np.mean(sens_results["dJdp"]["esn"]["adjoint"][0], axis=2)

    # direct_results =  pp.unpickle_file(
    # f"local_results/lorenz63/direct_results_{direct_results_list[i]}.pickle"
    # )[0]

    # setup random seed
    # np.random.seed(10)
    # random train-val split
    # train_idx = np.random.choice(len(sens_results[f"{param}_list"]),size=15,replace=False)
    # val_idx = np.setdiff1d(np.arange(len(sens_results[f"{param}_list"])),train_idx)

    # 1 out cross-validation
    val_error = np.zeros((len(sens_results[f"{param}_list"]), 10))
    for my_degree in np.arange(1, 11):
        for val_i in range(len(sens_results[f"{param}_list"])):
            val_idx = [val_i]
            train_idx = np.setdiff1d(
                np.arange(len(sens_results[f"{param}_list"])), val_idx
            )
            coeffs1 = np.polyfit(
                sens_results[f"{param}_list"][train_idx],
                J_true_mean[train_idx],
                deg=my_degree,
            )
            J_approx_val = np.polyval(coeffs1, sens_results[f"{param}_list"][val_idx])
            val_error[val_i, my_degree - 1] = np.mean(
                (J_true_mean[val_idx, 0] - J_approx_val) ** 2
            )

        print(
            f"Degree {my_degree}, validation error: {np.mean(val_error[:,my_degree-1]):.5f}"
        )

    my_best_degree = np.argmin(np.mean(val_error, axis=0)) + 1
    print("Best degree:", my_best_degree)
    if use_cv:
        use_degree = my_best_degree
    else:
        use_degree = degree_list[eParam[param]]
    coeffs = np.polyfit(sens_results[f"{param}_list"], J_true_mean, deg=use_degree)

    J_approx = np.polyval(coeffs, sens_results[f"{param}_list"])
    dcoeffs = np.arange(len(coeffs) - 1, 0, -1)[None, :].T
    dJdp_approx = np.polyval(dcoeffs * coeffs[:-1], sens_results[f"{param}_list"])
    my_dJdp_approx = np.polyval(dcoeffs * coeffs[:-1], my_param[eParam[param]])
    print(f"dJd{param} approx at {my_param} = {my_dJdp_approx}")

    # plt.subplot(2, len(param_vars), i + 1)
    # vis.plot_lines(
    #     sens_results[f"{param}_list"],
    #     J_true_mean,
    #     J_esn_mean,
    #     # J_approx,
    #     linestyle=[true_ls, pred_ls, direct_ls],
    #     linewidth=[true_lw, pred_lw, direct_lw],
    #     color=[true_color, pred_color, direct_color],
    #     marker=[true_marker, pred_marker, direct_marker],
    #     # xlabel=f"$\\{param}$",
    #     ylabel=f"$\\bar{{z}}$",
    # )
    # plt.xlim(xlims[eParam[param]])
    # plt.ylim(ylims_obj[eParam[param]])
    # ax = plt.gca()
    # ax.set_xticklabels([])
    # ax.xaxis.set_major_locator(MultipleLocator(xticks[eParam[param]]))
    # ax.annotate(titles[i], xy=(0.85, 0.85), xycoords="axes fraction")

    if i == 1:
        # legend_str = ["True", "ESN", "Polyfit"]
        legend_str = ["True", "ESN"]
        # plt.legend(legend_str, loc="lower right")
        plt.legend(legend_str, loc="upper left")

    plt.subplot(1, len(param_vars), i + 1)
    vis.plot_lines(
        sens_results[f"{param}_list"],
        dJdp_true_mean[:, eParam[param], 0],
        dJdp_esn_mean[:, eParam[param], 0],
        # direct_results["dJdp"],
        # dJdp_approx,
        linestyle=[true_ls, pred_ls, direct_ls],
        linewidth=[true_lw, pred_lw, direct_lw],
        color=[true_color, pred_color, direct_color],
        marker=[true_marker, pred_marker, direct_marker],
        xlabel=f"${param[0]}$",
        ylabel=f"$d\\bar{{z}}/d{param[0]}$",
    )
    plt.xlim(xlims[eParam[param]])
    plt.ylim(ylims_sens[eParam[param]])
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(xticks[eParam[param]]))
    # ax.annotate(titles[3 + i], xy=(0.85, 0.85), xycoords="axes fraction")

    # if i == 2:
    #     legend_str = ["True Adjoint", "ESN Adjoint", "Approx."]
    #     plt.legend(legend_str, loc='upper right')
if save_fig:
    fig.savefig(
        f"paper/graphics_ppt/figure_{figure_name}.png", bbox_inches="tight", dpi=300
    )
    # fig.savefig(f"paper_chaotic/graphics/figure_{figure_name}.pdf", bbox_inches="tight")
plt.show()
