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

save_fig = False
fig_name = "sensitivity"
# N_reservoir = 1200, connectivity = 20
model_path = Path("local_results/rijke/run_20231029_153121")

# same washout, eta_1_init = 1.0
save_paths_beta = [
    "20231114_162553",  # tau = 0.07
    "20231113_101654",  # tau = 0.12
    # "20231114_121526", # tau = 0.17
    # "20231113_125101", # tau = 0.2
    "20231114_174458",  # tau = 0.22
    # "20231113_112743", # tau = 0.25
    "20231113_130717",  # tau = 0.32
]
save_paths_tau = [
    "20231114_183542",  # beta = 1.25
    # "20231113_013027", # beta = 2.0
    "20231113_214746",  # beta = 2.5
    # "20231113_213932", # beta = 3.0
    "20231114_185152",  # beta = 3.75
    "20231113_040638",  # beta = 4.5
]

# not same washout, eta_1_init = 1.5
save_paths_beta = [
    "20231219_022703",  # tau = 0.07
    "20231218_210035",  # tau = 0.12
    "20231219_015234",  # tau = 0.22
    "20231219_031239",  # tau = 0.32
]
save_paths_tau = [
    "20231218_201438",  # beta = 1.25
    "20231218_215207",  # beta = 2.5
    "20231218_194028",  # beta = 3.75
    "20231218_193704",  # beta = 4.5
]

true_color = cmap(0)
pred_color = cmap(2)

true_lw = 5.0
pred_lw = 2.0
true_ls = "-"
pred_ls = "--"
true_marker = "none"
pred_marker = "none"
true_ms = 6
pred_ms = 8

titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

fig = plt.figure(figsize=(10, 10), constrained_layout=True)

# print model properties
config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]
if "top_idx" in results.keys():
    top_idx = results["top_idx"]
else:
    top_idx = 0
(
    ESN_dict,
    hyp_param_names,
    hyp_param_scales,
    hyp_params,
) = post.get_ESN_properties_from_results(config, results, dim=1, top_idx=top_idx)
print(ESN_dict)
[
    print(f"{hyp_param_name}: {hyp_param}")
    for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params)
]

for k, plot_name in enumerate(["varying_beta", "varying_tau"]):
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
        sens_results = pp.unpickle_file(
            model_path / f"sensitivity_results_{save_path}.pickle"
        )[0]
        dJdp_mean = {}
        dJdp_std = {}
        dJdp_mean = np.mean(sens_results["dJdp"]["esn"]["adjoint"], axis=0)
        dJdp_std = np.std(sens_results["dJdp"]["esn"]["adjoint"], axis=0)
        p_list = pp.make_param_mesh(
            [sens_results["beta_list"], sens_results["tau_list"]]
        )
        # for i in param_list:
        #     ax=plt.subplot(2,2,2*k+1+i)
        i = eParam[vary_param]
        ax = plt.subplot(len(save_paths), 2, k + 1 + 2 * j)
        ax.set_title(titles[k + 2 * j], loc="left")
        vis.plot_lines(
            p_list[:, eParam[vary_param]],
            sens_results["dJdp"]["true"]["adjoint"][:, i],
            dJdp_mean[:, i],
            linestyle=[true_ls, pred_ls],
            color=[true_color, pred_color],
            marker=[true_marker, pred_marker],
            markersize=[true_ms, pred_ms],
            linewidth=[true_lw, pred_lw],
        )

        plt.fill_between(
            p_list[:, eParam[vary_param]],
            dJdp_mean[:, i] - dJdp_std[:, i],
            dJdp_mean[:, i] + dJdp_std[:, i],
            alpha=0.2,
            antialiased=True,
            color=pred_color,
            zorder=2,
        )
        plt.xlabel(f"$\\{vary_param}$")
        plt.ylabel(f"$dJ/d\\{i.name}$")
        plt.title(f"$\\{not_vary_param} = {p_list[0, eParam[not_vary_param]]}$")
        # if j == 0 and k == 0:
        #     plt.legend(["True", "ESN"], loc="upper left")
        if j == 0 and k == 1:
            plt.legend(["True", "ESN"], loc="upper right")

        if plot_name == "varying_beta":
            ax.xaxis.set_major_locator(MultipleLocator(1))
        elif plot_name == "varying_tau":
            ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(visible=True)

if save_fig:
    fig.savefig(f"paper/graphics/figure_{fig_name}_v6.png", bbox_inches="tight")
plt.show()
