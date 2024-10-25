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

save_fig = False
figsize = (10, 6)
fig_name = 2

model_paths = [
    # Path("local_results/rijke/run_20231029_153121"),
    Path("local_results/rijke/run_20231029_153121"),
    Path("local_results/rijke/run_20240307_175258"),
    Path("local_results/rijke/run_20240307_175258"),
    # Path("local_results/rijke/run_20240307_175258"),
]
res_names = [
    # "20240805_153640",  # limit cycle beta = 2.0, tau = 0.25
    "20240805_162550",  # limit cycle beta = 4.5, tau = 0.12
    "20240805_145520",  # quasiperiodic beta = 6.1, tau = 0.2
    "20240805_150202",  # chaotic beta = 7.6, tau = 0.22
    # "20240722_204116",  # chaotic beta = 8.7, tau = 0.23
]
LTs = [1.0, 1.0, 8.5, 3.9]

config = post.load_config(model_paths[0])
N_g = config.simulation.N_g

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

ax_ = fig.subplots(2, len(model_paths))

for j, (model_path, res_name, LT) in enumerate(zip(model_paths, res_names, LTs)):
    res_path = model_path / f"init_sensitivity_results_{res_name}.pickle"
    res = pp.unpickle_file(res_path)[0]

    if len(model_paths) > 1:
        ax = ax_[:, j]
    else:
        ax = ax_

    mean_rel_err_u0 = np.zeros(len(res["loop_times"]))
    std_rel_err_u0 = np.zeros(len(res["loop_times"]))
    mean_rel_err_u0_x0 = np.zeros(len(res["loop_times"]))
    std_rel_err_u0_x0 = np.zeros(len(res["loop_times"]))
    mean_angles_u0 = np.zeros(len(res["loop_times"]))
    std_angles_u0 = np.zeros(len(res["loop_times"]))
    mean_angles_u0_x0 = np.zeros(len(res["loop_times"]))
    std_angles_u0_x0 = np.zeros(len(res["loop_times"]))

    for i, int_time in enumerate(res["loop_times"]):
        dJdu0_true = res["dJdu0"]["true"]["adjoint"][0, : 2 * N_g, :, i]
        dJdu0_esn = res["dJdu0"]["esn"]["adjoint"][:, 0, : 2 * N_g, :, i]
        dJdu0_esn_x0 = res["dJdu0"]["esn_from_x0"]["adjoint"][:, 0, : 2 * N_g, :, i]
        beta = res["beta_list"][0]
        tau = res["tau_list"][0]

        norm_true = np.linalg.norm(dJdu0_true, axis=0)  # norm over state variables
        norm_pred_u0 = np.linalg.norm(dJdu0_esn, axis=1)  # norm over state variables
        norm_pred_u0_x0 = np.linalg.norm(
            dJdu0_esn_x0, axis=1
        )  # norm over state variables

        # relative error
        diff_u0 = dJdu0_true - dJdu0_esn
        norm_diff_u0 = np.linalg.norm(diff_u0, axis=1)
        norm_rel_u0 = norm_diff_u0 / norm_true
        mean_rel_err_u0[i] = np.mean(norm_rel_u0)
        std_rel_err_u0[i] = np.std(norm_rel_u0)

        diff_u0_x0 = dJdu0_true - dJdu0_esn_x0
        norm_diff_u0_x0 = np.linalg.norm(diff_u0_x0, axis=1)
        norm_rel_u0_x0 = norm_diff_u0_x0 / norm_true
        mean_rel_err_u0_x0[i] = np.mean(norm_rel_u0_x0)
        std_rel_err_u0_x0[i] = np.std(norm_rel_u0_x0)

        # gradient direction
        dot_prod_u0 = np.einsum(
            "kij,ij->kj", dJdu0_esn, dJdu0_true
        )  # k: esn_idx, i: state_var_idx, j: loop_idx
        denom_u0 = np.einsum("kj,j->kj", norm_pred_u0, norm_true)
        cos_angles_u0 = dot_prod_u0 / denom_u0
        angles_u0 = np.degrees(np.arccos(cos_angles_u0))
        mean_angles_u0[i] = np.mean(angles_u0)
        std_angles_u0[i] = np.std(angles_u0)

        dot_prod_u0_x0 = np.einsum(
            "kij,ij->kj", dJdu0_esn_x0, dJdu0_true
        )  # k: esn_idx, i: state_var_idx, j: loop_idx
        denom_u0_x0 = np.einsum("kj,j->kj", norm_pred_u0_x0, norm_true)
        cos_angles_u0_x0 = dot_prod_u0_x0 / denom_u0_x0
        angles_u0_x0 = np.degrees(np.arccos(cos_angles_u0_x0))
        mean_angles_u0_x0[i] = np.mean(angles_u0_x0)
        std_angles_u0_x0[i] = np.std(angles_u0_x0)

        print(f"beta={beta:0.2f}, tau={tau:0.2f}")
        print(f"Integration time:{int_time}")
        print(f"Error dJ/du0: {mean_rel_err_u0[i]:0.3f} +/- {std_rel_err_u0[i]:0.3f}")
        print(
            f"Error dJ/du0 from x0: {mean_rel_err_u0_x0[i]:0.3f} +/- {std_rel_err_u0_x0[i]:0.3f}"
        )
        print(f"Angle dJ/du0: {mean_angles_u0[i]:0.3f} +/- {std_angles_u0[i]:0.3f}")
        print(
            f"Angle dJ/du0 from x0: {mean_angles_u0_x0[i]:0.3f} +/- {std_angles_u0_x0[i]:0.3f} \n"
        )

    ax[0].set_title(f"$\\beta = {beta:.2f}, \; \\tau = {tau:.2f}$")
    ax[0].errorbar(
        res["loop_times"] / LT,
        100 * mean_rel_err_u0,
        100 * std_rel_err_u0,
        color=pred_color,
        fmt="-o",
    )
    ax[0].errorbar(
        res["loop_times"] / LT,
        100 * mean_rel_err_u0_x0,
        100 * std_rel_err_u0_x0,
        color=pred_x0_color,
        fmt="--o",
    )
    ax[0].set_xticks(res["loop_times"] / LT)
    ax[0].set_xticklabels([])

    ax[1].errorbar(
        res["loop_times"] / LT,
        mean_angles_u0,
        std_angles_u0,
        color=pred_color,
        fmt="-o",
    )
    ax[1].errorbar(
        res["loop_times"] / LT,
        mean_angles_u0_x0,
        std_angles_u0_x0,
        color=pred_x0_color,
        fmt="--o",
    )
    if LT == 1.0:
        ax[1].set_xlabel("Integration time")
    else:
        ax[1].set_xlabel("Integration time [LT]")
    ax[1].set_xticks(res["loop_times"] / LT)
    if j == 0:
        ax[0].set_ylabel("Rel. error $\%$")
        ax[1].set_ylabel("Angle")

plt.figlegend(
    ["ESN $dJ/d\mathbf{y}_{in}(0)$", "ESN $(dJ/d\mathbf{x}(0))\mathbf{W}_{out}^{-1}$"],
    loc="upper center",
    ncols=2,
    bbox_to_anchor=(0.5, 1.1),
)
if save_fig:
    fig.savefig(
        f"paper/graphics/figure_init_sensitivity_error_{fig_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
plt.show()
