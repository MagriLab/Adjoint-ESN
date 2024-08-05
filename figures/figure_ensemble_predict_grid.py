import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam

plt.style.use("src/stylesheet.mplstyle")
save_fig = True
figure_size = (15, 5)
# N_reservoir = 1200, connectivity = 20
model_paths = [
    Path("final_results/standard/run_20231031_152932"),  # standard with [eta,mu]
    Path("final_results/standard/run_20231031_152907"),  # standard with [eta,mu,v]
    Path("final_results/rijke/run_20231029_153121"),  # rijke with reservoir
]

save_paths = [
    "20231101_182139",
    "20231101_153934",
    "20231031_010509",
]

titles = ["(a)", "(b)", "(c)"]
fig = plt.figure(figsize=figure_size, constrained_layout=True)

for model_idx, model_path in enumerate(model_paths):
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

    save_path = save_paths[model_idx]
    error_results = pp.unpickle_file(model_path / f"error_results_{save_path}.pickle")[
        0
    ]
    # print(error_results["tikhonov"])
    error_mat = error_results["error"]  # params x folds x ensemble
    error_mat_mean = np.mean(error_mat, axis=2)  # take the mean over ensembles
    error_mat_mean = np.mean(error_mat_mean, axis=1)  # take mean over folds

    beta_list = error_results["beta_list"]
    tau_list = error_results["tau_list"]

    beta_step = 4
    beta_ticks = np.arange(len(beta_list))
    beta_ticks = beta_ticks[2::beta_step]
    beta_ticklabels = [f"{beta_name:.1f}" for beta_name in beta_list[2::beta_step]]

    tau_step = 5
    tau_ticks = np.arange(len(tau_list))
    tau_ticks = tau_ticks[::tau_step]
    tau_ticklabels = [f"{tau_name:.2f}" for tau_name in tau_list[::tau_step]]

    # reshape as a grid rows increasing beta, columns increasing tau
    error_mat_mean_grid = error_mat_mean.reshape(len(beta_list), len(tau_list))

    # plot on the logarithmic scale
    plot_mat = np.log10(100 * error_mat_mean_grid)
    plot_mat = plot_mat.T  # take transpose, now beta are columns, tau are rows

    ax = fig.add_subplot(1, 3, model_idx + 1)
    pos = ax.imshow(plot_mat, vmin=-1, vmax=2, origin="lower", cmap="OrRd")
    # extent=(beta_list[0], beta_list[-1], tau_list[0], tau_list[-1]),
    # aspect=25)

    # set tau labels
    ax.set_yticks(tau_ticks)
    ax.set_yticklabels(tau_ticklabels)
    ax.set_ylabel("$\\tau$")

    # set beta labels
    ax.set_xticks(beta_ticks)
    ax.set_xticklabels(beta_ticklabels)
    ax.set_xlabel("$\\beta$")

    # ax.set_title(titles[model_idx], loc="left")
    ax.annotate(titles[model_idx], xy=(-0.1, 0.95), xycoords="axes fraction")

    cbar = fig.colorbar(pos, ax=ax, shrink=0.85, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(["$0.1$", "$1$", "$10$", "$100$"])
    cbar.ax.set_title("$\epsilon \%$")
    # add the patch to indicate fixed points
    fixed_pt_patch = patches.StepPatch(
        values=[15.5, 7.5, 4.5, 2.5, 0.5],
        edges=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        baseline=-0.5,
        fill=True,
        edgecolor="darkgrey",
        facecolor="lightgrey",
        linewidth=2,
        hatch="\\\\\\",
    )
    fp = ax.add_patch(fixed_pt_patch)

    # Add rectangles to train and validation points
    training_parameters = results["training_parameters"]
    numcols = len(beta_list)
    numrows = len(tau_list)
    origin = np.array([-0.5, -0.5])
    beta_space = beta_list[1] - beta_list[0]
    tau_space = tau_list[1] - tau_list[0]
    for train_param in training_parameters:
        beta_loc = int(np.ceil((train_param[eParam.beta] - beta_list[0]) / beta_space))
        tau_loc = int(np.ceil((train_param[eParam.tau] - tau_list[0]) / tau_space))

        # Create a Rectangle patch
        rect = patches.Rectangle(
            origin + np.array([beta_loc, tau_loc]),
            1,
            1,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
        )

        # Add the patch to the Axes
        lp1 = ax.add_patch(rect)

    validation_parameters = results["validation_parameters"]
    for val_param in validation_parameters:
        beta_loc = int(np.ceil((val_param[eParam.beta] - beta_list[0]) / beta_space))
        tau_loc = int(np.ceil((val_param[eParam.tau] - tau_list[0]) / tau_space))

        # Create a Rectangle patch
        rect = patches.Rectangle(
            origin + np.array([beta_loc, tau_loc]),
            1,
            1,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            linestyle=(1.0, (2.5, 1.5)),
        )

        # Add the patch to the Axes
        lp2 = ax.add_patch(rect)
    ax.legend(
        [lp1, lp2, fp],
        ["Train", "Validation", "Fixed pts."],
        ncols=3,
        columnspacing=0.5,
        bbox_to_anchor=[0.5, 1.08],
        loc="center",
        handlelength=0.75,
        handletextpad=0.35,
    )

if save_fig:
    fig.savefig("paper/graphics/figure_ensemble_predict.png")
    fig.savefig("paper/graphics/figure_ensemble_predict.pdf")
plt.show()
