import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam


def relative_error(Y_true, Y_pred):
    err_mat = np.zeros_like(Y_true)
    for i in range(Y_true.shape[0]):
        for j in range(Y_true.shape[1]):
            diff = np.abs(Y_true[i, j] - Y_pred[i, j])
            if np.abs(Y_true[i, j]) < 1e-3 and diff < 1e-3:
                err_mat[i, j] = 0
            else:
                err_mat[i, j] = diff / np.abs(Y_true[i, j])
    return err_mat


rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 12})
rc("text", usetex=True)
save_fig = True

# N_reservoir = 1200, connectivity = 20
model_path = Path("local_results/rijke/run_20231029_153121")

save_path = "20231111_093233"

titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

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

sens_results = pp.unpickle_file(model_path / f"sensitivity_results_{save_path}.pickle")[
    0
]
sens_mat_pred = sens_results["dJdp"]["esn"]["adjoint"]
sens_mat_mean = np.mean(sens_mat_pred, axis=0)  # ensembles x params x 2
sens_mat_true = sens_results["dJdp"]["true"]["adjoint"]

beta_list = sens_results["beta_list"]
tau_list = sens_results["tau_list"]

beta_step = 4
beta_ticks = np.arange(len(beta_list))
beta_ticks = beta_ticks[2::beta_step]
beta_ticklabels = [f"{beta_name:.1f}" for beta_name in beta_list[2::beta_step]]

tau_step = 5
tau_ticks = np.arange(len(tau_list))
tau_ticks = tau_ticks[::tau_step]
tau_ticklabels = [f"{tau_name:.2f}" for tau_name in tau_list[::tau_step]]

# reshape as a grid rows increasing beta, columns increasing tau
dJdbeta_mean_grid = sens_mat_mean[:, eParam.beta].reshape(len(beta_list), len(tau_list))
dJdbeta_mean_grid = dJdbeta_mean_grid.T

dJdbeta_true_grid = sens_mat_true[:, eParam.beta].reshape(len(beta_list), len(tau_list))
dJdbeta_true_grid = dJdbeta_true_grid.T

dJdtau_mean_grid = sens_mat_mean[:, eParam.tau].reshape(len(beta_list), len(tau_list))
dJdtau_mean_grid = dJdtau_mean_grid.T

dJdtau_true_grid = sens_mat_true[:, eParam.tau].reshape(len(beta_list), len(tau_list))
dJdtau_true_grid = dJdtau_true_grid.T

ax1 = fig.add_subplot(2, 2, 1)
pos1 = ax1.imshow(dJdbeta_true_grid, origin="lower", vmin=-5, vmax=30)
cbar = fig.colorbar(pos1, ax=ax1, shrink=0.85)

ax2 = fig.add_subplot(2, 2, 2)
pos2 = ax2.imshow(dJdbeta_mean_grid, origin="lower", vmin=-5, vmax=30)
cbar = fig.colorbar(pos2, ax=ax2, shrink=0.85)

# err_beta = relative_error(dJdbeta_true_grid,dJdbeta_mean_grid)
# ax3 = fig.add_subplot(2, 3, 3)
# pos3 = ax3.imshow(err_beta,
#                   origin="lower")
# cbar = fig.colorbar(pos3, ax=ax3, shrink = 0.85)

ax4 = fig.add_subplot(2, 2, 3)
pos4 = ax4.imshow(dJdtau_true_grid, origin="lower", vmin=-750, vmax=500)
cbar = fig.colorbar(pos4, ax=ax4, shrink=0.85)

ax5 = fig.add_subplot(2, 2, 4)
pos5 = ax5.imshow(dJdtau_mean_grid, origin="lower", vmin=-750, vmax=500)
cbar = fig.colorbar(pos5, ax=ax5, shrink=0.85)

# err_tau = relative_error(dJdtau_true_grid,dJdtau_mean_grid)
# ax6 = fig.add_subplot(2, 3, 6)
# pos6 = ax6.imshow(err_tau,
#                 origin="lower")
# cbar = fig.colorbar(pos6, ax=ax6, shrink = 0.85)

axs = [ax1, ax2, ax4, ax5]

for ax_idx, ax in enumerate(axs):
    # set tau labels
    ax.set_yticks(tau_ticks)
    ax.set_yticklabels(tau_ticklabels)
    ax.set_ylabel("$\\tau$")

    # set beta labels
    ax.set_xticks(beta_ticks)
    ax.set_xticklabels(beta_ticklabels)
    ax.set_xlabel("$\\beta$")

    ax.set_title(titles[ax_idx], loc="left")

if save_fig:
    fig.savefig("paper/graphics/figure_sensitivity_grid.png", bbox_inches="tight")
plt.show()
