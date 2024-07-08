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
save_fig = True
use_cv = False
figure_name = "lorenz_convergence"
figure_size = (15, 4)

cmap = cm.create_custom_colormap(type="discrete")
true_color = cmap(0)
pred_color = cmap(2)
direct_color = cmap(1)
# true_color = "silver"
# pred_color = "tab:red"
# direct_color = "black"
true_lw = 3
pred_lw = 5
direct_lw = 3.5
true_ls = "-"
pred_ls = "--"
direct_ls = ":"
true_marker = "None"
pred_marker = "None"
direct_marker = "None"

model_path = Path("local_results/lorenz63/run_20240208_121804")
sens_results_list = ["20240208_185946", "20240208_200922", "20240214_113451"]
eParam = Lorenz63.get_eParamVar()
param_vars = [my_eParam.name for my_eParam in eParam]
param_vars = ["sigma", "rho", "beta"]

p_list = [[8 / 3, 28, 10]]
n_loops = 10000
param = "rho"
p_idx = 0
esn_idx = 0
legend_str = []
fig = plt.figure(figsize=figure_size, constrained_layout=True)
for loop_time_idx in range(len(sens_results_list)):
    sens_results = pp.unpickle_file(
        model_path / f"sensitivity_results_{sens_results_list[loop_time_idx]}.pickle"
    )[0]
    dJdp_true = sens_results["dJdp"]["true"]["adjoint"][p_idx, eParam[param], :, 0]
    cum_mean = 1 / np.arange(1, n_loops + 1) * np.cumsum(dJdp_true[:n_loops])

    dJdp_esn = sens_results["dJdp"]["esn"]["adjoint"][
        esn_idx, p_idx, eParam[param], :, 0
    ]
    cum_mean_pred = 1 / np.arange(1, n_loops + 1) * np.cumsum(dJdp_esn[:n_loops])

    plt.subplot(1, len(sens_results_list), loop_time_idx + 1)
    # plt.plot(np.arange(1,n_loops+1), cum_mean)
    # plt.plot(np.arange(1,n_loops+1), cum_mean_pred, '--')

    vis.plot_lines(
        np.arange(1, n_loops + 1),
        cum_mean,
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
        marker=[true_marker, pred_marker],
        xlabel="Trajectories",
    )

    if loop_time_idx == 0:
        plt.ylabel(f"$d\\bar{{z}}/d{param[0]}$")
    else:
        ax = plt.gca()
        ax.set_yticklabels([])
    plt.ylim((0.5, 1.5))

    dJdp_true_mean = np.mean(dJdp_true)
    dJdp_esn_mean = np.mean(dJdp_esn)
    print(f"True: {dJdp_true_mean:0.3f}")
    print(f"ESN: {dJdp_esn_mean:0.3f}")
    plt.grid(visible=True)

fig.savefig(f"local_images/figure_{figure_name}.png", bbox_inches="tight", dpi=300)
plt.show()
