import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import visualizations as vis
from adjoint_esn.utils.enums import eParam

plt.style.use("src/stylesheet.mplstyle")
# plt.style.use("dark_background")

save_fig = True
figsize = (15, 5)

model_path = Path("final_results/rijke/run_20240307_175258")
res_cont_names = ["20240402_170758"]  # integration time 2 LT
res_ens_names = ["20240403_061239", "20240403_100025"]  # 2000 loops

cmap = cm.create_custom_colormap(type="continuous")
cmap2 = cm.create_custom_colormap(
    type="continuous", colors=[cmap(0.0), "white"], N=len(res_cont_names)
)
cmap3 = cm.create_custom_colormap(
    type="continuous", colors=[cmap(1.0), "white"], N=len(res_cont_names)
)
true_color_list = [cmap2(i) for i in range(len(res_cont_names))]  # teal
pred_color_list = [cmap3(i) for i in range(len(res_cont_names))]  # dark purple
true_color = cmap(0.0)  # teal
pred_color = cmap(1.0)  # dark purple

true_mean_color = "silver"
esn_mean_color = "grey"
true_lw = 5.0
pred_lw = 2.5

true_ls = "-"
pred_ls = "--"

LT = 8.5

titles = [f"({chr(i)})" for i in range(ord("a"), ord("z") + 1)]
tt = 0

fig = plt.figure(figsize=figsize, constrained_layout=True)
subfigs = fig.subfigures(nrows=1, ncols=2)

ax = subfigs[0].subplots(2, 1)
for res_idx, res_cont_name in enumerate(res_cont_names):
    cont_path = model_path / f"sensitivity_results_{res_cont_name}.pickle"
    res_cont = pp.unpickle_file(cont_path)[0]

    dJdp_true = res_cont["dJdp"]["true"]["adjoint"]
    dJdp_esn = res_cont["dJdp"]["esn"]["adjoint"]
    for i, param in enumerate(["beta", "tau"]):
        ax[i].plot(
            res_cont["loop_times"] / LT,
            dJdp_true[0, eParam[param], 0, :],
            color=true_color_list[res_idx],
            linestyle=true_ls,
            linewidth=true_lw,
        )

        dJdp_esn_mean = np.mean(dJdp_esn[:, 0, eParam[param], 0, :], axis=0)
        dJdp_esn_std = np.std(dJdp_esn[:, 0, eParam[param], 0, :], axis=0)
        ax[i].plot(
            res_cont["loop_times"] / LT,
            dJdp_esn_mean,
            color=pred_color_list[res_idx],
            linestyle=pred_ls,
            linewidth=pred_lw,
        )
        ax[i].fill_between(
            res_cont["loop_times"] / LT,
            dJdp_esn_mean + dJdp_esn_std,
            dJdp_esn_mean - dJdp_esn_std,
            alpha=0.2,
            color=pred_color_list[res_idx],
            antialiased=True,
            zorder=2,
        )
        ax[i].set_ylabel(f"$dJ/d\\{param}$")
        tt += 1

ax[0].annotate(titles[0], xy=(0.03, 0.7), xycoords="axes fraction")
ax[1].annotate(titles[1], xy=(0.03, 0.85), xycoords="axes fraction")
ax[0].set_xticklabels([])
ax[0].grid()
ax[1].grid()
ax[1].set_xlabel("Integration time [LT]")
# ax[0].legend(["True", "ESN"])

ax = subfigs[1].subplots(2, len(res_ens_names))
for res_idx, res_ens_name in enumerate(res_ens_names):
    ens_path = model_path / f"sensitivity_results_{res_ens_name}.pickle"
    res_ens = pp.unpickle_file(ens_path)[0]

    dJdp_true = res_ens["dJdp"]["true"]["adjoint"]
    dJdp_esn = res_ens["dJdp"]["esn"]["adjoint"]
    n_esns = dJdp_esn.shape[0]
    for i, param in enumerate(["beta", "tau"]):
        dJdp_true_mean = np.mean(dJdp_true[0, eParam[param], :, 0])
        dJdp_true_std = np.std(dJdp_true[0, eParam[param], :, 0])
        dJdp_esn_mean = np.mean(dJdp_esn[:, 0, eParam[param], :, 0])

        plt.sca(ax[i, res_idx])
        vis.plot_statistics_ensemble(
            *[dJdp_esn[e_idx, 0, eParam[param], :, 0] for e_idx in range(n_esns)],
            y_base=dJdp_true[0, eParam[param], :, 0],
            n_bins=1000,
            color=[true_color, pred_color],
            linestyle=[true_ls, pred_ls],
            linewidth=[true_lw, pred_lw],
        )
        # plt.hist(dJdp_true[0, eParam[param], :, 0],bins=2000, density=True)
        # plt.hist(dJdp_esn[0, 0, eParam[param], :, 0],bins=2000, density=True, alpha = 0.5)
        # vis.plot_statistics(dJdp_true[0,eParam[param],:,0],
        #                     *[dJdp_esn[e_idx,0,eParam[param],:,0] for e_idx in range(n_esns)],)
        ylims = ax[i, res_idx].get_ylim()
        h1 = plt.vlines(
            dJdp_true_mean,
            ymin=ylims[0],
            ymax=ylims[1],
            color=true_mean_color,
            linestyle=true_ls,
            linewidth=2.5,
        )
        h2 = plt.vlines(
            dJdp_esn_mean,
            ymin=ylims[0],
            ymax=ylims[1],
            color=esn_mean_color,
            linestyle=pred_ls,
            linewidth=2.5,
        )
        plt.xlabel(f"$dJ/d\\{param}$")
        plt.xlim(
            [dJdp_true_mean - 2 * dJdp_true_std, dJdp_true_mean + 2 * dJdp_true_std]
        )
        ax[i, res_idx].annotate(titles[tt], xy=(0.03, 0.85), xycoords="axes fraction")
        tt += 1
ax[0, 0].set_ylabel("PDF")
ax[1, 0].set_ylabel("PDF")

h = []
for child in plt.gca().get_children():
    if isinstance(child, plt.Line2D):
        h.append(child)
h.append(h1)
h.append(h2)
# subfigs[1].legend(
#     h,
#     ["True", "ESN", "True mean", "ESN mean"],
#     ncol=4,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.1),
# )
# ax[0, 0].legend(
#     h, ["True", "ESN"], ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3)
# )
legend_str = ["True", "ESN", "True mean", "ESN mean"]
plt.figlegend(
    h, legend_str, loc="upper right", ncols=len(legend_str), bbox_to_anchor=(1.0, 1.12)
)
if save_fig:
    fig.savefig(
        f"paper/graphics/figure_chaotic_sensitivity.png",
        bbox_inches="tight",
        dpi=300,
    )
    fig.savefig(f"paper/graphics/figure_chaotic_sensitivity.pdf", bbox_inches="tight")
plt.show()
