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
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.rijke_galerkin import sensitivity as sens
from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import signals
from adjoint_esn.utils.enums import eParam, get_eVar

plt.style.use("src/stylesheet.mplstyle")
cmap = cm.create_custom_colormap(type="discrete")

figure_size = (15, 4)

save_fig = False
fig_name = "lco1"

model_paths = [
    Path(
        "final_results/rijke/run_20231029_153121"
    ),  # rijke with reservoir, trained on beta = 1,2,3,4,5
    Path(
        "final_results/rijke/run_20240307_175258"
    ),  # rijke with reservoir, trained on beta = 6,6.5,7,7.5,8
]

if fig_name == "lco1":
    beta = 2.0
    tau = 0.25
    periodic = True
    titles = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    LT = 1.0
    t_label = "$t$"
    model_paths = [model_paths[0]]
    ts_path_str = "timeseries_beta_2_00_tau_0_25_results"
    ts_paths = ["20240402_115212"]

    lt_path_str = "long_term_beta_2_00_tau_0_25_results"
    lt_paths = ["20240402_120405"]
    plt_e_idx = 0
    plt_loop_idx = 0
elif fig_name == "lco2":
    beta = 4.5
    tau = 0.12
    periodic = True
    titles = [["(g)", "(h)", "(i)"], ["(j)", "(k)", "(l)"]]
    LT = 1.0
    t_label = "$t$"
    model_paths = [model_paths[0]]
    ts_path_str = "timeseries_beta_4_50_tau_0_12_results"
    ts_paths = ["20240402_115309"]

    lt_path_str = "long_term_beta_4_50_tau_0_12_results"
    lt_paths = ["20240402_120652"]
    plt_e_idx = 0
    plt_loop_idx = 0
elif fig_name == "period_double":
    beta = 7.5
    tau = 0.3
    periodic = True
    titles = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    LT = 1.0
    t_label = "$t$"
    ts_path_str = "timeseries_beta_7_50_tau_0_30_results"
    ts_paths = [
        "20240402_122752",
        "20240402_123125",
    ]

    lt_path_str = "long_term_beta_7_50_tau_0_30_results"
    lt_paths = [
        "20240402_123037",
        "20240402_123406",
    ]
    plt_e_idx = 0
    plt_loop_idx = 0
elif fig_name == "quasi":
    beta = 6.1
    tau = 0.2
    periodic = False
    titles = [["(g)", "(h)", "(i)"], ["(j)", "(k)", "(l)"]]
    LT = 1.0
    t_label = "$t$"
    ts_path_str = "timeseries_beta_6_10_tau_0_20_results"
    ts_paths = [
        "20240402_131958",
        "20240402_132522",
    ]

    lt_path_str = "long_term_beta_6_10_tau_0_20_results"
    lt_paths = [
        "20240402_132403",
        "20240402_132927",
    ]
    plt_e_idx = 0
    plt_loop_idx = 0
elif fig_name == "chaotic1":
    beta = 7.6
    tau = 0.22
    periodic = False
    titles = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    LT = 8.5
    t_label = "$t [LT]$"
    model_paths = model_paths
    ts_path_str = "timeseries_beta_7_60_tau_0_22_results"
    ts_paths = [
        "20240402_132343",
        "20240402_142116",
    ]

    lt_path_str = "long_term_beta_7_60_tau_0_22_results"
    lt_paths = [
        "20240402_142014",
        "20240402_143404",
    ]
    plt_e_idx = 0
    plt_loop_idx = 0
elif fig_name == "chaotic2":
    beta = 8.7
    tau = 0.23
    periodic = False
    titles = [["(g)", "(h)", "(i)"], ["(j)", "(k)", "(l)"]]
    LT = 3.9
    t_label = "$t [LT]$"
    model_paths = model_paths
    ts_path_str = "timeseries_beta_8_70_tau_0_23_results"
    ts_paths = [
        "20240402_132339",
        "20240402_132934",
    ]

    lt_path_str = "long_term_beta_8_70_tau_0_23_results"
    lt_paths = [
        "20240402_132900",
        "20240402_133638",
    ]
    plt_e_idx = 0
    plt_loop_idx = 1

true_color = cmap(0)
true_lw = 5.0
true_ls = "-"

fig1_names = ["lco1", "lco2"]
fig2_names = ["period_double", "quasi", "chaotic1", "chaotic2"]
if fig_name in fig1_names:
    legend_str = [
        "True",
        "ESN",
    ]
    pred_colors = [cmap(2)]
    pred_lws = [2.5]
    pred_lss = ["--"]
elif fig_name in fig2_names:
    legend_str = [
        "True",
        "$\\beta_{\mathrm{train}}=[1,5]$",
        "$\\beta_{\mathrm{train}}=[6,8]$",
    ]
    pred_colors = [cmap(1), cmap(2)]
    pred_lws = [2.5, 1.5]
    pred_lss = ["-", "--"]

linespecs = {
    "linestyle": [true_ls, *pred_lss],
    "linewidth": [true_lw, *pred_lws],
    "color": [true_color, *pred_colors],
}

n_models = len(model_paths)

config = post.load_config(model_paths[0])

input_vars = config.model.input_vars
output_vars = config.model.output_vars
N_g = config.simulation.N_g
eInputVar = get_eVar(input_vars, N_g)
eOutputVar = get_eVar(output_vars, N_g)

plt_idx = [eOutputVar.mu_1]

Y_PRED_SHORT = [None] * len(model_paths)
Y_PRED_LONG = [None] * len(model_paths)
for model_idx, model_path in enumerate(model_paths):
    param_vars = config.model.param_vars

    # get short timeseries results
    ts_res_path = model_path / f"{ts_path_str}_{ts_paths[model_idx]}.pickle"
    ts_res = pp.unpickle_file(ts_res_path)[0]
    y_short = ts_res["y_true"]
    Y_PRED_SHORT[model_idx] = ts_res["y_pred"]

    # get long term results
    lt_res_path = model_path / f"{lt_path_str}_{lt_paths[model_idx]}.pickle"
    long_term_res = pp.unpickle_file(lt_res_path)[0]
    Y_PRED_LONG[model_idx] = long_term_res["y_pred"]

n_ensemble = len(long_term_res["y_pred"])

# load long term data
data_dir = Path("data")
test_sim_time = long_term_res["loop_time"] + config.simulation.transient_time
p_sim = {"beta": beta, "tau": tau}
y_sim, t_sim = pp.load_data(
    beta=p_sim["beta"],
    tau=p_sim["tau"],
    x_f=0.2,
    N_g=N_g,
    sim_time=test_sim_time,
    sim_dt=config.simulation.sim_dt,
    data_dir=data_dir,
)

data = pp.create_dataset(
    y_sim,
    t_sim,
    p_sim,
    network_dt=config.model.network_dt,
    transient_time=config.simulation.transient_time,
    washout_time=0,
    loop_times=[long_term_res["loop_time"]],
    loop_names=["long"],
    input_vars=input_vars,
    output_vars=output_vars,
    param_vars=param_vars,
    N_g=N_g,
    u_f_order=config.model.u_f_order,
    start_idxs=[0, 0],
    tau=p_sim["tau"],
)
y_long = data["long"]["y"]

# SHORT-TERM AND TIME-ACCURATE PREDICTION
# Plot short term prediction timeseries
network_dt = config.model.network_dt
t_short = np.arange(0, len(y_short[plt_loop_idx])) * network_dt
fig = plt.figure(figsize=figure_size, constrained_layout=True)
subfigs = fig.subfigures(1, 3, width_ratios=[1.8, 1, 1.2])
for i in range(len(plt_idx)):
    ax = subfigs[0].add_subplot(len(plt_idx) + 1, 1, i + 1)
    vis.plot_lines(
        t_short / LT,
        y_short[plt_loop_idx][:, plt_idx[i]],
        *[
            Y_PRED_SHORT[model_idx][plt_loop_idx][plt_e_idx][:, plt_idx[i]]
            for model_idx in range(n_models)
        ],
        xlabel=t_label,
        ylabel=f"$\{plt_idx[i].name}$",
        **linespecs,
    )
    ax.annotate(titles[0][0], xy=(0.015, 0.85), xycoords="axes fraction")

for model_idx in range(1):
    error_measure = errors.rel_L2
    error = error_measure(
        y_short[plt_loop_idx][:, : 2 * N_g],
        Y_PRED_SHORT[model_idx][plt_loop_idx][plt_e_idx][:, : 2 * N_g],
    )
    print(error)

ax = subfigs[0].add_subplot(len(plt_idx) + 1, 1, len(plt_idx) + 1)
vis.plot_lines(
    t_short / LT,
    sens.acoustic_energy_inst(y_short[plt_loop_idx], N_g),
    *[
        sens.acoustic_energy_inst(Y_PRED_SHORT[model_idx][plt_loop_idx][plt_e_idx], N_g)
        for model_idx in range(n_models)
    ],
    xlabel=t_label,
    ylabel="$E_{ac}$",
    **linespecs,
)
ax.annotate(titles[1][0], xy=(0.015, 0.85), xycoords="axes fraction")

# LONG-TERM STATISTICS
# Plot statistics
for i in range(len(plt_idx)):
    ax = subfigs[1].add_subplot(len(plt_idx) + 1, 1, i + 1)
    vis.plot_statistics_ensemble(
        *[
            [Y_PRED_LONG[m][e][:, plt_idx[i]] for e in range(n_ensemble)]
            for m in range(n_models)
        ],
        y_base=y_long[:, plt_idx[i]],
        xlabel=f"$\{plt_idx[i].name}$",
        ylabel="PDF",
        **linespecs,
    )
    ax.annotate(titles[0][1], xy=(0.03, 0.85), xycoords="axes fraction")

# Plot statistics of acoustic energy
ax = subfigs[1].add_subplot(len(plt_idx) + 1, 1, len(plt_idx) + 1)
vis.plot_statistics_ensemble(
    *[
        [sens.acoustic_energy_inst(Y_PRED_LONG[m][e], N_g) for e in range(n_ensemble)]
        for m in range(n_models)
    ],
    y_base=sens.acoustic_energy_inst(y_long, N_g),
    xlabel="$E_{ac}$",
    ylabel="PDF",
    **linespecs,
)
ax.annotate(titles[1][1], xy=(0.03, 0.85), xycoords="axes fraction")

# AMPLITUDE SPECTRUM
# Plot amplitude spectrum
for i in range(len(plt_idx)):
    ax = subfigs[2].add_subplot(len(plt_idx) + 1, 1, i + 1)
    omega, amp_spec = signals.get_amp_spec(
        network_dt,
        y_long[:, plt_idx[i]],
        remove_mean=True,
        periodic=periodic,
    )
    AS_PRED = [[None] * n_ensemble for _ in range(n_models)]
    OMEGA_PRED = [[None] * n_ensemble for _ in range(n_models)]
    for model_idx in range(n_models):
        for e_idx in range(n_ensemble):
            (
                OMEGA_PRED[model_idx][e_idx],
                AS_PRED[model_idx][e_idx],
            ) = signals.get_amp_spec(
                network_dt,
                Y_PRED_LONG[model_idx][e_idx][:, plt_idx[i]],
                remove_mean=True,
                periodic=periodic,
            )
    ax.annotate(titles[0][2], xy=(0.03, 0.85), xycoords="axes fraction")
    vis.plot_asd(  # *[AS_PRED[e] for e in range(n_ensemble)],
        asd_y=[AS_PRED[model_idx][plt_e_idx] for model_idx in range(n_models)],
        omega_y=[OMEGA_PRED[model_idx][plt_e_idx] for model_idx in range(n_models)],
        asd_y_base=amp_spec,
        omega_y_base=omega,
        range=10,
        xlabel="$\omega$",
        ylabel=f"Amplitude($\{plt_idx[i].name}$)",
        **linespecs,
        alpha=0.8,
    )
    plt.legend(legend_str, loc="upper right", handlelength=1.0, handletextpad=0.5)
ax = subfigs[2].add_subplot(len(plt_idx) + 1, 1, len(plt_idx) + 1)
omega, amp_spec = signals.get_amp_spec(
    network_dt,
    sens.acoustic_energy_inst(data["long"]["y"], N_g),
    remove_mean=True,
    periodic=periodic,
)
# Plot amplitude spectrum of acoustic energy
AS_PRED = [[None] * n_ensemble for _ in range(n_models)]
OMEGA_PRED = [[None] * n_ensemble for _ in range(n_models)]
for model_idx in range(n_models):
    for e_idx in range(n_ensemble):
        OMEGA_PRED[model_idx][e_idx], AS_PRED[model_idx][e_idx] = signals.get_amp_spec(
            network_dt,
            sens.acoustic_energy_inst(Y_PRED_LONG[model_idx][e_idx], N_g),
            periodic=periodic,
        )
vis.plot_asd(
    asd_y=[AS_PRED[model_idx][plt_e_idx] for model_idx in range(n_models)],
    omega_y=[OMEGA_PRED[model_idx][plt_e_idx] for model_idx in range(n_models)],
    asd_y_base=amp_spec,
    omega_y_base=omega,
    range=10,
    xlabel="$\omega$",
    ylabel="Amplitude($E_{ac}$)",
    alpha=0.8,
    **linespecs,
)
ax.annotate(titles[1][2], xy=(0.03, 0.85), xycoords="axes fraction")
if save_fig:
    fig.savefig(f"paper/graphics/figure_{fig_name}.png", bbox_inches="tight")
    fig.savefig(f"paper/graphics/figure_{fig_name}.pdf", bbox_inches="tight")
plt.show()
