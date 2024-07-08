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
fig_name = "lco1_noisy_10"

model_paths = [
    Path(
        "local_results/rijke/run_20231029_153121_noise_10_new"
    ),  # rijke with reservoir, trained on beta = 1,2,3,4,5 + 10% noise
    Path(
        "local_results/rijke/run_20231029_153121_noise_10_tikh_5_new"
    ),  # rijke with reservoir, trained on beta = 1,2,3,4,5 + 10% noise
    # Path(
    # "local_results/rijke/run_20231029_153121_noise_10_tikh_4_new"
    #  ),  # rijke with reservoir, trained on beta = 1,2,3,4,5 + 5% noise
    Path(
        "local_results/rijke/run_20231029_153121_noise_10_tikh_3_new"
    ),  # rijke with reservoir, trained on beta = 1,2,3,4,5 + 10% noise
    # Path(
    # "local_results/rijke/run_20231029_153121_noise_10_tikh_2_new"
    # ),  # rijke with reservoir, trained on beta = 1,2,3,4,5 + 5% noise
]

periodic = True
titles = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"], ["(g)", "(h)", "(i)"]]
LT = 1.0
t_label = "$t$"
beta = 2.0
tau = 0.25
# model_paths = [model_paths[0],model_paths[1],]
ts_path_str = "timeseries_beta_2_00_tau_0_25_results"
ts_paths = [
    "20240708_200911",  # tikh = 1e-6
    "20240708_200953",  # tikh = 1e-5
    # "20240708_201044",  #tikh = 1e-4
    "20240708_201143",  # tikh = 1e-3
    # "20240708_201237" #tikh = 1e-2
]

lt_path_str = "long_term_beta_2_00_tau_0_25_results"
lt_paths = [
    "20240708_203833",  # tikh = 1e-6
    "20240708_203915",  # tikh = 1e-5
    # "20240708_201600", #tikh = 1e-4
    "20240708_201934",  # tikh = 1e-3
    # "20240708_201854" #tikh = 1e-2
]


plt_e_idx = 0
plt_loop_idx = 0

legend_str = [
    "Train",
    "True",
    "$\lambda = 10^{-6}$",
    "$\lambda = 10^{-5}$",
    "$\lambda = 10^{-3}$",
]

train_color = cmap(3)
train_lw = 1.0
train_ls = "None"

true_color = cmap(0)
true_lw = 4.5
true_ls = "-"

pred_colors = [cmap(4), cmap(1), cmap(2)]
pred_lws = [2.5, 2.0, 1.5]
pred_lss = ["-", "--", "-."]

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
plt_idx_pairs = [[eOutputVar.mu_1, eOutputVar.mu_2]]

Y_PRED_SHORT = [None] * len(model_paths)
Y_PRED_LONG = [None] * len(model_paths)
for model_idx, model_path in enumerate(model_paths):
    param_vars = config.model.param_vars

    # get short timeseries results
    ts_res_path = model_path / f"{ts_path_str}_{ts_paths[model_idx]}.pickle"
    ts_res = pp.unpickle_file(ts_res_path)[0]
    y_train = ts_res["y_train"]
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
t_train = np.arange(0, len(y_train)) * network_dt
t_short = np.arange(0, len(y_short[plt_loop_idx])) * network_dt
plt_loop_idx = 0
fig = plt.figure(figsize=figure_size, constrained_layout=True)
# subfigs_var = subfigs1[0].subfigures(1, 4, width_ratios=[0.9, 0.9, 1.0, 1.2])
subfigs = fig.subfigures(1, 3, width_ratios=[1.8, 1, 1.2])
for i in range(len(plt_idx)):
    ax = subfigs[0].add_subplot(len(plt_idx) + 1, 1, i + 1)
    plt.plot(
        t_train / LT,
        y_train[:, plt_idx[i]],
        color=train_color,
        marker="o",
        markersize=8,
        linestyle="None",
    )
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
    ax.annotate(titles[i][0], xy=(0.015, 0.85), xycoords="axes fraction")
    # Plot phase plot
    ax1 = subfigs[1].add_subplot(len(plt_idx) + 1, 1, i + 1)
    phase_space_steps = pp.get_steps(2.0, network_dt)
    vis.plot_phase_space(
        y_train,
        y_short[plt_loop_idx][-phase_space_steps:],
        *[
            Y_PRED_SHORT[model_idx][plt_loop_idx][plt_e_idx][-phase_space_steps:]
            for model_idx in range(n_models)
        ],
        idx_pair=plt_idx_pairs[i],
        linestyle=[train_ls, *linespecs["linestyle"]],
        linewidth=[train_lw, *linespecs["linewidth"]],
        marker=["o", "None", "None", "None", "None"],
        color=[train_color, *linespecs["color"]],
        xlabel=f"$\{plt_idx_pairs[i][0].name}$",
        ylabel=f"$\{plt_idx_pairs[i][1].name}$",
    )
    ax1.annotate(titles[i][1], xy=(0.015, 0.85), xycoords="axes fraction")

for model_idx in range(len(model_paths)):
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
ax.annotate(titles[2][0], xy=(0.015, 0.85), xycoords="axes fraction")

# LONG-TERM STATISTICS
# Plot statistics
# for i in range(len(plt_idx)):
#     ax = subfigs[2].add_subplot(len(plt_idx) + 1, 1, i + 1)
#     vis.plot_statistics_ensemble(
#         *[
#             [Y_PRED_LONG[m][e][:, plt_idx[i]] for e in range(n_ensemble)]
#             for m in range(n_models)
#         ],
#         y_base=y_long[:, plt_idx[i]],
#         xlabel=f"$\{plt_idx[i].name}$",
#         ylabel="PDF",
#         **linespecs,
#     )
#     ax.annotate(titles[0][1], xy=(0.03, 0.85), xycoords="axes fraction")

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
ax.annotate(titles[2][1], xy=(0.03, 0.85), xycoords="axes fraction")

# AMPLITUDE SPECTRUM
# Plot amplitude spectrum
for i in range(len(plt_idx)):
    ax = subfigs[2].add_subplot(len(plt_idx) + 1, 1, i + 1)
    omega_train, amp_spec_train = signals.get_amp_spec(
        network_dt,
        y_train[:, plt_idx[i]],
        remove_mean=True,
        periodic=periodic,
    )
    plt.plot(
        omega_train,
        amp_spec_train,
        color=train_color,
        marker="o",
        linestyle="None",
        markersize=4,
        zorder=10,
    )
    omega, amp_spec = signals.get_amp_spec(
        network_dt,
        y_short[plt_loop_idx][: len(y_train), plt_idx[i]],  # y_long[:, plt_idx[i]],
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
                Y_PRED_SHORT[model_idx][plt_loop_idx][e_idx][
                    : len(y_train), plt_idx[i]
                ],  # Y_PRED_LONG[model_idx][e_idx][:, plt_idx[i]],
                remove_mean=True,
                periodic=periodic,
            )
    ax.annotate(titles[i][2], xy=(0.03, 0.85), xycoords="axes fraction")
    vis.plot_asd(  # *[AS_PRED[e] for e in range(n_ensemble)],
        asd_y=[AS_PRED[model_idx][plt_e_idx] for model_idx in range(n_models)],
        omega_y=[OMEGA_PRED[model_idx][plt_e_idx] for model_idx in range(n_models)],
        asd_y_base=amp_spec,
        omega_y_base=omega,
        range=40.0,
        xlabel="$\omega$",
        ylabel=f"Amplitude($\{plt_idx[i].name}$)",
        **linespecs,
        alpha=0.8,
    )
    plt.yscale("log")
    plt.ylim([1e-5, 10])
plt.figlegend(
    legend_str, loc="upper center", ncols=len(legend_str), bbox_to_anchor=(0.5, 1.15)
)

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
    range=10.0,
    xlabel="$\omega$",
    ylabel="Amplitude($E_{ac}$)",
    alpha=0.8,
    **linespecs,
)
ax.annotate(titles[2][2], xy=(0.03, 0.85), xycoords="axes fraction")

if save_fig:
    fig.savefig(f"paper/graphics/figure_{fig_name}.png", bbox_inches="tight")
    fig.savefig(f"paper/graphics/figure_{fig_name}.pdf", bbox_inches="tight")
plt.show()
