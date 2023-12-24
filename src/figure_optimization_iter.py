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
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import signals
from adjoint_esn.utils.enums import eParam, get_eVar


def get_asd(dt, y, remove_mean=True, periodic=False):
    # remove mean
    if remove_mean == True:
        y = y - np.mean(y)
    if periodic:
        T_period = signals.period(y, dt)
        data_omega = 2 * np.pi / T_period
        print("Omega = ", data_omega)
        print("Period = ", T_period)
        # take the maximum number of periods
        # the real period isn't an exact multiple of the sampling time
        # therefore, the signal doesn't repeat itself at exact integer indices
        # so calculating the number of time steps in each period
        # does not work in order to cut the signal at the maximum number of periods
        # that's why we will cut between peaks, which is a more reliable option
        # though still not exact
        min_dist = pp.get_steps(T_period - 0.1, dt)
        (start_pk_idx, end_pk_idx) = signals.periodic_signal_peaks(y, T=min_dist)
        y_pre_fft = y[
            start_pk_idx:end_pk_idx
        ]  # don't include end peak for continuous signal
    else:
        y_pre_fft = y

    # find asd
    omega, asd = signals.amplitude_spectrum(y_pre_fft, dt)

    # to get the harmonic frequency from the asd
    # asd_peaks = find_peaks(asd, threshold=0.1)[0]
    # harmonic_freq = omega[asd_peaks][0]
    return omega, asd


rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)
save_fig = True
model_path = Path("local_results/rijke/run_20231029_153121")  # rijke with reservoir
optimization_path = "20231129_180629"

data_dir = Path("data")

fig_name = "optimization_path_1"

phase_space_steps_arr = 2
test_loop_times = [20, 100]
test_loop_names = ["short", "long"]
test_sim_time = 410

eta_1_init = 1.5

figure_size = (15, 2)

config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]
opt_results = pp.unpickle_file(
    model_path / f"optimization_results_{optimization_path}.pickle"
)[0]
# check energy and cut off iterations
try:
    valid_iter_idx = np.where(np.array(opt_results["hist"]["J"]) < 1e-4)[0][0]
    p_hist = np.array(opt_results["hist"]["p"])[: valid_iter_idx + 1]
except:
    p_hist = np.array(opt_results["hist"]["p"])

print(p_hist)
iter_idx = [int(i) for i in np.arange(len(p_hist))]
iter_idx = [0, 2, 4, 8]
p_hist = p_hist[iter_idx]

true_color = "silver"
pred_color = "tab:red"
true_lw = 6.0
pred_lw = 2.0
true_ls = "-"
pred_ls = "--"

# DATA creation
integrator = "odeint"

# number of galerkin modes
N_g = config.simulation.N_g

# simulation options
sim_time = config.simulation.sim_time
sim_dt = config.simulation.sim_dt

# which regimes to use for training and validation
train_param_list = results["training_parameters"]
train_idx_list = np.arange(len(train_param_list))

transient_time = config.simulation.transient_time

noise_level = config.simulation.noise_level

random_seed = config.random_seed

# network time step
network_dt = config.model.network_dt

washout_time = config.model.washout_time

# which states to use as input and output
# for standard ESN these should be the same, e.g. both 'eta_mu'
# for Rijke ESN, input and output should be 'eta_mu_v_tau' and 'eta_mu' respectively
input_vars = config.model.input_vars
output_vars = config.model.output_vars

eInputVar = get_eVar(input_vars, N_g)
eOutputVar = get_eVar(output_vars, N_g)

plt_idx = [eOutputVar.mu_1, eOutputVar.mu_2, eOutputVar.mu_3, eOutputVar.mu_4]
plt_idx_pairs = [[eOutputVar.mu_1, eOutputVar.mu_2], [eOutputVar.mu_3, eOutputVar.mu_4]]

# which system parameter is passed to the ESN
param_vars = config.model.param_vars

# if using Rijke ESN what is the order of u_f(t-tau) in the inputs,
# [u_f(t-tau), u_f(t-tau)^2 ..., u_f(t-tau)^(u_f_order)]
u_f_order = config.model.u_f_order

# length of training time series
train_time = config.train.time

loop_names = ["train"]
loop_times = [train_time]

print("Loading training data.")
DATA = {}
for loop_name in loop_names:
    DATA[loop_name] = {
        "u_washout": [],
        "p_washout": [],
        "u": [],
        "p": [],
        "y": [],
        "t": [],
    }

for p_idx, p in enumerate(train_param_list):
    p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
    y_sim, t_sim = pp.load_data(
        beta=p_sim["beta"],
        tau=p_sim["tau"],
        x_f=0.2,
        N_g=N_g,
        sim_time=sim_time,
        sim_dt=sim_dt,
        data_dir=data_dir,
        integrator=integrator,
    )

    regime_data = pp.create_dataset(
        y_sim,
        t_sim,
        p_sim,
        network_dt=network_dt,
        transient_time=transient_time,
        washout_time=washout_time,
        loop_times=loop_times,
        loop_names=loop_names,
        input_vars=input_vars,
        output_vars=output_vars,
        param_vars=param_vars,
        N_g=N_g,
        u_f_order=u_f_order,
        noise_level=noise_level,
        random_seed=random_seed,
    )

    for loop_name in loop_names:
        [
            DATA[loop_name][var].append(regime_data[loop_name][var])
            for var in DATA[loop_name].keys()
        ]

# dimension of the inputs
dim = DATA["train"]["u"][0].shape[1]

# get the properties of the best ESN from the results
(
    ESN_dict,
    hyp_param_names,
    hyp_param_scales,
    hyp_params,
) = post.get_ESN_properties_from_results(config, results, dim)
ESN_dict["verbose"] = False
print(ESN_dict)
[
    print(f"{hyp_param_name}: {hyp_param}")
    for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params)
]

# fix the seeds
input_seeds = [20, 21, 22]
reservoir_seeds = [23, 24]

# expand the ESN dict with the fixed seeds
ESN_dict["input_seeds"] = input_seeds
ESN_dict["reservoir_seeds"] = reservoir_seeds

my_ESN = post.create_ESN(
    ESN_dict, config.model.type, hyp_param_names, hyp_param_scales, hyp_params
)
print("Training ESN.")
my_ESN.train(
    DATA["train"]["u_washout"],
    DATA["train"]["u"],
    DATA["train"]["y"],
    P_washout=DATA["train"]["p_washout"],
    P_train=DATA["train"]["p"],
    train_idx_list=train_idx_list,
)
titles = ["(a)", "(b)", "(c)", "(d)"]
transient_time = 200
# Predict on the test dataset
for p_idx, p in enumerate(p_hist):
    fig = plt.figure(figsize=figure_size, constrained_layout=True)
    subfigs = fig.subfigures(1, 3, width_ratios=[1, 1.8, 1.2])

    # set up the initial conditions
    y0 = np.zeros((1, DATA["train"]["u_washout"][0].shape[1]))
    y0[0, 0] = eta_1_init
    u_washout_auto = np.repeat(y0, [len(DATA["train"]["u_washout"][0])], axis=0)

    y0_sim = np.zeros(2 * N_g + 10)
    y0_sim[0] = eta_1_init

    p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
    y_sim, t_sim = pp.load_data(
        beta=p_sim["beta"],
        tau=p_sim["tau"],
        x_f=0.2,
        N_g=N_g,
        sim_time=test_sim_time,
        sim_dt=sim_dt,
        data_dir=data_dir,
        y_init=y0_sim,
    )

    data = pp.create_dataset(
        y_sim,
        t_sim,
        p_sim,
        network_dt=network_dt,
        transient_time=transient_time,
        washout_time=washout_time,
        loop_times=test_loop_times,
        loop_names=test_loop_names,
        input_vars=input_vars,
        output_vars=output_vars,
        param_vars=param_vars,
        N_g=N_g,
        u_f_order=u_f_order,
        start_idxs=[0, 0],
    )

    print("Predicting.")
    if hasattr(my_ESN, "tau"):
        my_ESN.tau = p_sim["tau"]

    _, y_pred_short = my_ESN.closed_loop_with_washout(
        U_washout=data["short"]["u_washout"],
        N_t=len(data["short"]["u"]),
        P_washout=data["short"]["p_washout"],
        P=data["short"]["p"],
    )
    y_pred_short = y_pred_short[1:]

    # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
    transient_steps = pp.get_steps(transient_time, network_dt)
    # add the transient time that will be discarded later
    N_t_long = transient_steps + len(data["long"]["u"])
    p_long0 = np.zeros((1, data["long"]["p"].shape[1]))
    p_long0[0] = data["long"]["p"][0]
    p_long = np.repeat(p_long0, [N_t_long], axis=0)
    # Predict long-term
    _, y_pred_long = my_ESN.closed_loop_with_washout(
        U_washout=u_washout_auto,
        N_t=N_t_long,
        P_washout=data["long"]["p_washout"],
        P=p_long,
    )
    y_pred_long = y_pred_long[1:]
    y_pred_long = y_pred_long[transient_steps:, :]

    phase_space_steps = pp.get_steps(phase_space_steps_arr, network_dt)

    # SHORT-TERM AND TIME-ACCURATE PREDICTION
    # Plot phase plot
    ax1 = subfigs[0].add_subplot(1, 1, 1)
    vis.plot_phase_space(
        data["long"]["y"][-phase_space_steps:],
        y_pred_long[-phase_space_steps:],
        idx_pair=plt_idx_pairs[0],
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
        xlabel=f"$\{plt_idx_pairs[0][0].name}$",
        ylabel=f"$\{plt_idx_pairs[0][1].name}$",
    )
    ax1.set_xlim([-7.5, 7.5])
    ax1.set_ylim([-7.5, 7.5])

    # Plot short term prediction
    ax2 = subfigs[1].add_subplot(1, 1, 1)
    vis.plot_lines(
        data["short"]["t"] - data["short"]["t"][0],
        sens.acoustic_energy_inst(data["short"]["y"], N_g),
        sens.acoustic_energy_inst(y_pred_short, N_g),
        xlabel="$t$",
        ylabel="$E_{ac}$",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )
    # plt.hlines(y = np.mean(sens.acoustic_energy_inst(data["short"]["y"], N_g)), xmin = 0, xmax = 20, color = 'tab:blue')
    # plt.hlines(y = np.mean( sens.acoustic_energy_inst(y_pred_short, N_g)), xmin = 0, xmax = 20, color = 'tab:orange', linestyle="--")
    ax2.set_ylim([0, 35.0])

    # Plot power spectral density
    ax3 = subfigs[2].add_subplot(1, 1, 1)
    omega, asd = get_asd(
        network_dt,
        sens.acoustic_energy_inst(data["long"]["y"], N_g),
        remove_mean=True,
        periodic=True,
    )
    omega_pred, asd_pred = get_asd(
        network_dt,
        sens.acoustic_energy_inst(y_pred_long, N_g),
        remove_mean=True,
        periodic=True,
    )
    vis.plot_asd(
        asd_y=asd_pred,
        omega_y=omega_pred,
        asd_y_base=asd,
        omega_y_base=omega,
        range=10,
        xlabel="$\omega$",
        ylabel="$ASD(E_{ac})$",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )
    ax3.legend(["True", "ESN"], loc="upper right")
    subfigs[0].suptitle(titles[p_idx], x=0.0, y=1.025)
    if save_fig:
        fig.savefig(
            f"paper/graphics/figure_{fig_name}_iter_{iter_idx[p_idx]}.png",
            bbox_inches="tight",
        )
plt.show()
