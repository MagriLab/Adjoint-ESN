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
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 12})
rc("text", usetex=True)
save_fig = True
model_path = Path("local_results/rijke/run_20231029_153121")  # rijke with reservoir
data_dir = Path("data")

fig_name = "chaotic"

if fig_name == "lco_1":
    test_param_list = [[2.0, 0.25]]
    phase_space_steps_arr = [2]
    true_phase_lw = 2
    pred_phase_lw = 2
elif fig_name == "lco_2":
    test_param_list = [[4.5, 0.12]]
    phase_space_steps_arr = [2]
    true_phase_lw = 2
    pred_phase_lw = 2
elif fig_name == "period_double":
    test_param_list = [[7.5, 0.3]]
    phase_space_steps_arr = [4]
    true_phase_lw = 2
    pred_phase_lw = 2
elif fig_name == "quasi":
    test_param_list = [[6.1, 0.2]]
    phase_space_steps_arr = [100]
    true_phase_lw = 1
    pred_phase_lw = 1
elif fig_name == "chaotic":
    test_param_list = [[7.2, 0.2]]
    phase_space_steps_arr = [100]
    true_phase_lw = 1
    pred_phase_lw = 1

n_ensemble = 10
test_loop_times = [10, 500]
test_loop_names = ["short", "long"]
test_sim_time = 1000

config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]

true_color = "black"
pred_color = "red"
true_lw = 2
pred_lw = 2
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

# Predict on the test dataset
for p_idx, p in enumerate(test_param_list):
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    subfigs = fig.subfigures(1, 3, width_ratios=[1, 1, 1.2])

    p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
    y_sim, t_sim = pp.load_data(
        beta=p_sim["beta"],
        tau=p_sim["tau"],
        x_f=0.2,
        N_g=N_g,
        sim_time=test_sim_time,
        sim_dt=sim_dt,
        data_dir=data_dir,
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

    Y_PRED_SHORT = [None] * n_ensemble
    Y_PRED_LONG = [None] * n_ensemble
    for e_idx in range(n_ensemble):
        # fix the seeds
        input_seeds = [5 * e_idx, 5 * e_idx + 1, 5 * e_idx + 2]
        reservoir_seeds = [5 * e_idx + 3, 5 * e_idx + 4]

        # expand the ESN dict with the fixed seeds
        ESN_dict["input_seeds"] = input_seeds
        ESN_dict["reservoir_seeds"] = reservoir_seeds

        # create an ESN
        print(f"Creating ESN {e_idx+1}/{n_ensemble}.")
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
        Y_PRED_SHORT[e_idx] = y_pred_short

        # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
        y0 = np.zeros((1, data["long"]["u_washout"].shape[1]))
        y0[0, 0] = 1
        u_washout_auto = np.repeat(y0, [len(data["long"]["u_washout"])], axis=0)
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
        Y_PRED_LONG[e_idx] = y_pred_long

    pred_short_error = np.array(
        [errors.rel_L2(data["short"]["y"], Y_PRED_SHORT[i]) for i in range(n_ensemble)]
    )
    print("Short term prediction errors: ", pred_short_error)
    best_idx = np.argmin(pred_short_error)
    print("Best idx: ", best_idx)

    # SHORT-TERM AND TIME-ACCURATE PREDICTION
    # Plot short term prediction timeseries of the best of ensemble
    ax1 = subfigs[0].add_subplot(4, 1, 1)
    vis.plot_lines(
        data["short"]["t"] - data["short"]["t"][0],
        data["short"]["y"][:, plt_idx[0]],
        Y_PRED_SHORT[best_idx][:, plt_idx[0]],
        ylabel=f"$\{plt_idx[0].name}$",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )
    ax1.legend(["True", "ESN"], loc="center", bbox_to_anchor=[0.5, 1.2], ncol=2)
    ax1.set_xticklabels([])

    ax2 = subfigs[0].add_subplot(4, 1, 2)
    vis.plot_lines(
        data["short"]["t"] - data["short"]["t"][0],
        data["short"]["y"][:, plt_idx[1]],
        Y_PRED_SHORT[best_idx][:, plt_idx[1]],
        ylabel=f"$\{plt_idx[1].name}$",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )
    ax2.set_xticklabels([])

    ax3 = subfigs[0].add_subplot(4, 1, 3)
    vis.plot_lines(
        data["short"]["t"] - data["short"]["t"][0],
        data["short"]["y"][:, plt_idx[2]],
        Y_PRED_SHORT[best_idx][:, plt_idx[2]],
        ylabel=f"$\{plt_idx[2].name}$",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )
    ax3.set_xticklabels([])

    ax4 = subfigs[0].add_subplot(4, 1, 4)
    vis.plot_lines(
        data["short"]["t"] - data["short"]["t"][0],
        data["short"]["y"][:, plt_idx[3]],
        Y_PRED_SHORT[best_idx][:, plt_idx[3]],
        ylabel=f"$\{plt_idx[3].name}$",
        xlabel="$t$",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )

    # Plot phase plot of the best of ensemble
    phase_space_steps = pp.get_steps(phase_space_steps_arr[p_idx], network_dt)

    ax5 = subfigs[1].add_subplot(2, 1, 1)
    vis.plot_phase_space(
        data["long"]["y"][-phase_space_steps:],
        Y_PRED_LONG[best_idx][-phase_space_steps:],
        idx_pair=plt_idx_pairs[0],
        linestyle=[true_ls, pred_ls],
        linewidth=[true_phase_lw, pred_phase_lw],
        color=[true_color, pred_color],
        xlabel=f"$\{plt_idx_pairs[0][0].name}$",
        ylabel=f"$\{plt_idx_pairs[0][1].name}$",
    )

    ax6 = subfigs[1].add_subplot(2, 1, 2)
    vis.plot_phase_space(
        data["long"]["y"][-phase_space_steps:],
        Y_PRED_LONG[best_idx][-phase_space_steps:],
        idx_pair=plt_idx_pairs[1],
        linestyle=[true_ls, pred_ls],
        linewidth=[true_phase_lw, pred_phase_lw],
        color=[true_color, pred_color],
        xlabel=f"$\{plt_idx_pairs[1][0].name}$",
        ylabel=f"$\{plt_idx_pairs[1][1].name}$",
    )

    # # Plot statistics
    ax7 = subfigs[2].add_subplot(2, 2, 1)
    vis.plot_statistics_ensemble(
        *[Y_PRED_LONG[i][:, plt_idx[0]] for i in range(n_ensemble)],
        y_base=data["long"]["y"][:, plt_idx[0]],
        xlabel=f"$\{plt_idx[0].name}$",
        ylabel="PDF",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color],
    )

    ax8 = subfigs[2].add_subplot(2, 2, 3)
    vis.plot_statistics_ensemble(
        *[Y_PRED_LONG[i][:, plt_idx[1]] for i in range(n_ensemble)],
        y_base=data["long"]["y"][:, plt_idx[1]],
        xlabel=f"$\{plt_idx[1].name}$",
        ylabel="PDF",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color, pred_color],
    )

    ax9 = subfigs[2].add_subplot(2, 2, 2)
    vis.plot_statistics_ensemble(
        *[Y_PRED_LONG[i][:, plt_idx[2]] for i in range(n_ensemble)],
        y_base=data["long"]["y"][:, plt_idx[2]],
        xlabel=f"$\{plt_idx[2].name}$",
        ylabel="PDF",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color, pred_color],
    )

    ax10 = subfigs[2].add_subplot(2, 2, 4)
    vis.plot_statistics_ensemble(
        *[Y_PRED_LONG[i][:, plt_idx[3]] for i in range(n_ensemble)],
        y_base=data["long"]["y"][:, plt_idx[3]],
        xlabel=f"$\{plt_idx[3].name}$",
        ylabel="PDF",
        linestyle=[true_ls, pred_ls],
        linewidth=[true_lw, pred_lw],
        color=[true_color, pred_color, pred_color],
    )

if save_fig:
    fig.savefig(f"paper/graphics/figure_{fig_name}.png", bbox_inches="tight")
plt.show()
