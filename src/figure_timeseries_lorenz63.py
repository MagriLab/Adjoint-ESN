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

import adjoint_esn.utils.postprocessing as post
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 24})
rc("text", usetex=True)
save_fig = True
same_washout = False
model_path = Path("local_results/lorenz63/run_20240208_121804")

fig_name = "1"

# Create the mesh to get the data from
eParam = Lorenz63.get_eParamVar()
test_param_list = np.zeros((2, 3))

if fig_name == "1":
    test_param_list[0, eParam.beta] = 8 / 3
    test_param_list[0, eParam.rho] = 28.0
    test_param_list[0, eParam.sigma] = 10.0

    test_param_list[1, eParam.beta] = 2.0
    test_param_list[1, eParam.rho] = 52.0
    test_param_list[1, eParam.sigma] = 13.0
    LT = [1.1, 0.8]
    titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

n_ensemble = 1
short_loop_lt = 20
long_loop_lt = 5000
test_loop_names = ["short", "long"]
y_init = [-2.4, -3.7, 14.98]

figure_size = (15, 5)
config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]

true_color = "silver"
pred_color = "tab:red"
true_lw = 8.0
pred_lw = 2.5
true_ls = "-"
pred_ls = "--"
config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]

# which regimes to use for training and validation
train_param_list = results["training_parameters"]
train_idx_list = np.arange(len(train_param_list))

# which system parameter is passed to the ESN
param_vars = config.model.param_vars

# length of training time series
train_time = config.train.time

loop_names = ["train"]
loop_times = [train_time]

print("Loading training data.", flush=True)
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
    p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}
    my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
        Lorenz63,
        p_sim,
        sim_time=config.simulation.sim_time,
        sim_dt=config.simulation.sim_dt,
        random_seed=config.random_seed,
        integrator=config.simulation.integrator,
    )
    regime_data = pp.create_dataset_dyn_sys(
        my_sys,
        y_sim,
        t_sim,
        p_sim,
        network_dt=config.model.network_dt,
        transient_time=config.simulation.transient_time,
        washout_time=config.model.washout_time,
        loop_times=loop_times,
        loop_names=loop_names,
        input_vars=config.model.input_vars,
        param_vars=config.model.param_vars,
        noise_level=config.simulation.noise_level,
        random_seed=config.random_seed + p_idx,
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

# generate and train ESN realisations
ESN_list = [None] * n_ensemble
for e_idx in range(n_ensemble):
    # fix the seeds
    input_seeds = [5 * e_idx, 5 * e_idx + 1, 5 * e_idx + 2]
    reservoir_seeds = [5 * e_idx + 3, 5 * e_idx + 4]
    if n_ensemble == 1:
        input_seeds = [5, 6, 7]
        reservoir_seeds = [8, 9]

    # expand the ESN dict with the fixed seeds
    ESN_dict["input_seeds"] = input_seeds
    ESN_dict["reservoir_seeds"] = reservoir_seeds

    # create an ESN
    print(f"Creating ESN {e_idx+1}/{n_ensemble}.", flush=True)
    my_ESN = post.create_ESN(
        ESN_dict, "standard", hyp_param_names, hyp_param_scales, hyp_params
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
    ESN_list[e_idx] = my_ESN

# Predict on the test dataset
test_transient_time = config.simulation.transient_time
test_washout_time = config.model.washout_time

eVars = my_sys.get_eVar()
plt_idx = [eVars.z]

fig = plt.figure(figsize=figure_size, constrained_layout=True)
# subfigs = fig.subfigures(1, 2, width_ratios=[2.0, 0.8])
widths = width_ratios = [2.0, 0.8]
spec = fig.add_gridspec(
    ncols=2, nrows=len(plt_idx) * len(test_param_list), width_ratios=widths
)

for p_idx, p in enumerate(test_param_list):
    test_loop_times = [short_loop_lt * LT[p_idx], long_loop_lt * LT[p_idx]]
    test_sim_time = max(test_loop_times) + test_transient_time + test_washout_time
    p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}

    regime_str = (
        f'beta = {p_sim["beta"]}, rho = {p_sim["rho"]}, sigma = {p_sim["sigma"]},'
    )
    print("Regime:", regime_str)

    # set up the initial conditions
    y0 = np.zeros((1, DATA["train"]["u_washout"][0].shape[1]))
    y0[0, :] = y_init
    u_washout_auto = np.repeat(y0, [len(DATA["train"]["u_washout"][0])], axis=0)

    my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
        Lorenz63,
        p_sim,
        sim_time=test_sim_time,
        sim_dt=config.simulation.sim_dt,
        integrator=config.simulation.integrator,
        y_init=y_init,
    )
    data = pp.create_dataset_dyn_sys(
        my_sys,
        y_sim,
        t_sim,
        p_sim,
        network_dt=config.model.network_dt,
        transient_time=test_transient_time,
        washout_time=test_washout_time,
        loop_times=test_loop_times,
        loop_names=test_loop_names,
        input_vars=config.model.input_vars,
        param_vars=config.model.param_vars,
        start_idxs=[0, 0],
    )
    Y_PRED_SHORT = [None] * n_ensemble
    Y_PRED_LONG = [None] * n_ensemble
    for e_idx in range(n_ensemble):
        my_ESN = ESN_list[e_idx]
        print(f"Predicting ESN {e_idx+1}/{n_ensemble}.", flush=True)

        _, y_pred_short = my_ESN.closed_loop_with_washout(
            U_washout=data["short"]["u_washout"],
            N_t=len(data["short"]["u"]),
            P_washout=data["short"]["p_washout"],
            P=data["short"]["p"],
        )
        y_pred_short = y_pred_short[1:]
        Y_PRED_SHORT[e_idx] = y_pred_short

        # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
        if same_washout:
            # Predict long-term
            _, y_pred_long = my_ESN.closed_loop_with_washout(
                U_washout=data["long"]["u_washout"],
                N_t=len(data["long"]["u"]),
                P_washout=data["long"]["p_washout"],
                P=data["long"]["p"],
            )
            y_pred_long = y_pred_long[1:]
        else:
            # let evolve for a longer time and then remove washout
            N_transient_test = pp.get_steps(
                test_transient_time, config.model.network_dt
            )
            N_washout = pp.get_steps(config.model.washout_time, config.model.network_dt)
            N_long = N_transient_test + len(data["long"]["u"])

            P_washout = data["long"]["p"][0] * np.ones((N_washout, 1))
            p_long = data["long"]["p"][0] * np.ones((N_long, 1))
            _, y_pred_long = my_ESN.closed_loop_with_washout(
                U_washout=u_washout_auto,
                N_t=N_long,
                P_washout=P_washout,
                P=p_long,
            )
            y_pred_long = y_pred_long[1:]
            y_pred_long = y_pred_long[N_transient_test:, :]

        Y_PRED_LONG[e_idx] = y_pred_long

    pred_short_error = np.array(
        [errors.rel_L2(data["short"]["y"], Y_PRED_SHORT[i]) for i in range(n_ensemble)]
    )
    print("Short term prediction errors: ", pred_short_error)

    best_idx = np.argmin(pred_short_error)
    print("Best idx: ", best_idx)
    best_idx = 0

    # SHORT-TERM AND TIME-ACCURATE PREDICTION
    # Plot short term prediction timeseries of the best of ensemble
    for i in range(len(plt_idx)):
        # ax = subfigs[0].add_subplot(len(plt_idx)*len(test_param_list), 1, p_idx + i + 1)
        # ax = subfigs[0].add_subplot(len(plt_idx)*len(test_param_list), 1, p_idx + i + 1)
        ax = fig.add_subplot(spec[p_idx + i, 0])
        vis.plot_lines(
            (data["short"]["t"] - data["short"]["t"][0]) / LT[p_idx],
            data["short"]["y"][:, plt_idx[i]],
            Y_PRED_SHORT[best_idx][:, plt_idx[i]],
            ylabel=f"${plt_idx[i].name}$",
            linestyle=[true_ls, pred_ls],
            linewidth=[true_lw, pred_lw],
            color=[true_color, pred_color],
        )

        if i < len(plt_idx) - 1 or p_idx < len(test_param_list) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$t$ [LT]")

        ylims = ax.get_ylim()
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.annotate(titles[2 * (p_idx + i)], xy=(0.95, 0.85), xycoords="axes fraction")

        # Plot statistics
        # ax2 = subfigs[1].add_subplot(len(plt_idx)*len(test_param_list), 1, p_idx + i + 1)
        ax2 = fig.add_subplot(spec[p_idx + i, 1])
        vis.plot_statistics_ensemble(
            *[Y_PRED_LONG[e][:, plt_idx[i]] for e in range(n_ensemble)],
            y_base=data["long"]["y"][:, plt_idx[i]],
            linestyle=[true_ls, pred_ls],
            linewidth=[true_lw, pred_lw],
            color=[true_color, pred_color],
            orientation="var_on_y",
        )

        ax2.set_xlim([-0.001, 0.05])
        ax2.xaxis.set_major_locator(MultipleLocator(0.02))
        ax2.set_ylim(ylims)
        ax2.yaxis.set_major_locator(MultipleLocator(20))
        ax2.set_yticklabels([])

        if i == 0 and p_idx == 0:
            ax2.legend(
                ["True", "ESN"],
                ncol=1,
                columnspacing=0.6,
                loc="center left",
                handlelength=1.5,
                handletextpad=0.25,
                # bbox_to_anchor=(1.05, -0.08),
                # frameon=False,
            )
        if i < len(plt_idx) - 1 or p_idx < len(test_param_list) - 1:
            ax2.set_xticklabels([])
        else:
            ax2.set_xlabel("PDF")
        ax2.annotate(
            titles[2 * (p_idx + i) + 1], xy=(0.85, 0.85), xycoords="axes fraction"
        )
        # ax2.annotate(titles[2*(p_idx + i)+1], (-0.1, 1.1), annotation_clip=False)
    # subfigs[0].suptitle(title, x=0.0, y=1.025)
if save_fig:
    fig.savefig(f"paper_chaotic/graphics/figure_{fig_name}.png", bbox_inches="tight")
    fig.savefig(f"paper_chaotic/graphics/figure_{fig_name}.pdf", bbox_inches="tight")
plt.show()
