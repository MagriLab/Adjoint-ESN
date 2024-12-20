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
from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)
plt.style.use("dark_background")

save_fig = False
same_washout = True
model_path = Path("local_results/lorenz63/run_20240208_121804")

fig_name = "lorenz_phase"

# Create the mesh to get the data from
eParam = Lorenz63.get_eParamVar()
test_param_list = np.zeros((3, 3))

which_param = "rho"
if which_param == "beta":
    test_param_list[:, eParam.beta] = np.linspace(2.0, 3.0, len(test_param_list))
    test_param_list[:, eParam.rho] = np.array([28.0] * len(test_param_list))
    test_param_list[:, eParam.sigma] = np.array([10.0] * len(test_param_list))
elif which_param == "rho":
    test_param_list[:, eParam.beta] = np.array([8 / 3] * len(test_param_list))
    test_param_list[:, eParam.rho] = np.linspace(25.0, 55.0, len(test_param_list))
    test_param_list[:, eParam.sigma] = np.array([10.0] * len(test_param_list))
elif which_param == "sigma":
    test_param_list[:, eParam.beta] = np.array([8 / 3] * len(test_param_list))
    test_param_list[:, eParam.rho] = np.array([28.0] * len(test_param_list))
    test_param_list[:, eParam.sigma] = np.linspace(6.0, 18.0, len(test_param_list))
# LT = [1.1, 0.8]
LT = [1] * len(test_param_list)
titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

n_ensemble = 1
short_loop_lt = 20
long_loop_lt = 20
test_loop_names = ["short", "long"]
y_init = [-2.4, -3.7, 14.98]

figure_size = (12, 5)
config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]

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
PLOT_TRUE_DATA = []
PLOT_PRED_DATA = []
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

    PLOT_TRUE_DATA.append(data["long"]["y"])
    PLOT_PRED_DATA.append(Y_PRED_LONG[best_idx])

cmap = cm.create_custom_colormap(type="discrete")
true_color = cmap(0)
pred_color = cmap(2)

# true_color = "#196B24"
# pred_color = "#A02B93"
cmap2 = cm.create_custom_colormap(
    type="continuous", colors=[true_color, "white"], N=len(test_param_list)
)
cmap3 = cm.create_custom_colormap(
    type="continuous", colors=[pred_color, "white"], N=len(test_param_list)
)
colors = [
    [cmap2(i) for i in range(len(test_param_list))],
    [cmap3(i) for i in range(len(test_param_list))],
]
legend = [
    f"${which_param[0]}$ = {test_param_list[i, eParam[which_param]]}"
    for i in range(len(test_param_list))
]
ani = vis.plot_lorenz63_attractor(
    fig,
    PLOT_TRUE_DATA,
    PLOT_PRED_DATA,
    len(Y_PRED_LONG[best_idx]),
    colors=colors,
    animate=True,
    legend=legend,
)
if save_fig:
    fig.savefig(
        f"local_images/figure_{fig_name}_{which_param}.png",
        bbox_inches="tight",
        dpi=300,
    )
    ani.save(
        f"local_images/figure_{fig_name}_ani_{which_param}.gif",
        writer="pillow",
        fps=10,
    )
    # ani.save(f"local_images/figure_{fig_name}_ani_rho.gif", writer='imagemagick', fps=10, extra_args=['-loop', str(3)])
plt.show()
