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
import adjoint_esn.utils.solve_ode as solve_ode
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.utils import dynamical_systems_sensitivity as sens
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63

traj_time = 1100  # length of each trajectory

# regime which we apply perturbation to
eParam = Lorenz63.get_eParamVar()
test_p = np.zeros((3))
# test_p[eParam.beta] = 8/3
# test_p[eParam.rho] = 28.0
# test_p[eParam.sigma] = 10.0

test_p[eParam.beta] = 1.75
test_p[eParam.rho] = 52.0
test_p[eParam.sigma] = 13.0

# list of initial conditions
y_init_list = [
    [-2.4, -3.7, 14.98],
    [8.0, -2.0, 36.05],
    [1.0, 1.0, 1.0],
    [7.43, 10.02, 29.62],
]

# amount of perturbation
h = 5e-1

# plotting options
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)
fig_size = (15, 12)
true_ls = "-"
true_h_ls = "-"
pred_ls = "--"
pred_h_ls = "--"
true_c = "silver"
true_h_c = "purple"
pred_c = "red"
pred_h_c = "limegreen"
true_lw = 4
true_h_lw = 2.0
pred_lw = 2.0
pred_h_lw = 1.5
plot_time = 3
loop_time_arr = np.arange(0.05, plot_time + 0.05, 0.05)

# load model
# ESN 1
# model_path = Path(
#         "local_results/lorenz63/run_20240105_140530"
#     )
# ESN 3
# model_path = Path(
#         "local_results/lorenz63/run_20240206_181642"
#     )
# ESN 4
model_path = Path("local_results/lorenz63/run_20240208_121804")
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
# expand the ESN dict with the fixed seeds
ESN_dict["input_seeds"] = [5, 6, 7]
ESN_dict["reservoir_seeds"] = [8, 9]
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


def objective_fun(u):
    return np.mean(u[:, 2])


def dobjective_fun(u):
    dobj = np.zeros(len(u))
    dobj[2] = 1
    return dobj


# test on multiple trajectories
test_p_sim = {
    "beta": test_p[eParam.beta],
    "rho": test_p[eParam.rho],
    "sigma": test_p[eParam.sigma],
}
test_sim_time = config.simulation.transient_time + config.model.washout_time + traj_time
plot_steps = pp.get_steps(plot_time, config.model.network_dt)

for y_init_idx, y_init in enumerate(y_init_list):
    fig1 = plt.figure(figsize=fig_size, constrained_layout=True)
    # simulate original system
    my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
        Lorenz63,
        test_p_sim,
        sim_time=test_sim_time,
        sim_dt=config.simulation.sim_dt,
        integrator=config.simulation.integrator,
        y_init=y_init,
    )
    eVars = my_sys.get_eVar()
    data = pp.create_dataset_dyn_sys(
        my_sys,
        y_sim,
        t_sim,
        test_p_sim,
        network_dt=config.model.network_dt,
        transient_time=config.simulation.transient_time,
        washout_time=config.model.washout_time,
        loop_times=[traj_time],
        input_vars=config.model.input_vars,
        param_vars=config.model.param_vars,
    )
    t_bar = data["loop_0"]["t"] - data["loop_0"]["t"][0]
    y_bar = data["loop_0"]["u"]

    # predict with ESN
    N = len(data["loop_0"]["u"])
    x_pred, y_pred = my_ESN.closed_loop_with_washout(
        U_washout=data["loop_0"]["u_washout"],
        N_t=N - 1,
        P_washout=data["loop_0"]["p_washout"],
        P=data["loop_0"]["p"],
    )

    # perturb and predict
    for param_idx in range(my_sys.N_param):
        param_name = eParam(param_idx).name
        current_param = getattr(my_sys, param_name)

        # perturb from the right
        setattr(my_sys, param_name, current_param + h)

        y_bar_right = solve_ode.integrate(
            my_sys.ode, y_bar[0], t_bar, integrator=config.simulation.integrator
        )
        # set the current parameter back
        setattr(my_sys, param_name, current_param)

        # perturbed by h
        P_right = data["loop_0"]["p"].copy()
        P_right[:, param_idx] += h
        _, y_pred_right = my_ESN.closed_loop(x_pred[0, :], N - 1, P_right)

        fig1.add_subplot(4, 3, param_idx + 1)
        vis.plot_lines(
            t_bar[:plot_steps],
            y_bar[:plot_steps, eVars.z],
            y_bar_right[:plot_steps, eVars.z],
            y_pred[:plot_steps, eVars.z],
            y_pred_right[:plot_steps, eVars.z],
            xlabel="$t$",
            ylabel="$z$",
            title=f"$\\{param_name} + {h}$",
            linestyle=[true_ls, true_h_ls, pred_ls, pred_h_ls],
            color=[true_c, true_h_c, pred_c, pred_h_c],
            linewidth=[true_lw, true_h_lw, pred_lw, pred_h_lw],
        )

        fig1.add_subplot(4, 3, param_idx + 4)
        vis.plot_phase_space(
            y_bar[:plot_steps],
            y_bar_right[:plot_steps],
            y_pred[:plot_steps],
            y_pred_right[:plot_steps],
            idx_pair=[eVars.x, eVars.z],
            xlabel="$x$",
            ylabel="$z$",
            linestyle=[true_ls, true_h_ls, pred_ls, pred_h_ls],
            color=[true_c, true_h_c, pred_c, pred_h_c],
            linewidth=[true_lw, true_h_lw, pred_lw, pred_h_lw],
        )
        plt.plot(
            y_bar[0, eVars.x], y_bar[0, eVars.z], "o", color="tab:blue", markersize=8
        )

        fig1.add_subplot(4, 3, param_idx + 7)
        vis.plot_statistics(
            y_bar[:, eVars.z],
            y_bar_right[:, eVars.z],
            y_pred[:, eVars.z],
            y_pred_right[:, eVars.z],
            xlabel="$z$",
            ylabel="PDF",
            linestyle=[true_ls, true_h_ls, pred_ls, pred_h_ls],
            color=[true_c, true_h_c, pred_c, pred_h_c],
            linewidth=[true_lw, true_h_lw, pred_lw, pred_h_lw],
        )

        legend_str = ["True", "True + h", "Pred", "Pred + h"]
        plt.figlegend(
            legend_str,
            loc="upper center",
            ncol=len(legend_str),
            bbox_to_anchor=(0.5, 1.1),
        )

    # integrate adjoint sensitivity of true and predicted
    dJdp_true = np.zeros((len(loop_time_arr), 3))
    dJdp_esn = np.zeros((len(loop_time_arr), 3))
    for loop_time_idx, loop_time in enumerate(loop_time_arr):
        N_loop_sim = pp.get_steps(loop_time, config.simulation.sim_dt)
        N_loop_network = pp.get_steps(loop_time, config.model.network_dt)
        t_bar_loop = data["loop_0"]["t"][0 : N_loop_sim + 1]
        t_bar_loop = t_bar_loop - t_bar_loop[0]
        y_bar_loop = data["loop_0"]["u"][0 : N_loop_sim + 1]
        dJdp_true[loop_time_idx, :] = sens.true_adjoint_sensitivity(
            my_sys,
            t_bar_loop,
            y_bar_loop,
            dobjective_fun,
            integrator=config.simulation.integrator,
        )

        x_pred_loop = x_pred[0 : N_loop_network + 1]
        y_pred_loop = y_pred[0 : N_loop_network + 1]
        p_loop = data["loop_0"]["p"][0 : N_loop_network + 1]
        dJdp_esn[loop_time_idx, :] = my_ESN.adjoint_sensitivity(
            x_pred_loop, y_pred_loop, N_loop_network, dJdy_fun=dobjective_fun
        )
    for param_idx in range(my_sys.N_param):
        param_name = eParam(param_idx).name
        fig1.add_subplot(4, 3, param_idx + 10)
        vis.plot_lines(
            loop_time_arr,
            dJdp_true[:, param_idx],
            dJdp_esn[:, param_idx],
            xlabel="$t$",
            ylabel=f"$d\\bar{{z}}/d\\{param_name}$",
            linestyle=[true_ls, pred_ls],
            color=[true_c, pred_c],
            linewidth=[true_lw, pred_lw],
        )
    fig1.savefig(
        f"paper_chaotic/graphics/figure_response_2_init_{y_init_idx}_ESN_4.png",
        bbox_inches="tight",
    )
plt.show()
