import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import adjoint_esn.utils.postprocessing as post
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar


def acoustic_energy(y, N_g):
    return 1 / 4 * np.mean(np.sum(y[:, : 2 * N_g] ** 2, axis=1))


def energy_decreased(y, N_g, percent_threshold=5):
    half_steps = int(np.round(len(y) / 2))
    e1 = acoustic_energy(y[:half_steps], N_g)
    e2 = acoustic_energy(y[half_steps:], N_g)
    return e1 - e2 > (percent_threshold / 100) * e1 or e2 < 5e-4


rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 12})
rc("text", usetex=True)
save_fig = True
model_path = Path("local_results/rijke/run_20231029_153121")  # rijke with reservoir
data_dir = Path("data")

test_sim_time = 304
test_loop_times = [100]
test_transient_time = 200
n_ensemble = 10
eta_1_init = 1.5

beta_list = np.arange(0.2, 1.7, 0.05)
tau_list = np.arange(0.05, 0.35, 0.01)

percent_threshold = 5

config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]

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

# generate and train ESN realisations
ESN_list = [None] * n_ensemble
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
    ESN_list[e_idx] = my_ESN

p_energy_decreased = partial(
    energy_decreased, N_g=N_g, percent_threshold=percent_threshold
)
bifn_points = {
    "true": np.zeros(len(tau_list)),
    "esn": np.zeros((n_ensemble, len(tau_list))),
}

for tau_idx, tau in enumerate(tau_list):
    for beta_idx, beta in enumerate(beta_list):
        p_sim = {"beta": beta, "tau": tau}
        y_init = np.zeros(2 * N_g + 10)
        y_init[0] = eta_1_init
        y_sim, t_sim = pp.load_data(
            beta=p_sim["beta"],
            tau=p_sim["tau"],
            x_f=0.2,
            N_g=N_g,
            sim_time=test_sim_time,
            sim_dt=sim_dt,
            data_dir=data_dir,
            y_init=y_init,
        )

        data = pp.create_dataset(
            y_sim,
            t_sim,
            p_sim,
            network_dt=network_dt,
            transient_time=test_transient_time,
            washout_time=washout_time,
            loop_times=test_loop_times,
            input_vars=input_vars,
            output_vars=output_vars,
            param_vars=param_vars,
            N_g=N_g,
            u_f_order=u_f_order,
        )

        loop_name = list(data.keys())[0]
        if p_energy_decreased(data[loop_name]["y"]):
            bifn_points["true"][tau_idx] = beta
            print(f'True bifn pt: beta = {p_sim["beta"]}, tau = {p_sim["tau"]}')
            true_break_flag = False
        else:
            true_break_flag = True

        for e_idx, my_ESN in enumerate(ESN_list):
            if hasattr(my_ESN, "tau"):
                my_ESN.tau = p_sim["tau"]

            y_init_network = np.zeros((1, data[loop_name]["u_washout"].shape[1]))
            y_init_network[0, 0] = eta_1_init
            u_washout_auto = np.repeat(
                y_init_network, [len(data[loop_name]["u_washout"])], axis=0
            )
            transient_steps = pp.get_steps(test_transient_time, network_dt)
            # add the transient time that will be discarded later
            N_t_long = transient_steps + len(data[loop_name]["u"])
            p_long0 = np.zeros((1, data[loop_name]["p"].shape[1]))
            p_long0[0] = data[loop_name]["p"][0]
            p_long = np.repeat(p_long0, [N_t_long], axis=0)

            _, y_pred = my_ESN.closed_loop_with_washout(
                U_washout=u_washout_auto,
                N_t=N_t_long,
                P_washout=data[loop_name]["p_washout"],
                P=p_long,
            )
            y_pred = y_pred[1:]
            y_pred = y_pred[transient_steps:, :]

            if p_energy_decreased(y_pred):
                bifn_points["esn"][e_idx, tau_idx] = beta
                print(
                    f'ESN {e_idx} pred bifn pt: beta = {p_sim["beta"]}, tau = {p_sim["tau"]}'
                )
                pred_break_flag = False
            else:
                pred_break_flag = True

        if true_break_flag and pred_break_flag:
            break

bifn_point_results = {
    "bifn_points": bifn_points,
    "eta_1_init": eta_1_init,
    "beta_list": beta_list,
    "tau_list": tau_list,
}

print(f"Saving results to {model_path}.", flush=True)
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
pp.pickle_file(
    model_path / f"bifn_point_results_{dt_string}.pickle", bifn_point_results
)
