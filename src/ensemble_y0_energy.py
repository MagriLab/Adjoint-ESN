import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime
from pathlib import Path

import numpy as np

import adjoint_esn.rijke_galerkin.sensitivity as sens
import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar

model_path = Path("local_results/rijke/run_20231029_153121")  # rijke with reservoir
data_dir = Path("data")

test_time = 100
test_transient_time = 200
n_ensemble = 1
eta_1_init = 1.5

beta_list = np.arange(0.5, 5.5, 0.1)
tau_list = np.arange(0.05, 0.35, 0.01)
p_list = pp.make_param_mesh([beta_list, tau_list])

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


J = np.zeros((n_ensemble, len(p_list)))

y0 = np.zeros((1, DATA["train"]["u_washout"][0].shape[1]))
y0[0, 0] = eta_1_init
u_washout_auto = np.repeat(y0, [len(DATA["train"]["u_washout"][0])], axis=0)
transient_steps = pp.get_steps(transient_time, network_dt)
test_steps = pp.get_steps(test_time, network_dt)
# add the transient time that will be discarded later
N_t_long = transient_steps + test_steps

for esn_idx in range(n_ensemble):
    my_ESN = ESN_list[esn_idx]
    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}

        regime_str = f'beta = {p_sim["beta"]}, tau = {p_sim["tau"]}'
        print("Regime:", regime_str)
        if hasattr(my_ESN, "tau"):
            my_ESN.tau = p[eParam.tau]

        # Wash-out phase to get rid of the effects of reservoir states initialised as zero
        # initialise the reservoir states before washout
        x0_washout = np.zeros(my_ESN.N_reservoir)

        # let the ESN run in open-loop for the wash-out
        # get the initial reservoir to start the actual open/closed-loop,
        # which is the last reservoir state
        p_washout = p_sim["beta"] * np.ones((u_washout_auto.shape[0], 1))
        X_tau = my_ESN.open_loop(x0=x0_washout, U=u_washout_auto, P=p_washout)
        P_grad = p_washout[0] * np.ones((N_t_long, 1))
        P_grad = np.vstack((p_washout[-my_ESN.N_tau - 1 :, :], P_grad))
        X_pred_grad, Y_pred_grad = my_ESN.closed_loop(
            X_tau[-my_ESN.N_tau - 1 :, :], N_t=N_t_long, P=P_grad
        )

        # remove transient
        X_tau = X_pred_grad[:transient_steps, :]
        X_pred_grad = X_pred_grad[transient_steps:, :]
        Y_pred_grad = Y_pred_grad[transient_steps:, :]
        P_grad = P_grad[transient_steps:, :]

        J[e_idx, p_idx] = sens.acoustic_energy(Y_pred_grad[1:, :], N_g)

        print(f"ESN {esn_idx} J = {J[esn_idx, p_idx]}")

energy_results = {
    "J": J,
    "beta_list": beta_list,
    "tau_list": tau_list,
}
print(f"Saving results to {model_path}.", flush=True)
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
pp.pickle_file(model_path / f"energy_results_{dt_string}.pickle", energy_results)
