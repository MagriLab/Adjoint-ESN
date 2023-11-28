import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

import adjoint_esn.rijke_galerkin.sensitivity as sens
import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar


def round_tau(tau, dt):
    div = tau / dt
    return dt * np.round(div)


def gradient_descent(
    start, norm, dt, my_get_J_and_dJdp, learn_rate, bounds=None, max_iter=50, tol=0.001
):
    p = start
    p_hat = p / norm
    hist = {"p": [], "p_hat": [], "J": [], "dJdp": [], "dJdp_hat": []}

    J_prev = 1e5

    for iter in range(max_iter):
        regime_str = f"beta = {p[eParam.beta]}, tau = {p[eParam.tau]}"
        print(f"Iteration: {iter}, Regime: {regime_str}", flush=True)
        regime_hat_str = f"beta = {p_hat[eParam.beta]}, tau = {p_hat[eParam.tau]}"
        print(f"Normalised regime: {regime_hat_str}", flush=True)

        J, dJdp = my_get_J_and_dJdp(p)
        dJdp_hat = norm * dJdp
        print(f"J = {J}, dJdp = {dJdp}", flush=True)
        print(f"Normalised dJdp = {dJdp_hat}", flush=True)

        # history tracing
        hist["p"].append(p)
        hist["p_hat"].append(p_hat)
        hist["J"].append(J)
        hist["dJdp"].append(dJdp)
        hist["dJdp_hat"].append(dJdp_hat)

        diff = learn_rate * dJdp_hat
        J_diff = (J_prev - J) / J_prev

        # update J prev
        J_prev = J

        # check stopping conditions
        if np.linalg.norm(diff) < tol:
            break

        if J_diff < tol:
            break

        # take a step in the normalised direction
        p_hat = p_hat - diff

        # revert the normalisation and check for bounds
        p = norm * p_hat

        # round tau
        p[eParam.tau] = round_tau(p[eParam.tau], dt)

        # check the bounds
        if bounds:
            p = np.array([max(p[i], bounds[i][0]) for i in range(len(p))])
            p = np.array([min(p[i], bounds[i][1]) for i in range(len(p))])

        # update p_hat after checking the bounds
        p_hat = p / norm

    J, dJdp = my_get_J_and_dJdp(p)
    regime_str = f"beta = {p[eParam.beta]}, tau = {p[eParam.tau]}"
    print(f"Optimal parameters, Regime: {regime_str}", flush=True)
    print(f"J = {J}, dJdp = {dJdp}", flush=True)

    # history tracing
    hist["p"].append(p)
    hist["J"].append(J)
    hist["dJdp"].append(dJdp)

    return hist, p


def get_J_and_dJdp(p, my_ESN, u_washout_auto, N, N_transient):
    p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}

    if hasattr(my_ESN, "tau"):
        my_ESN.tau = p[eParam.tau]

    # Wash-out phase to get rid of the effects of reservoir states initialised as zero
    # initialise the reservoir states before washout
    x0_washout = np.zeros(my_ESN.N_reservoir)

    # let the ESN run in open-loop for the wash-out
    # get the initial reservoir to start the actual open/closed-loop,
    # which is the last reservoir state
    p_washout = p_sim["beta"] * np.ones(
        (u_washout_auto.shape[0], 1)
    )  # BEWARE THIS ONLY APPLIES TO RIJKE, CHANGE THIS!!!!
    X_tau = my_ESN.open_loop(x0=x0_washout, U=u_washout_auto, P=p_washout)
    N_long = N_transient + N
    P_grad = p_washout[0] * np.ones((N_long, 1))
    P_grad = np.vstack((p_washout[-my_ESN.N_tau - 1 :, :], P_grad))
    X_pred_grad, Y_pred_grad = my_ESN.closed_loop(
        X_tau[-my_ESN.N_tau - 1 :, :], N_t=N_long, P=P_grad
    )

    # remove transient
    X_tau = X_pred_grad[:N_transient, :]
    X_pred_grad = X_pred_grad[N_transient:, :]
    Y_pred_grad = Y_pred_grad[N_transient:, :]
    P_grad = P_grad[N_transient:, :]

    J = sens.acoustic_energy(Y_pred_grad[1:, :], N_g)
    dJdp = my_ESN.adjoint_sensitivity(X_pred_grad, Y_pred_grad, N, X_tau)
    return J, dJdp


model_path = Path("local_results/rijke/run_20231029_153121")  # rijke with reservoir
data_dir = Path("data")

test_time = 100
test_transient_time = 200
eta_1_init = 1.5

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
print(ESN_dict, flush=True)
[
    print(f"{hyp_param_name}: {hyp_param}")
    for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params)
]

# generate and train ESN realisations
# fix the seeds
input_seeds = [20, 21, 22]
reservoir_seeds = [23, 24]

# expand the ESN dict with the fixed seeds
ESN_dict["input_seeds"] = input_seeds
ESN_dict["reservoir_seeds"] = reservoir_seeds

# create an ESN
print(f"Creating ESN.", flush=True)
my_ESN = post.create_ESN(
    ESN_dict, config.model.type, hyp_param_names, hyp_param_scales, hyp_params
)
print("Training ESN.", flush=True)
my_ESN.train(
    DATA["train"]["u_washout"],
    DATA["train"]["u"],
    DATA["train"]["y"],
    P_washout=DATA["train"]["p_washout"],
    P_train=DATA["train"]["p"],
    train_idx_list=train_idx_list,
)

y0 = np.zeros((1, DATA["train"]["u_washout"][0].shape[1]))
y0[0, 0] = eta_1_init
u_washout_auto = np.repeat(y0, [len(DATA["train"]["u_washout"][0])], axis=0)
transient_steps = pp.get_steps(transient_time, network_dt)
test_steps = pp.get_steps(test_time, network_dt)

p0 = [None] * 2
p0[eParam.beta] = 4.0
p0[eParam.tau] = 0.25

bounds = ((0.5, 5.5), (0.05, 0.35))
p_mean = np.mean(bounds, axis=1)
my_get_J_and_dJdp = partial(
    get_J_and_dJdp,
    my_ESN=my_ESN,
    u_washout_auto=u_washout_auto,
    N=test_steps,
    N_transient=transient_steps,
)

hist, opt_p = gradient_descent(
    start=np.array(p0),
    norm=np.array(p_mean),
    dt=network_dt,
    my_get_J_and_dJdp=my_get_J_and_dJdp,
    bounds=bounds,
    learn_rate=np.array([0.01, 0.01]),
    max_iter=30,
)

optimization_results = {
    "hist": hist,
    "optimal_parameters": opt_p,
    "input_seeds": input_seeds,
    "reservoir_seeds": reservoir_seeds,
    "eta_1_init": eta_1_init,
}

print(f"Saving results to {model_path}.", flush=True)
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
pp.pickle_file(
    model_path / f"optimization_results_{dt_string}.pickle", optimization_results
)
