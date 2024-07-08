import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from adjoint_esn.utils import errors
from adjoint_esn.utils import postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam


def main(args):
    """Make predictions on random folds and save the error for an ensemble of ESNs"""
    # load model
    model_path = Path(args.model_dir) / f"run_{args.run_name}"
    config = post.load_config(model_path)
    results = pp.unpickle_file(model_path / "results.pickle")[0]

    # set data directory
    data_dir = Path(args.data_dir)

    # number of ensemble of ESNs to make predictions
    n_ensemble = args.n_ensemble_esn

    # error measure
    error_measure = errors.rel_L2

    # options for data creation
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

    # which system parameter is passed to the ESN
    param_vars = config.model.param_vars

    # if using Rijke ESN what is the order of u_f(t-tau) in the inputs,
    # [u_f(t-tau), u_f(t-tau)^2 ..., u_f(t-tau)^(u_f_order)]
    u_f_order = config.model.u_f_order

    # length of training time series
    train_time = config.train.time
    loop_names = ["train"]
    loop_times = [train_time]

    # create or load training data
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

        if "training_random_seeds" in results.keys():
            noise_seed = results["training_random_seeds"][p_idx]
        else:
            noise_seed = random_seed + p_idx

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
            random_seed=noise_seed,
            tau=p_sim["tau"],
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
        print(f"Creating ESN {e_idx+1}/{n_ensemble}.", flush=True)
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

    # get washout data
    test_sim_time = transient_time + washout_time + 1.0
    p_sim = {"beta": args.beta, "tau": args.tau}
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
        loop_times=[1.0],  # we just want to load the washout data
        loop_names=["long"],
        input_vars=input_vars,
        output_vars=output_vars,
        param_vars=param_vars,
        N_g=N_g,
        u_f_order=u_f_order,
        tau=p_sim["tau"],
    )

    Y_PRED_ARR = [None] * n_ensemble

    for e_idx in range(n_ensemble):
        print(f"Testing ESN {e_idx+1}/{n_ensemble}")
        my_ESN = ESN_list[e_idx]
        if hasattr(my_ESN, "tau"):
            my_ESN.tau = p_sim["tau"]

        # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
        N_t = pp.get_steps(args.loop_time, network_dt)
        if args.same_washout:
            p_long0 = np.zeros((1, data["long"]["p"].shape[1]))
            p_long0[0] = data["long"]["p"][0]
            p_long = np.repeat(p_long0, [N_t], axis=0)
            # Predict long-term
            _, y_pred = my_ESN.closed_loop_with_washout(
                U_washout=data["long"]["u_washout"],
                N_t=N_t,
                P_washout=data["long"]["p_washout"],
                P=p_long,
            )
            y_pred = y_pred[1:]
        else:
            y0 = np.zeros((1, data["long"]["u_washout"].shape[1]))
            y0[0, 0] = 1.5
            u_washout_auto = np.repeat(y0, [len(data["long"]["u_washout"])], axis=0)
            transient_steps = pp.get_steps(transient_time, network_dt)
            # add the transient time that will be discarded later
            N_t_long = transient_steps + N_t
            p_long0 = np.zeros((1, data["long"]["p"].shape[1]))
            p_long0[0] = data["long"]["p"][0]
            p_long = np.repeat(p_long0, [N_t_long], axis=0)
            # Predict long-term
            _, y_pred = my_ESN.closed_loop_with_washout(
                U_washout=u_washout_auto,
                N_t=N_t_long,
                P_washout=data["long"]["p_washout"],
                P=p_long,
            )
            y_pred = y_pred[1 + transient_steps :, :]

        Y_PRED_ARR[e_idx] = y_pred

    long_term_results = {
        "y_pred": Y_PRED_ARR,
        "beta": args.beta,
        "tau": args.tau,
        "loop_time": args.loop_time,
        "same_washout": args.same_washout,
    }

    print(f"Saving results to {model_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    # construct data path name
    beta_name = f"{args.beta:.2f}"
    beta_name = beta_name.replace(".", "_")
    tau_name = f"{args.tau:.2f}"
    tau_name = tau_name.replace(".", "_")

    pp.pickle_file(
        model_path
        / f"long_term_beta_{beta_name}_tau_{tau_name}_results_{dt_string}.pickle",
        long_term_results,
    )


# add option not to save the truth
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--loop_time", type=float, default=20)
    parser.add_argument("--n_ensemble_esn", type=int, default=5)
    parser.add_argument("--same_washout", default=False, action="store_true")
    parsed_args = parser.parse_args()
    main(parsed_args)
