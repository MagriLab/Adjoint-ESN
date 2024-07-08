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

    # number of loops
    n_loops = 1

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

    # determine index of the training parameter that we want to test on
    p_test = [None] * 2
    p_test[eParam.beta] = args.beta
    p_test[eParam.tau] = args.tau
    test_idx = np.where(np.sum(np.equal(train_param_list, p_test), axis=1) == 2)[0][0]

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

    # get data
    test_sim_time = transient_time + np.sum(args.loop_time) + washout_time * n_loops
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
        loop_times=[args.loop_time],
        input_vars=input_vars,
        output_vars=output_vars,
        param_vars=param_vars,
        N_g=N_g,
        u_f_order=u_f_order,
        tau=p_sim["tau"],
    )

    Y_ARR = [data[loop_name]["y"] for loop_name in data.keys()]
    Y_PRED_ARR = [[None] * n_ensemble for _ in range(n_loops)]

    for e_idx in range(n_ensemble):
        print(f"Testing ESN {e_idx+1}/{n_ensemble}")
        my_ESN = ESN_list[e_idx]
        if hasattr(my_ESN, "tau"):
            my_ESN.tau = p_sim["tau"]
        for loop_idx, loop_name in enumerate(data.keys()):
            _, y_pred = my_ESN.closed_loop_with_washout(
                U_washout=DATA["train"]["u_washout"][test_idx],
                N_t=len(data[loop_name]["u"]),
                P_washout=data[loop_name]["p_washout"],
                P=data[loop_name]["p"],
            )
            y_pred = y_pred[1:]
            error = error_measure(
                data[loop_name]["y"][:, : 2 * N_g], y_pred[:, : 2 * N_g]
            )
            print(f"Loop {loop_name[-1]} error: {error}")
            Y_PRED_ARR[loop_idx][e_idx] = y_pred

    timeseries_results = {
        "y_train": DATA["train"]["y"][test_idx],
        "y_true": Y_ARR,
        "y_pred": Y_PRED_ARR,
        "beta": args.beta,
        "tau": args.tau,
        "loop_times": args.loop_time,
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
        / f"timeseries_beta_{beta_name}_tau_{tau_name}_results_{dt_string}.pickle",
        timeseries_results,
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
    parsed_args = parser.parse_args()
    main(parsed_args)
