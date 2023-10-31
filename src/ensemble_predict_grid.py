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
    # options
    test_time = 200
    test_loop_time = 20
    n_folds = 5
    seed = 0  # random seed for fold start idxs

    beta_list = np.arange(0.5,7.5,0.25)
    tau_list = np.arange(0.05, 0.35, 0.01)

    # create an ensemble of ESNs to make predictions
    n_ensemble = 5

    # make a list of test parameters
    test_param_list = pp.make_param_mesh([beta_list, tau_list])

    # error measure
    error_measure = errors.rel_L2

    # load experiment config and results
    experiment_path = Path(args.experiment_path)
    config = post.load_config(experiment_path)
    results = pp.unpickle_file(experiment_path / "results.pickle")[0]

    # load the data for training
    data_dir = Path(args.data_dir)

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
    N_train = DATA["train"]["u"][0].shape[0]
    test_sim_time = transient_time + train_time + washout_time + test_time

    N_test = pp.get_steps(test_time, network_dt)
    N_washout = pp.get_steps(washout_time, network_dt)
    N_test_loop = pp.get_steps(test_loop_time, network_dt)

    rnd = np.random.RandomState(seed=seed)
    start_idxs = (
        N_train
        + N_washout
        + rnd.randint(low=0, high=N_test - (N_washout + N_test_loop), size=(n_folds,))
    )
    test_loop_times = [test_loop_time] * n_folds

    print("Loading model.")
    # get the properties of the best ESN from the results
    (
        ESN_dict,
        hyp_param_names,
        hyp_param_scales,
        hyp_params,
    ) = post.get_ESN_properties_from_results(config, results, dim, top_idx=args.top_idx)
    ESN_dict["verbose"] = False
    print(ESN_dict)
    [
        print(f"{hyp_param_name}: {hyp_param}")
        for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params)
    ]

    # create matrix to save error
    error_mat = np.zeros((len(test_param_list), n_folds, n_ensemble))
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

        for p_idx, p in enumerate(test_param_list):
            p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
            regime_str = f'beta = {p_sim["beta"]}, tau = {p_sim["tau"]},'
            print(f"Testing ESN on {regime_str}")
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
                input_vars=input_vars,
                output_vars=output_vars,
                param_vars=param_vars,
                N_g=N_g,
                u_f_order=u_f_order,
                start_idxs=start_idxs,
            )

            if hasattr(my_ESN, "tau"):
                my_ESN.tau = p_sim["tau"]
            for loop_idx, loop_name in enumerate(data.keys()):
                _, y_pred = my_ESN.closed_loop_with_washout(
                    U_washout=data[loop_name]["u_washout"],
                    N_t=len(data[loop_name]["u"]),
                    P_washout=data[loop_name]["p_washout"],
                    P=data[loop_name]["p"],
                )
                y_pred = y_pred[1:]
                error = error_measure(data[loop_name]["y"][:,:2*N_g], y_pred[:,:2*N_g]) # do this propery with the enums
                print(f"Loop {loop_name} error: {error}")
                error_mat[p_idx, loop_idx, e_idx] = error

    error_results = {
        "error": error_mat,
        "test_time": test_time,
        "fold_time": test_loop_time,
        "seed": seed,
        "test_parameters": test_param_list,
        "beta_list": beta_list,
        "tau_list": tau_list,
        "top_idx": args.top_idx
    }

    print(f"Saving results to {experiment_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(experiment_path / f"error_results_{dt_string}.pickle", error_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--top_idx",type=int,default=0)
    parsed_args = parser.parse_args()
    main(parsed_args)
