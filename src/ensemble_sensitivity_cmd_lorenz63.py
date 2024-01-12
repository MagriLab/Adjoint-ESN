import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import dynamical_systems_sensitivity as sens
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63
from adjoint_esn.utils.enums import eParam, get_eVar


def objective_fun(u):
    return np.mean(u[:, 2])


def dobjective_fun(u):
    dobj = np.zeros(len(u))
    dobj[2] = 1
    return dobj


def main(args):
    model_path = Path(
        "local_results/lorenz63/run_20240105_140530"
    )  # rijke with reservoir
    data_dir = Path("data")

    test_sim_time = 50
    n_loops = 10
    test_loop_times = [test_sim_time]
    test_loop_time_arr = [0.5]
    test_transient_time = 20.0
    n_ensemble = 3

    if len(args.beta) == 3:
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    elif len(args.beta) == 1:
        beta_list = [args.beta[0]]

    if len(args.rho) == 3:
        rho_list = np.arange(args.rho[0], args.rho[1], args.rho[2])
    elif len(args.rho) == 1:
        rho_list = [args.rho[0]]

    if len(args.sigma) == 3:
        sigma_list = np.arange(args.sigma[0], args.sigma[1], args.sigma[2])
    elif len(args.sigma) == 1:
        sigma_list = [args.sigma[0]]

    # Create the mesh to get the data from
    eParam = Lorenz63.get_eParamVar()
    param_mesh_input = [None] * 3
    param_mesh_input[eParam.beta] = beta_list
    param_mesh_input[eParam.rho] = rho_list
    param_mesh_input[eParam.sigma] = sigma_list
    p_list = pp.make_param_mesh(param_mesh_input)

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

    finite_difference_method = "central"
    methods = ["adjoint"]
    dJdp = {"true": {}, "esn": {}}
    for method_name in methods:
        dJdp["true"][method_name] = np.zeros(
            (len(p_list), my_sys.N_param, n_loops, len(test_loop_time_arr))
        )
        dJdp["esn"][method_name] = np.zeros(
            (n_ensemble, len(p_list), len(param_vars), n_loops, len(test_loop_time_arr))
        )

    J = {
        "true": np.zeros((len(p_list), n_loops, len(test_loop_time_arr))),
        "esn": np.zeros((n_ensemble, len(p_list), n_loops, len(test_loop_time_arr))),
    }

    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}

        regime_str = (
            f'beta = {p_sim["beta"]}, rho = {p_sim["rho"]}, sigma = {p_sim["sigma"]},'
        )
        print("Regime:", regime_str)

        # set up the initial conditions
        y0 = np.zeros((1, DATA["train"]["u_washout"][0].shape[1]))
        y0[0, :] = args.y_init
        u_washout_auto = np.repeat(y0, [len(DATA["train"]["u_washout"][0])], axis=0)

        my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
            Lorenz63,
            p_sim,
            sim_time=test_sim_time,
            sim_dt=config.simulation.sim_dt,
            integrator=config.simulation.integrator,
            y_init=args.y_init,
        )
        data = pp.create_dataset_dyn_sys(
            my_sys,
            y_sim,
            t_sim,
            p_sim,
            network_dt=config.model.network_dt,
            transient_time=test_transient_time,
            washout_time=config.model.washout_time,
            loop_times=test_loop_times,
            input_vars=config.model.input_vars,
            param_vars=config.model.param_vars,
        )
        print("Running true.")
        for loop_time_idx, test_loop_time in enumerate(test_loop_time_arr):
            # print("Loop time:", test_loop_time)
            N_loop = pp.get_steps(test_loop_time, config.model.network_dt)

            for loop_idx in range(n_loops):
                print(f"Loop {loop_idx}")
                # split the simulation data
                t_bar = data["loop_0"]["t"][
                    loop_idx * N_loop : (loop_idx + 1) * N_loop + 1
                ]
                t_bar = t_bar - t_bar[0]
                y_bar = data["loop_0"]["u"][
                    loop_idx * N_loop : (loop_idx + 1) * N_loop + 1
                ]

                for method_name in methods:
                    if method_name == "direct":
                        dJdp["true"][method_name][
                            p_idx, :, loop_idx, loop_time_idx
                        ] = sens.true_direct_sensitivity(
                            my_sys,
                            t_bar,
                            y_bar,
                            dobjective_fun,
                            integrator=config.simulation.integrator,
                        )
                    elif method_name == "adjoint":
                        dJdp["true"][method_name][
                            p_idx, :, loop_idx, loop_time_idx
                        ] = sens.true_adjoint_sensitivity(
                            my_sys,
                            t_bar,
                            y_bar,
                            dobjective_fun,
                            integrator=config.simulation.integrator,
                        )
                    elif method_name == "numerical":
                        dJdp["true"][method_name][
                            p_idx, :, loop_idx, loop_time_idx
                        ] = sens.true_finite_difference_sensitivity(
                            my_sys,
                            t_bar,
                            y_bar,
                            h=1e-5,
                            objective_fun=objective_fun,
                            method=finite_difference_method,
                            integrator=config.simulation.integrator,
                        )
                    # print(
                    #     f'True dJ/dp, {method_name} = {dJdp["true"][method_name][p_idx,:,loop_idx,loop_time_idx]}',
                    #     flush=True,
                    # )

                J["true"][p_idx, loop_idx, loop_time_idx] = objective_fun(y_bar[1:])
                # print(f'True J = {J["true"][p_idx,loop_idx,loop_time_idx]}', flush=True)

        for esn_idx in range(n_ensemble):
            print(f"Running ESN {esn_idx}.")
            my_ESN = ESN_list[esn_idx]
            # Wash-out phase to get rid of the effects of reservoir states initialised as zero
            # initialise the reservoir states before washout
            N = len(data["loop_0"]["u"])

            N_transient_test = pp.get_steps(
                test_transient_time, config.model.network_dt
            )
            if args.same_washout == True:
                # predict on the whole timeseries
                X_pred, Y_pred = my_ESN.closed_loop_with_washout(
                    U_washout=data["loop_0"]["u_washout"],
                    N_t=N,
                    P_washout=data["loop_0"]["p_washout"],
                    P=data["loop_0"]["p"],
                )
            else:
                # let evolve for a longer time and then remove washout
                N_long = N_transient_test + N
                P_long = data["loop_0"]["p"][0] * np.ones((N_long, 1))
                X_pred, Y_pred = my_ESN.closed_loop_with_washout(
                    U_washout=u_washout_auto,
                    N_t=N_long,
                    P_washout=data["loop_0"]["p_washout"],
                    P=P_long,
                )

                # remove transient
                X_pred = X_pred[N_transient_test:, :]
                Y_pred = Y_pred[N_transient_test:, :]

            for loop_time_idx, test_loop_time in enumerate(test_loop_time_arr):
                print("Loop time:", test_loop_time)
                N_loop = pp.get_steps(test_loop_time, config.model.network_dt)
                for loop_idx in range(n_loops):
                    # print(f"Loop {loop_idx}")
                    # split the prediction data
                    t_loop = data["loop_0"]["t"][
                        loop_idx * N_loop : (loop_idx + 1) * N_loop + 1
                    ]
                    t_loop = t_loop - t_loop[0]
                    x_pred_loop = X_pred[
                        loop_idx * N_loop : (loop_idx + 1) * N_loop + 1
                    ]
                    y_pred_loop = Y_pred[
                        loop_idx * N_loop : (loop_idx + 1) * N_loop + 1
                    ]
                    p_loop = data["loop_0"]["p"][
                        loop_idx * N_loop : (loop_idx + 1) * N_loop + 1
                    ]
                    for method_name in methods:
                        if method_name == "direct":
                            dJdp["esn"]["direct"][
                                esn_idx, p_idx, :, loop_idx, loop_time_idx
                            ] = my_ESN.direct_sensitivity(
                                x_pred_loop,
                                y_pred_loop,
                                N_loop,
                                dJdy_fun=dobjective_fun,
                            )
                        elif method_name == "adjoint":
                            dJdp["esn"]["adjoint"][
                                esn_idx, p_idx, :, loop_idx, loop_time_idx
                            ] = my_ESN.adjoint_sensitivity(
                                x_pred_loop,
                                y_pred_loop,
                                N_loop,
                                dJdy_fun=dobjective_fun,
                            )
                        elif method_name == "numerical":
                            dJdp["esn"]["finite"][
                                esn_idx, p_idx, :, loop_idx, loop_time_idx
                            ] = my_ESN.finite_difference_sensitivity(
                                X=x_pred_loop,
                                Y=y_pred_loop,
                                P=p_loop,
                                N=N_loop,
                                h=1e-5,
                                method=finite_difference_method,
                                J_fun=objective_fun,
                            )
                        # print(
                        # f'ESN {esn_idx} dJ/dp, {method_name} = {dJdp["esn"][method_name][esn_idx,p_idx,:,loop_idx,loop_time_idx]}',
                        # flush=True,
                        # )
                    J["esn"][esn_idx, p_idx, loop_idx, loop_time_idx] = objective_fun(
                        y_pred_loop[1:]
                    )
                    # print(f'ESN {esn_idx} J = {J["esn"][esn_idx,p_idx,loop_idx,loop_time_idx]}', flush = True)

    sensitivity_results = {
        "dJdp": dJdp,
        "J": J,
        "beta_list": beta_list,
        "rho_list": rho_list,
        "sigma_list": sigma_list,
        "same_washout": args.same_washout,
        "y_init": args.y_init,
    }

    print(f"Saving results to {model_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(
        model_path / f"sensitivity_results_{dt_string}.pickle", sensitivity_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", nargs="+", type=float)
    parser.add_argument("--rho", nargs="+", type=float)
    parser.add_argument("--sigma", nargs="+", type=float)
    parser.add_argument("--same_washout", default=False, action="store_true")
    parser.add_argument("--y_init", nargs="+", type=float)
    parsed_args = parser.parse_args()
    main(parsed_args)
