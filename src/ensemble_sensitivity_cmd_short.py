import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import adjoint_esn.rijke_galerkin.sensitivity as sens
import adjoint_esn.utils.postprocessing as post
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.rijke_galerkin.solver import Rijke
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar


def main(args):
    model_path = Path(f"local_results/rijke/run_{args.run_name}")
    data_dir = Path("data")

    n_loops = args.n_loops
    if args.loop_times[0] > 0:
        test_loop_time_arr = np.array(args.loop_times)
    elif args.loop_time_cont[0] > 0:
        test_loop_time_arr = np.arange(
            args.loop_time_cont[0], args.loop_time_cont[1], args.loop_time_cont[2]
        )

    n_ensemble = args.n_ensemble_esn
    eta_1_init = args.eta_1_init

    if len(args.beta) == 3:
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    elif len(args.beta) == 1:
        beta_list = [args.beta[0]]

    if len(args.tau) == 3:
        tau_list = np.arange(args.tau[0], args.tau[1], args.tau[2])
    elif len(args.tau) == 1:
        tau_list = [args.tau[0]]

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

    if args.train_noise_level >= 0.0:
        noise_level = args.train_noise_level
    else:
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
        if n_ensemble == 1:
            input_seeds = [20, 21, 22]
            reservoir_seeds = [23, 24]
        else:
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

    finite_difference_method = "central"
    methods = ["adjoint"]
    dJdp = {"true": {}, "esn": {}}
    for method_name in methods:
        dJdp["true"][method_name] = np.zeros(
            (len(p_list), 2, n_loops, len(test_loop_time_arr))
        )
        if config.model.type == "standard":
            dJdp["esn"][method_name] = np.zeros(
                (
                    n_ensemble,
                    len(p_list),
                    len(param_vars),
                    n_loops,
                    len(test_loop_time_arr),
                )
            )
        elif config.model.type == "rijke":
            dJdp["esn"][method_name] = np.zeros(
                (n_ensemble, len(p_list), 2, n_loops, len(test_loop_time_arr))
            )

    J = {
        "true": np.zeros((len(p_list), n_loops, len(test_loop_time_arr))),
        "esn": np.zeros((n_ensemble, len(p_list), n_loops, len(test_loop_time_arr))),
    }

    test_transient_time = config.simulation.transient_time
    # when running with same_washout=True, loops start after the washout
    # when running with same_washout=False, loops start after the transient
    # that's why the initial condition of the first loop changes
    if args.same_washout:
        test_washout_time = config.model.washout_time
    else:
        test_washout_time = 0

    # add fast jacobian condition
    if config.model.input_only_mode == False and config.model.r2_mode == False:
        fast_jac = True
    print("Fast jac condition:", fast_jac)

    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}

        regime_str = f'beta = {p_sim["beta"]}, tau = {p_sim["tau"]}'
        print("Regime:", regime_str)

        test_sim_time = (
            max(test_loop_time_arr) + test_transient_time + test_washout_time
        )
        test_loop_times = [max(test_loop_time_arr)]

        # set up the initial conditions
        y0 = np.zeros((1, DATA["train"]["u_washout"][0].shape[1]))
        y0[0, 0] = eta_1_init
        u_washout_auto = np.repeat(y0, [len(DATA["train"]["u_washout"][0])], axis=0)

        y0_sim = np.zeros(2 * N_g + 10)
        y0_sim[0] = eta_1_init

        y_sim, t_sim = pp.load_data(
            beta=p_sim["beta"],
            tau=p_sim["tau"],
            x_f=0.2,
            N_g=N_g,
            sim_time=test_sim_time,
            sim_dt=sim_dt,
            data_dir=data_dir,
            integrator=integrator,
            y_init=y0_sim,
        )

        data = pp.create_dataset(
            y_sim,
            t_sim,
            p_sim,
            network_dt=network_dt,
            transient_time=test_transient_time,
            washout_time=test_washout_time,
            loop_times=test_loop_times,
            input_vars=input_vars,
            output_vars=output_vars,
            param_vars=param_vars,
            N_g=N_g,
            u_f_order=u_f_order,
            tau=p_sim["tau"],
        )

        my_rijke = Rijke(
            N_g=N_g,
            N_c=10,
            c_1=0.1,
            c_2=0.06,
            beta=p_sim["beta"],
            x_f=0.2,
            tau=p_sim["tau"],
            heat_law="kings_poly",
            damping="modal",
        )
        print("Running true.", flush=True)
        if args.get_adjoint_of == "both" or args.get_adjoint_of == "true":
            for loop_time_idx, test_loop_time in enumerate(test_loop_time_arr):
                print("Loop time:", test_loop_time)
                N_transient_ = pp.get_steps(test_transient_time, sim_dt)
                N_washout_ = pp.get_steps(test_washout_time, sim_dt)
                y_init_prev = y_sim[N_transient_ + N_washout_]
                for loop_idx in range(n_loops):
                    y_bar, t_bar = pp.load_data(
                        beta=p_sim["beta"],
                        tau=p_sim["tau"],
                        x_f=0.2,
                        N_g=N_g,
                        sim_time=test_loop_time,
                        sim_dt=sim_dt,
                        data_dir=data_dir,
                        integrator=integrator,
                        y_init=y_init_prev,
                    )
                    y_init_prev = y_bar[-1]
                    for method_name in methods:
                        if method_name == "direct":
                            dJdp["true"][method_name][
                                p_idx, :, loop_idx, loop_time_idx
                            ] = sens.true_direct_sensitivity(
                                my_rijke, t_bar, y_bar, integrator
                            )
                        elif method_name == "adjoint":
                            dJdp["true"][method_name][
                                p_idx, :, loop_idx, loop_time_idx
                            ] = sens.true_adjoint_sensitivity(
                                my_rijke, t_bar, y_bar, integrator
                            )
                        elif method_name == "numerical":
                            dJdp["true"][method_name][
                                p_idx, :, loop_idx, loop_time_idx
                            ] = sens.true_finite_difference_sensitivity(
                                my_rijke,
                                t_bar,
                                y_bar,
                                h=1e-5,
                                h_tau=network_dt,
                                method=finite_difference_method,
                                integrator=integrator,
                            )
                    J["true"][p_idx, loop_idx, loop_time_idx] = sens.acoustic_energy(
                        y_bar[1:, :], N_g
                    )
                    if loop_idx % 100 == 0:
                        print(f"Loop {loop_idx}")
                        print(
                            f'True dJ/dp, {method_name} = {dJdp["true"][method_name][p_idx, :, loop_idx, loop_time_idx]}',
                            flush=True,
                        )
                        print(
                            f'True J = {J["true"][p_idx, loop_idx, loop_time_idx]}',
                            flush=True,
                        )
        if args.get_adjoint_of == "both" or args.get_adjoint_of == "esn":
            for esn_idx in range(n_ensemble):
                my_ESN = ESN_list[esn_idx]
                # Wash-out phase to get rid of the effects of reservoir states initialised as zero
                # initialise the reservoir states before washout
                N = len(data["loop_0"]["u"])
                x0_washout = np.zeros(my_ESN.N_reservoir)
                N_transient_test = pp.get_steps(test_transient_time, network_dt)

                my_ESN.tau = p_sim["tau"]
                # let the ESN run in open-loop for the wash-out
                # get the initial reservoir to start the actual open/closed-loop,
                # which is the last reservoir state
                if args.same_washout == True:
                    X_tau_1 = my_ESN.open_loop(
                        x0=x0_washout,
                        U=data["loop_0"]["u_washout"],
                        P=data["loop_0"]["p_washout"],
                    )[-my_ESN.N_tau - 1 :, :]
                    P_new = np.vstack(
                        (
                            data["loop_0"]["p_washout"][-my_ESN.N_tau - 1 :, :],
                            data["loop_0"]["p"],
                        )
                    )
                    X_pred, Y_pred = my_ESN.closed_loop(X_tau_1, N_t=N, P=P_new)
                else:
                    # let evolve for a longer time and then remove washout
                    N_transient_test = pp.get_steps(
                        test_transient_time, config.model.network_dt
                    )
                    N_washout = pp.get_steps(
                        config.model.washout_time, config.model.network_dt
                    )
                    N_long = N_transient_test + N

                    P_washout = data["loop_0"]["p"][0] * np.ones((N_washout, 1))
                    P_long = data["loop_0"]["p"][0] * np.ones((N_long, 1))
                    X_pred, Y_pred = my_ESN.closed_loop_with_washout(
                        U_washout=u_washout_auto,
                        N_t=N_long,
                        P_washout=P_washout,
                        P=P_long,
                    )

                    # remove transient
                    X_tau_1 = X_pred[
                        N_transient_test - my_ESN.N_tau - 1 : N_transient_test, :
                    ]
                    X_pred = X_pred[N_transient_test:, :]
                    Y_pred = Y_pred[N_transient_test:, :]

                for loop_time_idx, test_loop_time in enumerate(test_loop_time_arr):
                    X_tau = X_tau_1  # reset X_tau
                    print("Loop time:", test_loop_time, flush=True)
                    N_loop = pp.get_steps(test_loop_time, config.model.network_dt)
                    p_loop = data["loop_0"]["p"][0] * np.ones(
                        (N_loop + my_ESN.N_tau + 1, 1)
                    )
                    for loop_idx in range(n_loops):
                        # split the prediction data
                        x_pred_loop, y_pred_loop = my_ESN.closed_loop(
                            X_tau, N_t=N_loop, P=p_loop
                        )
                        for method_name in methods:
                            if method_name == "direct":
                                dJdp["esn"][method_name][
                                    esn_idx, p_idx, :, loop_idx, loop_time_idx
                                ] = my_ESN.direct_sensitivity(
                                    x_pred_loop,
                                    y_pred_loop,
                                    N_loop,
                                    X_tau,
                                    fast_jac=fast_jac,
                                )
                            elif method_name == "adjoint":
                                dJdp["esn"][method_name][
                                    esn_idx, p_idx, :, loop_idx, loop_time_idx
                                ] = my_ESN.adjoint_sensitivity(
                                    x_pred_loop,
                                    y_pred_loop,
                                    N_loop,
                                    X_tau,
                                    fast_jac=fast_jac,
                                )
                            elif method_name == "numerical":
                                dJdp["esn"][method_name][
                                    esn_idx, p_idx, :, loop_idx, loop_time_idx
                                ] = my_ESN.finite_difference_sensitivity(
                                    X=x_pred_loop,
                                    Y=y_pred_loop,
                                    P=p_loop,
                                    N=N_loop,
                                    X_tau=X_tau,
                                    method=finite_difference_method,
                                )
                        X_tau = x_pred_loop[-my_ESN.N_tau - 1 :]
                        J["esn"][
                            esn_idx, p_idx, loop_idx, loop_time_idx
                        ] = sens.acoustic_energy(y_pred_loop[1:], N_g)

                        if loop_idx % 100 == 0:
                            print(f"Loop {loop_idx}", flush=True)
                            print(
                                f'ESN {esn_idx} dJ/dp, {method_name} = {dJdp["esn"][method_name][esn_idx,p_idx,:,loop_idx, loop_time_idx]}',
                                flush=True,
                            )
                            print(
                                f'ESN {esn_idx} J = {J["esn"][esn_idx, p_idx, loop_idx, loop_time_idx]}'
                            )

    sensitivity_results = {
        "dJdp": dJdp,
        "J": J,
        "beta_list": beta_list,
        "tau_list": tau_list,
        "same_washout": args.same_washout,
        "eta_1_init": args.eta_1_init,
        "loop_times": test_loop_time_arr,
    }

    print(f"Saving results to {model_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(
        model_path / f"sensitivity_results_{dt_string}.pickle", sensitivity_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--beta", nargs="+", type=float)
    parser.add_argument("--tau", nargs="+", type=float)
    parser.add_argument("--same_washout", default=False, action="store_true")
    parser.add_argument("--eta_1_init", default=1.0, type=float)
    parser.add_argument("--train_noise_level", type=float, default=0.0)
    parser.add_argument("--n_loops", default=10, type=int)
    parser.add_argument("--n_ensemble_esn", default=1, type=int)
    parser.add_argument("--loop_times", nargs="+", type=float, default=[-1])
    parser.add_argument("--loop_time_cont", nargs="+", type=float, default=[-1])
    parser.add_argument("--get_adjoint_of", type=str, default="both")
    parsed_args = parser.parse_args()
    main(parsed_args)
