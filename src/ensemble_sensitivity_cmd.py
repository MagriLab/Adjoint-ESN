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
    # load model
    model_path = Path(args.model_dir) / f"run_{args.run_name}"
    config = post.load_config(model_path)
    results = pp.unpickle_file(model_path / "results.pickle")[0]

    # set data directory
    data_dir = Path(args.data_dir)

    # options
    n_ensemble = args.n_ensemble_esn
    eta_1_init = args.eta_1_init

    # make a list of test parameters
    if len(args.beta) == 3:
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    elif len(args.beta) == 1:
        beta_list = [args.beta[0]]

    if len(args.tau) == 3:
        tau_list = np.arange(args.tau[0], args.tau[1], args.tau[2])
    elif len(args.tau) == 1:
        tau_list = [args.tau[0]]
    p_list = pp.make_param_mesh([beta_list, tau_list])

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

    finite_difference_method = "central"
    methods = ["adjoint"]
    dJdp = {"true": {}, "esn": {}}
    for method_name in methods:
        dJdp["true"][method_name] = np.zeros((len(p_list), 2))
        if config.model.type == "standard":
            dJdp["esn"][method_name] = np.zeros(
                (n_ensemble, len(p_list), len(param_vars))
            )
        elif config.model.type == "rijke":
            dJdp["esn"][method_name] = np.zeros((n_ensemble, len(p_list), 2))

    J = {"true": np.zeros(len(p_list)), "esn": np.zeros((n_ensemble, len(p_list)))}

    test_transient_time = config.simulation.transient_time
    test_washout_time = config.model.washout_time

    # add fast jacobian condition
    if config.model.input_only_mode == False and config.model.r2_mode == False:
        fast_jac = True
    print("Fast jac condition:", fast_jac)

    test_sim_time = args.loop_time + test_transient_time + test_washout_time
    test_loop_times = [args.loop_time]

    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}

        regime_str = f'beta = {p_sim["beta"]}, tau = {p_sim["tau"]}'
        print("Regime:", regime_str)

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

        loop_name = list(data.keys())[0]

        N_transient = pp.get_steps(transient_time, sim_dt)
        y_bar = y_sim[N_transient:, :]
        t_bar = t_sim[N_transient:]

        N_washout = pp.get_steps(washout_time, sim_dt)
        N_loop = pp.get_steps(test_loop_times[0], sim_dt) + 1
        y_bar = y_bar[N_washout : N_washout + N_loop, :]
        t_bar = t_bar[N_washout : N_washout + N_loop] - t_bar[N_washout]

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

        for method_name in methods:
            if method_name == "direct":
                dJdp["true"][method_name][p_idx] = sens.true_direct_sensitivity(
                    my_rijke, t_bar, y_bar, integrator
                )
            elif method_name == "adjoint":
                dJdp["true"][method_name][p_idx] = sens.true_adjoint_sensitivity(
                    my_rijke, t_bar, y_bar, integrator
                )
            elif method_name == "numerical":
                dJdp["true"][method_name][
                    p_idx
                ] = sens.true_finite_difference_sensitivity(
                    my_rijke,
                    t_bar,
                    y_bar,
                    h=1e-5,
                    h_tau=network_dt,
                    method=finite_difference_method,
                    integrator=integrator,
                )
            print(
                f'True dJ/dp, {method_name} = {dJdp["true"][method_name][p_idx]}',
                flush=True,
            )

        J["true"][p_idx] = sens.acoustic_energy(y_bar[1:, :], N_g)
        print(f'True J = {J["true"][p_idx]}', flush=True)

        for esn_idx in range(n_ensemble):
            my_ESN = ESN_list[esn_idx]
            # Wash-out phase to get rid of the effects of reservoir states initialised as zero
            # initialise the reservoir states before washout
            x0_washout = np.zeros(my_ESN.N_reservoir)
            N = len(data[loop_name]["u"])

            N_transient_test = pp.get_steps(test_transient_time, network_dt)
            if hasattr(my_ESN, "tau"):
                my_ESN.tau = p_sim["tau"]
                # let the ESN run in open-loop for the wash-out
                # get the initial reservoir to start the actual open/closed-loop,
                # which is the last reservoir state
                if args.same_washout == True:
                    X_tau = my_ESN.open_loop(
                        x0=x0_washout,
                        U=data[loop_name]["u_washout"],
                        P=data[loop_name]["p_washout"],
                    )
                    P_grad = np.vstack(
                        (
                            data[loop_name]["p_washout"][-my_ESN.N_tau - 1 :, :],
                            data[loop_name]["p"],
                        )
                    )
                    X_pred_grad, Y_pred_grad = my_ESN.closed_loop(
                        X_tau[-my_ESN.N_tau - 1 :, :], N_t=N, P=P_grad
                    )
                else:
                    # assume don't have access to the washout data
                    X_tau = my_ESN.open_loop(
                        x0=x0_washout, U=u_washout_auto, P=data[loop_name]["p_washout"]
                    )
                    # let evolve for a longer time and then remove washout
                    N_long = N_transient_test + N
                    P_grad = data[loop_name]["p_washout"][0] * np.ones((N_long, 1))
                    P_grad = np.vstack(
                        (data[loop_name]["p_washout"][-my_ESN.N_tau - 1 :, :], P_grad)
                    )
                    X_pred_grad, Y_pred_grad = my_ESN.closed_loop(
                        X_tau[-my_ESN.N_tau - 1 :, :], N_t=N_long, P=P_grad
                    )

                    # remove transient
                    X_tau = X_pred_grad[:N_transient_test, :]
                    X_pred_grad = X_pred_grad[N_transient_test:, :]
                    Y_pred_grad = Y_pred_grad[N_transient_test:, :]
                    P_grad = P_grad[N_transient_test:, :]

                for method_name in methods:
                    if method_name == "direct":
                        dJdp["esn"][method_name][
                            esn_idx, p_idx
                        ] = my_ESN.direct_sensitivity(
                            X_pred_grad,
                            Y_pred_grad,
                            N,
                            X_tau,
                            fast_jac=fast_jac,
                        )
                    elif method_name == "adjoint":
                        dJdp["esn"][method_name][
                            esn_idx, p_idx
                        ] = my_ESN.adjoint_sensitivity(
                            X_pred_grad,
                            Y_pred_grad,
                            N,
                            X_tau,
                            fast_jac=fast_jac,
                        )
                    elif method_name == "numerical":
                        dJdp["esn"][method_name][
                            esn_idx, p_idx
                        ] = my_ESN.finite_difference_sensitivity(
                            X_pred_grad,
                            Y_pred_grad,
                            X_tau,
                            P_grad,
                            N,
                            method=finite_difference_method,
                        )
                    print(
                        f'ESN {esn_idx} dJ/dp, {method_name} = {dJdp["esn"][method_name][esn_idx,p_idx]}',
                        flush=True,
                    )
            else:
                X_pred_grad, Y_pred_grad = my_ESN.closed_loop_with_washout(
                    U_washout=data[loop_name]["u_washout"],
                    N_t=N,
                    P_washout=data[loop_name]["p_washout"],
                    P=data[loop_name]["p"],
                )

                for method_name in methods:
                    if method_name == "direct":
                        dJdp["esn"][method_name][
                            esn_idx, p_idx
                        ] = my_ESN.direct_sensitivity(
                            X_pred_grad,
                            Y_pred_grad,
                            N,
                            N_g,
                            fast_jac=fast_jac,
                        )
                    elif method_name == "adjoint":
                        dJdp["esn"][method_name][
                            esn_idx, p_idx
                        ] = my_ESN.adjoint_sensitivity(
                            X_pred_grad, Y_pred_grad, N, N_g, fast_jac=fast_jac
                        )
                    elif method_name == "numerical":
                        dJdp["esn"][method_name][
                            esn_idx, p_idx
                        ] = my_ESN.finite_difference_sensitivity(
                            X=X_pred_grad,
                            Y=Y_pred_grad,
                            P=data[loop_name]["p"],
                            N=N,
                            N_g=N_g,
                            method=finite_difference_method,
                        )
                    print(
                        f'ESN {esn_idx} dJ/dp, {method_name} = {dJdp["esn"][method_name][esn_idx,p_idx]}',
                        flush=True,
                    )

            J["esn"][esn_idx, p_idx] = sens.acoustic_energy(Y_pred_grad[1:, :], N_g)
            print(f'ESN {esn_idx} J = {J["esn"][esn_idx, p_idx]}')

    sensitivity_results = {
        "dJdp": dJdp,
        "J": J,
        "beta_list": beta_list,
        "tau_list": tau_list,
        "same_washout": args.same_washout,
        "eta_1_init": args.eta_1_init,
    }

    print(f"Saving results to {model_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(
        model_path / f"sensitivity_results_{dt_string}.pickle", sensitivity_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--beta", nargs="+", type=float, default=[0.6])
    parser.add_argument("--tau", nargs="+", type=float, default=[0.2])
    parser.add_argument("--same_washout", default=False, action="store_true")
    parser.add_argument("--eta_1_init", default=1.0, type=float)
    parser.add_argument("--n_ensemble_esn", default=1, type=int)
    parser.add_argument("--loop_time", type=float, default=[2])
    parsed_args = parser.parse_args()
    main(parsed_args)
