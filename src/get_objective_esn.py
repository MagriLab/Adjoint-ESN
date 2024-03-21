import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.utils import lyapunov as lyap
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63


def objective_fun(u):
    return np.mean(u[:, 2])


def dobjective_fun(u):
    dobj = np.zeros(len(u))
    dobj[2] = 1
    return dobj


def Lyapunov_Time(sys, params, transient_time, sim_dt, integrator):
    sim_time = 1000 + transient_time
    my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
        sys, params, sim_time, sim_dt, y_init=[-2.4, -3.7, 14.98], integrator=integrator
    )
    y, t = pp.discard_transient(y_sim, t_sim, transient_time)
    LEs, _, _, _ = lyap.calculate_LEs(
        sys=my_sys,
        sys_type="continuous",
        X=y,
        t=t,
        transient_time=transient_time,
        dt=sim_dt,
        target_dim=None,
        norm_step=1,
    )
    LEs_target = LEs[-1]
    LT = 1 / LEs_target[0]
    if LT <= 0:
        return 1
    else:
        return LT


def main(args):
    model_path = Path(f"local_results/lorenz63/run_{args.run_name}")
    n_ensemble = args.n_ensemble_esn

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
    ESN_list = [None] * args.n_ensemble
    for e_idx in range(n_ensemble):
        # fix the seeds
        input_seeds = [5 * e_idx, 5 * e_idx + 1, 5 * e_idx + 2]
        reservoir_seeds = [5 * e_idx + 3, 5 * e_idx + 4]
        if n_ensemble == 1:
            input_seeds = [5, 6, 7]
            reservoir_seeds = [8, 9]

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

    J = {
        "true": np.zeros(len(p_list)),
        "esn": np.zeros((n_ensemble, len(p_list))),
    }

    test_transient_time = config.simulation.transient_time
    # when running with same_washout=True, loops start after the washout
    # when running with same_washout=False, loops start after the transient
    # that's why the initial condition of the first loop changes
    if args.same_washout:
        test_washout_time = config.model.washout_time
    else:
        test_washout_time = 0

    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}

        params = {}
        for param in eParam:
            params[param.name] = p[param]
        regime_str = " ".join([f"{param.name} = {p[param]:0.3f}" for param in eParam])
        print(regime_str)

        # determine Lyapunov time and choose simulation time accordingly
        LT = Lyapunov_Time(
            Lorenz63,
            p_sim,
            config.simulation.transient_time,
            config.simulation.sim_dt,
            integrator=config.simulation.integrator,
        )
        test_sim_time = args.sim_time_LT * LT + test_transient_time + test_washout_time

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
            washout_time=test_washout_time,
            loop_times=args.sim_time_LT * LT,
            input_vars=config.model.input_vars,
            param_vars=config.model.param_vars,
        )
        J["true"][p_idx] = objective_fun(data["loop_0"]["y"])

        Y_PRED_LONG = [None] * n_ensemble
        for e_idx in range(n_ensemble):
            my_ESN = ESN_list[e_idx]
            print(f"Predicting ESN {e_idx+1}/{n_ensemble}.", flush=True)

            # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
            if args.same_washout:
                # Predict long-term
                _, y_pred_long = my_ESN.closed_loop_with_washout(
                    U_washout=data["loop_0"]["u_washout"],
                    N_t=len(data["loop_0"]["u"]),
                    P_washout=data["loop_0"]["p_washout"],
                    P=data["loop_0"]["p"],
                )
                y_pred_long = y_pred_long[1:]
            else:
                # let evolve for a longer time and then remove washout
                N_transient_test = pp.get_steps(
                    test_transient_time, config.model.network_dt
                )
                N_washout = pp.get_steps(
                    config.model.washout_time, config.model.network_dt
                )
                N_long = N_transient_test + len(data["loop_0"]["u"])

                P_washout = data["loop_0"]["p"][0] * np.ones((N_washout, 1))
                p_long = data["loop_0"]["p"][0] * np.ones((N_long, 1))
                _, y_pred_long = my_ESN.closed_loop_with_washout(
                    U_washout=u_washout_auto,
                    N_t=N_long,
                    P_washout=P_washout,
                    P=p_long,
                )
                y_pred_long = y_pred_long[1:]
                y_pred_long = y_pred_long[N_transient_test:, :]

            Y_PRED_LONG[e_idx] = y_pred_long
            J["esn"][e_idx, p_idx] = objective_fun(Y_PRED_LONG[e_idx])
    objective_results = {
        "J": J,
        "beta_list": beta_list,
        "rho_list": rho_list,
        "sigma_list": sigma_list,
        "same_washout": args.same_washout,
        "y_init": args.y_init,
        "sim_time_LT": args.sim_time_LT,
    }

    print(f"Saving results to {model_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(
        model_path / f"objective_results_{dt_string}.pickle", objective_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--beta", nargs="+", type=float)
    parser.add_argument("--rho", nargs="+", type=float)
    parser.add_argument("--sigma", nargs="+", type=float)
    parser.add_argument("--same_washout", default=False, action="store_true")
    parser.add_argument("--y_init", nargs="+", type=float)
    parser.add_argument("--sim_time_LT", type=int)
    parser.add_argument("--n_ensemble_esn", default=5, type=int)
    parsed_args = parser.parse_args()
    main(parsed_args)
