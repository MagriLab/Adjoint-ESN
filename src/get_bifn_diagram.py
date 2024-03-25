import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal

import adjoint_esn.utils.postprocessing as post
from adjoint_esn.rijke_galerkin import sensitivity as sens
from adjoint_esn.rijke_galerkin.solver import Rijke
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar


def main(args):
    same_washout = False

    model_paths = [
        Path(
            "local_results/rijke/run_20231029_153121"
        ),  # rijke with reservoir, trained on beta = 1,2,3,4,5
        Path(
            "local_results/rijke/run_20240307_175258"
        ),  # rijke with reservoir, trained on beta = 6,6.5,7,7.5,8
    ]

    data_dir = Path("data")

    n_ensemble = 1
    test_loop_times = [1000]
    test_loop_names = ["long"]

    if len(args.beta) == 3:
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    elif len(args.beta) == 1:
        beta_list = [args.beta[0]]

    if len(args.tau) == 3:
        tau_list = np.arange(args.tau[0], args.tau[1], args.tau[2])
    elif len(args.tau) == 1:
        tau_list = [args.tau[0]]

    test_param_list = pp.make_param_mesh([beta_list, tau_list])

    # DATA creation
    integrator = "odeint"
    n_models = len(model_paths)
    ESN_list = [[None] * n_ensemble for _ in range(n_models)]

    for model_idx, model_path in enumerate(model_paths):
        config = post.load_config(model_path)
        results = pp.unpickle_file(model_path / "results.pickle")[0]

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

        plt_idx_arr = [
            eOutputVar.mu_1,
            eOutputVar.mu_2,
            eOutputVar.mu_3,
            eOutputVar.mu_4,
        ]

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
        for e_idx in range(n_ensemble):
            # fix the seeds
            if n_ensemble == 1:
                input_seeds = [20, 21, 22]
                reservoir_seeds = [23, 24]
                plt_e_idx = 0
            else:
                input_seeds = [5 * e_idx, 5 * e_idx + 1, 5 * e_idx + 2]
                reservoir_seeds = [5 * e_idx + 3, 5 * e_idx + 4]
                plt_e_idx = 4
            # expand the ESN dict with the fixed seeds
            ESN_dict["input_seeds"] = input_seeds
            ESN_dict["reservoir_seeds"] = reservoir_seeds

            # create an ESN
            print(f"Creating ESN {e_idx+1}/{n_ensemble}.", flush=True)
            my_ESN = post.create_ESN(
                ESN_dict,
                config.model.type,
                hyp_param_names,
                hyp_param_scales,
                hyp_params,
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
            ESN_list[model_idx][e_idx] = my_ESN

    test_sim_time = max(test_loop_times) + transient_time + washout_time

    Y_PKS = [[None] * (len(plt_idx_arr) + 2) for _ in range(len(test_param_list))]
    # [n_params]

    Y_PRED_PKS = [
        [
            [[None] * n_ensemble for _ in range(n_models)]
            for _ in range((len(plt_idx_arr) + 2))
        ]
        for _ in range(len(test_param_list))
    ]
    # [n_params,n_models,n_ensemble]

    # Predict on the test dataset
    for p_idx, p in enumerate(test_param_list):
        p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
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
            loop_names=test_loop_names,
            input_vars=input_vars,
            output_vars=output_vars,
            param_vars=param_vars,
            N_g=N_g,
            u_f_order=u_f_order,
            start_idxs=[0],
            tau=p_sim["tau"],
        )
        for ii, plt_idx in enumerate(plt_idx_arr):
            pks_true = signal.find_peaks(data["long"]["y"][:, plt_idx])[0]
            Y_PKS[p_idx][ii] = data["long"]["y"][pks_true, plt_idx]

        E_ac_true = sens.acoustic_energy_inst(data["long"]["y"], N_g)
        pks_true_ac = signal.find_peaks(E_ac_true)[0]
        Y_PKS[p_idx][-2] = E_ac_true[pks_true_ac]

        u_f_true = Rijke.toVelocity(
            N_g, data["long"]["y"][:, eOutputVar.eta_1 : eOutputVar.eta_1 + N_g], x=0.2
        )[:, 0]
        pks_true_u_f = signal.find_peaks(u_f_true, N_g)[0]
        Y_PKS[p_idx][-1] = u_f_true[pks_true_u_f]

        for model_idx in range(n_models):
            for e_idx in range(n_ensemble):
                my_ESN = ESN_list[model_idx][e_idx]
                print(f"Predicting ESN {e_idx+1}/{n_ensemble}.", flush=True)
                if hasattr(my_ESN, "tau"):
                    my_ESN.tau = p_sim["tau"]

                # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
                if same_washout:
                    # Predict long-term
                    _, y_pred_long = my_ESN.closed_loop_with_washout(
                        U_washout=data["long"]["u_washout"],
                        N_t=len(data["long"]["u"]),
                        P_washout=data["long"]["p_washout"],
                        P=data["long"]["p"],
                    )
                    y_pred_long = y_pred_long[1:]
                else:
                    y0 = np.zeros((1, data["long"]["u_washout"].shape[1]))
                    y0[0, 0] = 1.0
                    u_washout_auto = np.repeat(
                        y0, [len(data["long"]["u_washout"])], axis=0
                    )
                    transient_steps = pp.get_steps(transient_time, network_dt)
                    # add the transient time that will be discarded later
                    N_t_long = transient_steps + len(data["long"]["u"])
                    p_long0 = np.zeros((1, data["long"]["p"].shape[1]))
                    p_long0[0] = data["long"]["p"][0]
                    p_long = np.repeat(p_long0, [N_t_long], axis=0)
                    # Predict long-term
                    _, y_pred_long = my_ESN.closed_loop_with_washout(
                        U_washout=u_washout_auto,
                        N_t=N_t_long,
                        P_washout=data["long"]["p_washout"],
                        P=p_long,
                    )
                    y_pred_long = y_pred_long[1:]
                    y_pred_long = y_pred_long[transient_steps:, :]

                for ii, plt_idx in enumerate(plt_idx_arr):
                    pks_pred = signal.find_peaks(y_pred_long[:, plt_idx])[0]
                    Y_PRED_PKS[p_idx][ii][model_idx][e_idx] = y_pred_long[
                        pks_pred, plt_idx
                    ]  # [n_params,n_plt,n_models,n_ensemble]

                E_ac_pred = sens.acoustic_energy_inst(y_pred_long, N_g)
                pks_pred_ac = signal.find_peaks(E_ac_pred)[0]
                Y_PRED_PKS[p_idx][-2][model_idx][e_idx] = E_ac_pred[pks_pred_ac]

                u_f_pred = Rijke.toVelocity(
                    N_g,
                    y_pred_long[:, eOutputVar.eta_1 : eOutputVar.eta_1 + N_g],
                    x=0.2,
                )[:, 0]
                pks_pred_u_f = signal.find_peaks(u_f_pred, N_g)[0]
                Y_PRED_PKS[p_idx][-1][model_idx][e_idx] = u_f_pred[pks_pred_u_f]

    plot_names = [plt_idx.name for plt_idx in plt_idx_arr]
    plot_names.append("E_ac")
    plot_names.append("u_f")
    bifn_results = {
        "beta_list": beta_list,
        "tau_list": tau_list,
        "true_peaks": Y_PKS,
        "pred_peaks": Y_PRED_PKS,
        "model_paths": model_paths,
        "test_loop_times": test_loop_times,
        "same_washout": same_washout,
        "plot_names": plot_names,
    }

    print(f"Saving results.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(f"local_results/rijke/bifn_results_{dt_string}.pickle", bifn_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", nargs="+", type=float)
    parser.add_argument("--tau", nargs="+", type=float)
    parsed_args = parser.parse_args()
    main(parsed_args)
