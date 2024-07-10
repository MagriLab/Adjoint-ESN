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
import yaml

from adjoint_esn.utils import errors
from adjoint_esn.utils import postprocessing as post
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import scalers
from adjoint_esn.utils.enums import eParam, get_eVar
from adjoint_esn.validation import loop


def main(args):
    # load model
    model_path = Path(args.model_dir) / f"run_{args.run_name}"
    config = post.load_config(model_path)
    results = pp.unpickle_file(model_path / "results.pickle")[0]

    # set data directory
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
    val_param_list = results["validation_parameters"]
    train_random_seeds = results["training_random_seeds"]
    val_random_seeds = results["validation_random_seeds"]

    train_idx_list = np.arange(len(train_param_list))
    val_idx_list = np.arange(
        len(train_param_list), len(train_param_list) + len(val_param_list)
    )

    full_list = np.vstack([train_param_list, val_param_list])
    full_random_seeds = [*train_random_seeds, *val_random_seeds]

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
    val_time = config.val.time

    loop_names = ["train", "val"]
    loop_times = [train_time, val_time]

    # validation configurations
    N_washout = pp.get_steps(config.model.washout_time, config.model.network_dt)
    N_val = pp.get_steps(config.val.fold_time, config.model.network_dt)

    N_transient = 0
    n_param = len(config.model.param_vars)

    # ESN configurations
    ESN_dict = {
        "reservoir_size": config.model.reservoir_size,
        "parameter_dimension": n_param,
        "reservoir_connectivity": config.model.connectivity,
        "r2_mode": config.model.r2_mode,
        "input_only_mode": config.model.input_only_mode,
        "input_weights_mode": config.model.input_weights_mode,
        "reservoir_weights_mode": config.model.reservoir_weights_mode,
        "tikhonov": results["tikhonov"][0],
        "leak_factor": results["leak_factor"][0],
        "input_scaling": results["input_scaling"][0],
        "u_f_scaling": results["u_f_scaling"][0],
        "parameter_normalization": [
            results["parameter_normalization_mean"][0],
            results["parameter_normalization_var"][0],
        ],
        "spectral_radius": results["spectral_radius"][0],
    }
    if config.model.type == "standard":
        ESN_dict["dimension"] = dim
    elif config.model.type == "rijke":
        ESN_dict["N_g"] = config.simulation.N_g
        ESN_dict["x_f"] = 0.2
        ESN_dict["dt"] = config.model.network_dt
        ESN_dict["u_f_order"] = config.model.u_f_order

    tikh_in = args.tikh_in
    tikh_end = args.tikh_end
    grid_range = [[tikh_in, tikh_end]]
    hyp_param_names = ["tikhonov"]
    hyp_param_scales = ["log10"]

    for i in range(len(grid_range)):
        for j in range(2):
            scaler = getattr(scalers, hyp_param_scales[i])
            grid_range[i][j] = scaler(grid_range[i][j])

    tikhs = np.linspace(grid_range[0][0], grid_range[0][1], args.n_tikhs)
    err_val = np.zeros((args.n_noise_realisations, args.n_tikhs))

    if args.n_folds <= 0:
        n_folds = config.val.n_folds
    else:
        n_folds = args.n_folds
    for noise_idx in range(args.n_noise_realisations):
        # load data
        print(
            f"Noise realisation {noise_idx+1}/{args.n_noise_realisations}.", flush=True
        )
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

        for p_idx, p in enumerate(full_list):
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
                noise_seed = full_random_seeds[p_idx] + 100 * noise_idx
            else:
                noise_seed = random_seed + p_idx + 100 * noise_idx

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

        for tikh_idx, tikh in enumerate(tikhs):
            params = [tikh]
            err_val[noise_idx, tikh_idx] = loop(
                params,
                hyp_param_names,
                hyp_param_scales,
                ESN_dict,
                U_washout_train=DATA["train"]["u_washout"],
                U_train=DATA["train"]["u"],
                U_val=DATA["val"]["u"],
                Y_train=DATA["train"]["y"],
                Y_val=DATA["val"]["y"],
                n_folds=n_folds,
                n_realisations=config.val.n_realisations,
                N_washout=N_washout,
                N_val=N_val,
                N_trans=N_transient,
                P_washout_train=DATA["train"]["p_washout"],
                P_train=DATA["train"]["p"],
                P_val=DATA["val"]["p"],
                train_idx_list=train_idx_list,
                val_idx_list=val_idx_list,
                p_list=full_list,
                ESN_type="rijke",  # "standard" or "rijke"
                error_measure=getattr(errors, config.val.error_measure),
                LT=None,  # only needed if error measure predictability horizon
                network_dt=None,  # only needed if error measure predictability horizon
            )

    error_results = {
        "err_val": err_val,
        "tikhs": tikhs,
        "noise_level": config.simulation.noise_level,
    }

    print(f"Saving results to {model_path}.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(model_path / f"error_val_results_{dt_string}.pickle", error_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--n_noise_realisations", default=1, type=int)
    parser.add_argument("--n_tikhs", default=11, type=int)
    parser.add_argument("--tikh_in", default=1e-6, type=float)
    parser.add_argument("--tikh_end", default=1e-1, type=float)
    parser.add_argument("--n_folds", default=-1, type=int)
    parsed_args = parser.parse_args()
    main(parsed_args)
