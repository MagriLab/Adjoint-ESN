import argparse
import os
import sys
from pathlib import Path

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
from datetime import datetime

import wandb
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.validation_v2 import validate


# os.environ["WANDB_SERVICE_WAIT"] = 300
def load_data(
    data_path,
    dt=1e-1,
    t_washout_len=32,
    t_train_len=256,
    t_val_len=32,
    grid_upsample=4,
    input_var="eta_mu_v",
):
    # load data
    data_dict = pp.read_h5(data_path)

    # select the observed variables
    if input_var == "eta_mu":
        U_sim = data_dict["y"][:, 0 : 2 * data_dict["N_g"]]
    elif input_var == "eta_mu_v":
        U_sim = data_dict["y"][:, 0 : 2 * data_dict["N_g"] + data_dict["N_c"]]
    elif input_var == "eta_mu_v_tau":
        input_idx = np.arange(2 * data_dict["N_g"])
        input_idx = input_idx.tolist()
        input_idx.append(2 * data_dict["N_g"] + data_dict["N_c"] - 1)
        U_sim = data_dict["y"][:, input_idx]
    elif input_var == "pres":
        U_sim = data_dict["P"][:, 1:-1:grid_upsample]
    elif input_var == "vel":
        U_sim = data_dict["U"][:, ::grid_upsample]
    elif input_var == "pres_vel":
        U_sim = np.hstack(
            (data_dict["P"][:, 1:-1:grid_upsample], data_dict["U"][:, ::grid_upsample])
        )
    t_sim = data_dict["t"]

    # upsample
    data_dt = data_dict["t"][1] - data_dict["t"][0]
    upsample = pp.get_steps(dt, data_dt)

    t = t_sim[::upsample]
    U = U_sim[::upsample, :]

    # cut the transient
    t_transient_len = data_dict["t_transient"]
    N_transient = pp.get_steps(t_transient_len, dt)
    U = U[N_transient:, :]
    t = t[N_transient:] - t[N_transient]

    # separate into washout, train, test
    N_washout = pp.get_steps(t_washout_len, dt)
    N_train = pp.get_steps(t_train_len, dt)
    N_val = int(np.round(t_val_len / dt))

    U_washout_train = U[0:N_washout, :]
    U_train = U[N_washout : N_washout + N_train - 1, :]
    Y_train = U[N_washout + 1 : N_washout + N_train, :]
    t_train = t[N_washout + 1 : N_washout + N_train]

    U_washout_val = U[N_washout + N_train : 2 * N_washout + N_train, :]
    U_val = U[2 * N_washout + N_train : 2 * N_washout + N_train + N_val - 1, :]
    Y_val = U[2 * N_washout + N_train + 1 : 2 * N_washout + N_train + N_val, :]
    t_val = t[2 * N_washout + N_train + 1 : 2 * N_washout + N_train + N_val]

    return (
        U_washout_train,
        U_train,
        Y_train,
        t_train,
        U_washout_val,
        U_val,
        Y_val,
        t_val,
    )


def create_dataset(
    p_list,
    dt=1e-1,
    t_washout_len=32,
    t_train_len=256,
    t_val_len=32,
    grid_upsample=4,
    input_var="eta_mu_v",
    p_var="all",
):
    len_p_list = len(p_list)
    U_washout_train = [None] * len_p_list
    U_train = [None] * len_p_list
    Y_train = [None] * len_p_list
    t_train = [None] * len_p_list
    P_washout_train = [None] * len_p_list
    P_train = [None] * len_p_list

    U_washout_val = [None] * len_p_list
    U_val = [None] * len_p_list
    Y_val = [None] * len_p_list
    t_val = [None] * len_p_list
    P_washout_val = [None] * len_p_list
    P_val = [None] * len_p_list

    for p_idx, params in enumerate(p_list):
        beta = params[0]
        tau = params[1]
        beta_name = f"{beta:.2f}"
        beta_name = beta_name.replace(".", "_")
        tau_name = f"{tau:.2f}"
        tau_name = tau_name.replace(".", "_")
        sim_path = Path(
            f"src/data_new/rijke_kings_poly_N_g_4_beta_{beta_name}_tau_{tau_name}.h5"
        )
        # print(sim_path.absolute(), flush=True)
        (
            U_washout_train[p_idx],
            U_train[p_idx],
            Y_train[p_idx],
            t_train[p_idx],
            U_washout_val[p_idx],
            U_val[p_idx],
            Y_val[p_idx],
            t_val[p_idx],
        ) = load_data(
            sim_path,
            dt,
            t_washout_len,
            t_train_len,
            t_val_len,
            grid_upsample,
            input_var,
        )
        if p_var == "all":
            train_param_var = params
        elif p_var == "beta":
            train_param_var = beta
        elif p_var == "tau":
            train_param_var = tau
        P_washout_train[p_idx] = train_param_var * np.ones(
            (len(U_washout_train[p_idx]), 1)
        )
        P_train[p_idx] = train_param_var * np.ones((len(U_train[p_idx]), 1))
        P_washout_val[p_idx] = train_param_var * np.ones((len(U_washout_val[p_idx]), 1))
        P_val[p_idx] = train_param_var * np.ones((len(U_val[p_idx]), 1))

    return (
        U_washout_train,
        P_washout_train,
        U_train,
        P_train,
        Y_train,
        t_train,
        U_washout_val,
        P_washout_val,
        U_val,
        P_val,
        Y_val,
        t_val,
    )


def input_norm_and_bias(U):
    # input scaling and bias
    U_mean = U.mean(axis=0)
    U_std = U.std(axis=0)
    # m = U.min(axis=0)
    # M = U.max(axis=0)
    # U_norm = M-m
    scale = (U_mean, U_std)
    bias = np.array([np.mean(np.abs((U - scale[0]) / scale[1]))])
    return scale, bias


def main(args):
    # mesh to choose training data from
    if args.p_var == "all":
        beta_list = np.arange(1.2, 2.9, 0.1)
        tau_list = np.arange(0.12, 0.29, 0.01)
    elif args.p_var == "beta":
        beta_list = np.arange(0.6, 9.0, 0.1)
        tau_list = np.array([0.2])
    elif args.p_var == "tau":
        beta_list = np.array([2.5])
        tau_list = np.arange(0.12, 0.29, 0.01)

    beta_mesh, tau_mesh = np.meshgrid(beta_list, tau_list)
    p_mesh = np.hstack([beta_mesh.flatten()[:, None], tau_mesh.flatten()[:, None]])

    # randomly choose the training and validation regimes
    if args.n_train + args.n_val <= len(p_mesh):
        if args.selection == ["random"]:
            rnd = np.random.RandomState(seed=args.seed)
            train_val_idx_list = rnd.choice(
                len(p_mesh), size=args.n_train + args.n_val, replace=False
            )
        else:
            train_val_idx_list = np.array([*map(int, args.selection)])
        p_train_val_list = p_mesh[train_val_idx_list, :]
    else:
        raise ValueError(
            "The number of training and validation regimes is greater than the mesh."
        )
    print("Creating dataset.", flush=True)
    # create the dataset containing the train and validation regimes
    data_config = {
        "dt": args.time_step,
        "t_washout_len": 4,
        "t_train_len": 256,
        "t_val_len": 64,
        "grid_upsample": 0,
        "input_var": args.input_var,
    }
    (
        U_washout_train,
        P_washout_train,
        U_train,
        P_train,
        Y_train,
        _,
        U_washout_val,
        P_washout_val,
        U_val,
        P_val,
        Y_val,
        _,
    ) = create_dataset(p_train_val_list, **data_config, p_var=args.p_var)
    train_idx_list = np.arange(args.n_train)
    if args.validate_on_train:
        val_idx_list = np.arange(0, args.n_train + args.n_val)
    else:
        val_idx_list = np.arange(args.n_train, args.n_train + args.n_val)
    print("Train indices: ", train_idx_list)
    print("Validation indices: ", val_idx_list)

    dim = U_train[0].shape[1]
    if args.ESN_version == "v1":
        input_scale, input_bias = input_norm_and_bias(np.vstack(U_train))
        output_bias = np.array([1.0])
        r2_mode = False
    elif args.ESN_version == "v2":
        input_scale = [None] * 2
        input_scale[0] = np.zeros(dim)
        input_scale[1] = np.ones(dim)
        input_bias = np.array([])
        output_bias = np.array([])
        r2_mode = True

    print("Dimension", dim)
    print("Creating hyperparameter search range.", flush=True)
    # In case we want to start from a grid_search,
    n_param = P_train[0].shape[1]
    # the first n_grid_x*n_grid_y points are from grid search
    hyp_param_names = [
        "spectral_radius",
        "input_scaling",
        "leak_factor",
    ]
    hyp_param_names.extend(["parameter_normalization_mean"] * n_param)
    hyp_param_names.extend(["parameter_normalization_var"] * n_param)

    hyp_param_scales = [
        "uniform",
        "log10",
        "uniform",
    ]
    hyp_param_scales.extend(["uniform"] * n_param)
    hyp_param_scales.extend(["log10"] * n_param)

    # range for hyperparameters (spectral radius and input scaling)
    spec_in = 0.1
    spec_end = 1.0
    in_scal_in = np.log10(0.01)  # std of 0.5
    in_scal_end = np.log10(10.0)  # std of 0.05
    leak_in = 0.1
    leak_end = 1.0
    p_mean = np.mean(np.vstack(P_train), axis=0)
    p_std = np.std(np.vstack(P_train), axis=0)
    p_norm_mean_in = -1.0 * p_mean
    p_norm_mean_end = 2.0 * p_mean
    p_norm_mean = np.array([p_norm_mean_in, p_norm_mean_end]).T
    p_norm_var_in = np.log10(0.2 * p_std)  # std of 5
    p_norm_var_end = np.log10(40.0 * p_std)  # std of 0.025
    p_norm_var = np.array([p_norm_var_in, p_norm_var_end]).T
    grid_range = [
        [spec_in, spec_end],
        [in_scal_in, in_scal_end],
        [leak_in, leak_end],
    ]
    grid_range.extend(p_norm_mean.tolist())
    grid_range.extend(p_norm_var.tolist())
    # n_grid = [4] * len(grid_range)
    N_washout = int(np.round(data_config["t_washout_len"] / data_config["dt"]))
    t_val_len = 8
    N_val = int(np.round(t_val_len / data_config["dt"]))
    # N_fwd = int(np.round(2 * data_config["t_washout_len"] / data_config["dt"]))
    N_transient = 0
    noise_std = args.noise_std
    n_folds = args.n_folds
    n_realisations = args.n_realisations
    ESN_dict = {
        "reservoir_size": args.reservoir_size,
        "dimension": dim,
        "parameter_dimension": n_param,
        "reservoir_connectivity": args.connectivity,
        "input_normalization": input_scale,
        "input_bias": input_bias,
        "output_bias": output_bias,
        "r2_mode": r2_mode,
    }
    for refine in range(args.n_refinements):
        print("Starting validation.", flush=True)
        print("Grid_range", grid_range, flush=True)
        min_dict = validate(
            grid_range,
            hyp_param_names,
            hyp_param_scales,
            n_calls=5,
            n_initial_points=5,
            n_ensemble=1,
            ESN_dict=ESN_dict,
            tikh=args.tikh,
            U_washout_train=U_washout_train,
            U_train=U_train,
            U_val=U_val,
            Y_train=Y_train,
            Y_val=Y_val,
            P_washout_train=P_washout_train,
            P_train=P_train,
            P_val=P_val,
            n_folds=n_folds,
            n_realisations=n_realisations,
            N_washout_steps=N_washout,
            N_val_steps=N_val,
            N_transient_steps=N_transient,
            train_idx_list=train_idx_list,
            val_idx_list=val_idx_list,
        )

        results = {
            "data_config": data_config,
            "p_train_val_list": p_train_val_list,
            "train_idx_list": train_idx_list,
            "val_idx_list": val_idx_list,
            "hyp_param_names": hyp_param_names,
            "hyp_param_scales": hyp_param_scales,
            "hyp_grid_range": grid_range,
            "ESN_dict": ESN_dict,
            "N_washout": N_washout,
            "N_val": N_val,
            "N_transient": N_transient,
            "n_folds": n_folds,
            "noise_std": noise_std,
            "min_dict": min_dict,
        }
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        save_path = Path(f"src/results/val_run_{dt_string}.pickle")
        print(save_path, flush=True)
        pp.pickle_file(str(save_path.absolute()), results)

        # REFINEMENT
        # range for hyperparameters (spectral radius and input scaling)
        spec_in = (spec_in + min_dict["params"][0][0][0]) / 2
        spec_end = (spec_end + min_dict["params"][0][0][0]) / 2
        in_scal_in = (in_scal_in + np.log10(min_dict["params"][0][0][1])) / 2
        in_scal_end = (in_scal_end + np.log10(min_dict["params"][0][0][1])) / 2
        leak_in = (leak_in + min_dict["params"][0][0][2]) / 2
        leak_end = (leak_end + min_dict["params"][0][0][2]) / 2
        p_norm_mean_in = (
            p_norm_mean_in + min_dict["params"][0][0][3 : 3 + n_param]
        ) / 2
        p_norm_mean_end = (
            p_norm_mean_end + min_dict["params"][0][0][3 : 3 + n_param]
        ) / 2
        p_norm_mean = np.array([p_norm_mean_in, p_norm_mean_end]).T
        p_norm_var_in = (
            p_norm_var_in
            + np.log10(min_dict["params"][0][0][3 + n_param : 3 + 2 * n_param])
        ) / 2
        p_norm_var_end = (
            p_norm_var_end
            + np.log10(min_dict["params"][0][0][3 + n_param : 3 + 2 * n_param])
        ) / 2
        p_norm_var = np.array([p_norm_var_in, p_norm_var_end]).T
        grid_range = [
            [spec_in, spec_end],
            [in_scal_in, in_scal_end],
            [leak_in, leak_end],
        ]
        grid_range.extend(p_norm_mean.tolist())
        grid_range.extend(p_norm_var.tolist())
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and validates ESN on Rijke tube data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed to choose training data from the mesh",
    )
    parser.add_argument(
        "--input_var",
        type=str,
        default="eta_mu_v_tau",
        help="which state variable to include as input",
    )
    parser.add_argument(
        "--p_var",
        type=str,
        default="beta",
        help="which parameters to include",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=1e-2,
        help="time step",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=2,
        help="number of training regimes",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=1,
        help="number of validation regimes",
    )
    parser.add_argument("--selection", nargs="+", default=[4, 14, 9])
    parser.add_argument(
        "--validate_on_train",
        default=False,
        action="store_true",
        help="whether to use the training regimes in validation",
    )
    parser.add_argument(
        "--reservoir_size",
        type=int,
        default=300,
        help="size of the ESN reservoir",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=3,
        help="connectivity of the ESN reservoir",
    )
    parser.add_argument(
        "--tikh",
        type=float,
        default=1e-3,
        help="tikhonov",
    )
    parser.add_argument(
        "--n_ensemble",
        type=int,
        default=1,
        help="number of initialisations of the validation",
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0,
        help="percentage noise level",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="number of folds",
    )
    parser.add_argument(
        "--n_realisations",
        type=int,
        default=3,
        help="number of realisations",
    )
    parser.add_argument(
        "--n_refinements",
        type=int,
        default=3,
        help="number of refinements",
    )
    parser.add_argument(
        "--ESN_version",
        type=str,
        default="v2",
        help="version of ESN",
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
