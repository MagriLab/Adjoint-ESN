import argparse
import os
import sys

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.validation import validate


def load_data(
    data_path,
    dt=1e-1,
    t_washout_len=32,
    t_train_len=256,
    grid_upsample=4,
    train_var="gal",
):
    # load data
    data_dict = pp.read_h5(data_path)

    # select the observed variables
    if train_var == "gal":
        U_sim = data_dict["y"][:, 0 : 2 * data_dict["N_g"]]
    elif train_var == "pres":
        U_sim = data_dict["P"][:, 1:-1:grid_upsample]
    elif train_var == "vel":
        U_sim = data_dict["U"][:, ::grid_upsample]
    elif train_var == "pres_vel":
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

    U_washout_train = U[0:N_washout, :]

    U_train = U[N_washout : N_washout + N_train - 1, :]
    Y_train = U[N_washout + 1 : N_washout + N_train, :]
    t_train = t[N_washout + 1 : N_washout + N_train]

    U_data = U[: N_washout + N_train]
    return U_washout_train, U_train, Y_train, t_train, U_data


def create_dataset(
    p_list,
    dt=1e-1,
    t_washout_len=32,
    t_train_len=256,
    grid_upsample=4,
    train_var="gal",
    p_var="both",
):
    len_p_list = len(p_list)
    U_washout_train = [None] * len_p_list
    U_train = [None] * len_p_list
    U_data = [None] * len_p_list
    Y_train = [None] * len_p_list
    t_train = [None] * len_p_list
    P_washout_train = [None] * len_p_list
    P_train = [None] * len_p_list

    for p_idx, params in enumerate(p_list):
        beta = params[0]
        tau = params[1]
        beta_name = f"{beta:.2f}"
        beta_name = beta_name.replace(".", "_")
        tau_name = f"{tau:.2f}"
        tau_name = tau_name.replace(".", "_")
        sim_str = f"src/data/rijke_kings_poly_beta_{beta_name}_tau_{tau_name}.h5"
        (
            U_washout_train[p_idx],
            U_train[p_idx],
            Y_train[p_idx],
            t_train[p_idx],
            U_data[p_idx],
        ) = load_data(sim_str, dt, t_washout_len, t_train_len, grid_upsample, train_var)
        if p_var == "both":
            train_param_var = params
        elif p_var == "beta":
            train_param_var = beta
        elif p_var == "tau":
            train_param_var = tau
        P_washout_train[p_idx] = train_param_var * np.ones(
            (len(U_washout_train[p_idx]), 1)
        )
        P_train[p_idx] = train_param_var * np.ones((len(U_train[p_idx]), 1))
    U_data = np.vstack(U_data)
    return U_washout_train, P_washout_train, U_train, P_train, Y_train, t_train, U_data


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
    beta_list = np.array([2.5])
    tau_list = np.arange(0.15, 0.26, 0.01)
    beta_mesh, tau_mesh = np.meshgrid(beta_list, tau_list)
    p_mesh = np.hstack([beta_mesh.flatten()[:, None], tau_mesh.flatten()[:, None]])

    # randomly choose the training and validation regimes
    if args.n_train + args.n_val <= len(p_mesh):
        rnd = np.random.RandomState(seed=args.seed)
        train_val_idx_list = rnd.choice(
            len(p_mesh), size=args.n_train + args.n_val, replace=False
        )
        p_train_val_list = p_mesh[train_val_idx_list, :]
    else:
        raise ValueError(
            "The number of training and validation regimes is greater than the mesh."
        )

    # create the dataset containing the train and validation regimes
    data_config = {
        "dt": 1e-1,
        "t_washout_len": 8,
        "t_train_len": 256,
        "grid_upsample": 4,
        "train_var": "gal",
    }
    (
        U_washout_train,
        P_washout_train,
        U_train,
        P_train,
        Y_train,
        t_train,
        U_data,
    ) = create_dataset(p_train_val_list, **data_config, p_var="tau")
    train_idx_list = np.arange(args.n_train)
    if args.validate_on_train:
        val_idx_list = np.arange(0, args.n_train + args.n_val)
    else:
        val_idx_list = np.arange(args.n_train, args.n_train + args.n_val)

    input_scale, input_bias = input_norm_and_bias(U_data)
    dim = U_train[0].shape[1]

    # In case we want to start from a grid_search,
    # the first n_grid_x*n_grid_y points are from grid search
    hyp_param_names = [
        "spectral_radius",
        "input_scaling",
        #'leak_factor',
        "parameter_normalization_mean",
        "parameter_normalization_var",
    ]

    hyp_param_scales = ["uniform", "log10", "uniform", "log10"]

    # range for hyperparameters (spectral radius and input scaling)
    spec_in = 0.1
    spec_end = 1.0
    in_scal_in = np.log10(0.1)
    in_scal_end = np.log10(10.0)
    leak_in = 0.1
    leak_end = 1.0
    p_norm_mean_in = -0.4
    p_norm_mean_end = 0.4
    p_norm_var_in = np.log10(0.1)
    p_norm_var_end = np.log10(10.0)
    grid_range = [
        [spec_in, spec_end],
        [in_scal_in, in_scal_end],
        # [leak_in, leak_end],
        [p_norm_mean_in, p_norm_mean_end],
        # [p_norm_mean_in, p_norm_mean_end],
        [p_norm_var_in, p_norm_var_end],
        # [p_norm_var_in, p_norm_var_end],
    ]

    n_grid = [4, 4, 4, 4]
    N_washout = 80
    N_val = 640
    N_fwd = 80
    noise_std = 0
    ESN_dict = {
        "reservoir_size": 1000,
        "dimension": dim,
        "parameter_dimension": 1,
        "reservoir_connectivity": 3,
        "input_normalization": input_scale,
        "input_bias": input_bias,
    }
    min_dict = validate(
        n_grid,
        grid_range,
        hyp_param_names,
        hyp_param_scales,
        n_bo=4,
        n_initial=0,
        n_ensemble=3,
        ESN_dict=ESN_dict,
        U_washout=U_washout_train,
        U=U_train,
        Y=Y_train,
        P_washout=P_washout_train,
        P=P_train,
        n_folds=1,
        N_init_steps=N_washout,
        N_fwd_steps=N_fwd,
        N_washout_steps=N_washout,
        N_val_steps=N_val,
        train_idx_list=train_idx_list,
        val_idx_list=val_idx_list,
        noise_std=noise_std,
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
        "N_fwd": N_fwd,
        "noise_std": noise_std,
        "min_dict": min_dict,
    }

    pp.pickle_file("src/results/validation_run.pickle", results)
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
        "--n_train",
        type=int,
        default=4,
        help="number of training regimes",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=2,
        help="number of validation regimes",
    )
    parser.add_argument(
        "--validate_on_train",
        type=bool,
        default=True,
        help="whether to use the training regimes in validation",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
