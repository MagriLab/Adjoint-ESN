import argparse
import os
import sys
from pathlib import Path

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
# import multiprocessing as mp

import matplotlib.pyplot as plt

from adjoint_esn.esn import ESN
from adjoint_esn.utils import preprocessing as pp
from train_val_rijke_esn import create_dataset


def set_ESN(my_ESN, param_names, params):
    # set the ESN with the new parameters
    for param_name in set(param_names):
        # get the unique strings in the list with set
        # now the indices of the parameters with that name
        # (because ESN has attributes that are set as arrays and not single scalars)
        param_idx_list = np.where(np.array(param_names) == param_name)[0]

        new_param = np.zeros(len(param_idx_list))
        for new_idx in range(len(param_idx_list)):
            param_idx = param_idx_list[new_idx]
            new_param[new_idx] = params[param_idx]

        if len(param_idx_list) == 1:
            new_param = new_param[0]

        setattr(my_ESN, param_name, new_param)
    return


def get_washout_and_true_grad(p_mesh, dt, t_washout_len, input_var, p_var):
    len_p_mesh = len(p_mesh)
    U_washout = [None] * len_p_mesh
    P_washout = [None] * len_p_mesh
    P = [None] * len_p_mesh
    J = np.zeros(len(p_mesh))
    dJ_dbeta = np.zeros(len(p_mesh))
    dJ_dtau = np.zeros(len(p_mesh))
    for p_idx, params in enumerate(p_mesh):
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
        # load data
        data_dict = pp.read_h5(sim_path)

        # get the washout
        if input_var == "eta_mu":
            U_sim = data_dict["y"][:, 0 : 2 * data_dict["N_g"]]
        elif input_var == "eta_mu_v":
            U_sim = data_dict["y"][:, 0 : 2 * data_dict["N_g"] + data_dict["N_c"]]
        elif input_var == "eta_mu_v_tau":
            input_idx = np.arange(2 * data_dict["N_g"])
            input_idx = input_idx.tolist()
            input_idx.append(2 * data_dict["N_g"] + data_dict["N_c"] - 1)
            U_sim = data_dict["y"][:, input_idx]

        # upsample
        data_dt = data_dict["t"][1] - data_dict["t"][0]
        upsample = pp.get_steps(dt, data_dt)
        U = U_sim[::upsample, :]

        # cut the transient
        t_transient_len = data_dict["t_transient"]
        N_transient = pp.get_steps(t_transient_len, dt)
        U = U[N_transient:, :]

        # separate into washout, train, test
        N_washout = pp.get_steps(t_washout_len, dt)
        U_washout[p_idx] = U[0:N_washout, :]
        if p_var == "all":
            train_param_var = params
        elif p_var == "beta":
            train_param_var = beta
        elif p_var == "tau":
            train_param_var = tau
        P_washout[p_idx] = train_param_var * np.ones((len(U_washout[p_idx]), 1))
        P[p_idx] = train_param_var * np.ones((len(U[N_washout:]), 1))

        # get energy
        J[p_idx] = 1 / 4 * np.mean(np.sum(U[:, : 2 * data_dict["N_g"]] ** 2, axis=1))

        # get the gradients
        dJ_dbeta[p_idx] = data_dict["dJ_dbeta"]
        dJ_dtau[p_idx] = data_dict["dJ_dtau"]
    return U_washout, P_washout, P, J, dJ_dbeta, dJ_dtau


def run_esn_grad_num(my_ESN, U_washout, N_t, P_washout, P, N_g):
    # OBJECTIVE SQUARED L2 OF OUTPUT STATES (ACOUSTIC ENERGY)
    X_pred_grad, Y_pred_grad = my_ESN.closed_loop_with_washout(
        U_washout, N_t - 1, P_washout, P
    )

    # calculate gradient for a timeseries, numerical method
    # time averaged objective
    h = 1e-5
    J = 1 / 4 * np.mean(np.sum(Y_pred_grad[:, : 2 * N_g] ** 2, axis=1))

    dJ_dp_num = np.zeros((my_ESN.N_param_dim))
    for i in range(my_ESN.N_param_dim):
        P_left = P.copy()
        P_left[:, i] -= h
        P_right = P.copy()
        P_right[:, i] += h
        _, Y_left = my_ESN.closed_loop(X_pred_grad[0, :], N_t - 1, P_left)
        _, Y_right = my_ESN.closed_loop(X_pred_grad[0, :], N_t - 1, P_right)
        J_left = 1 / 4 * np.mean(np.sum(Y_left[:, : 2 * N_g] ** 2, axis=1))
        J_right = 1 / 4 * np.mean(np.sum(Y_right[:, : 2 * N_g] ** 2, axis=1))
        dJ_dp_num[i] = (J_right - J_left) / (2 * h)
    return J, dJ_dp_num


def get_train_val_plt_idx(
    gamma_list, alpha_plt_list, alpha_col_idx, gamma_col_idx, p_train_list, p_val_list
):
    """Get the indices for plotting train and validation points
    Args:
        alpha: the variable we fix for plotting
        gamma: the other variable
        gamma_list: list of gamma in the mesh that will be plotted
        alpha_plt_list: list of alpha that will be fixed for plotting
        alpha_col_idx, gamma_col_idx: column indices of alpha and gamma variables
        p_train/val_list: list of alpha and gamma for train/val
    """
    gamma_plt_train_idx_list = [None] * len(alpha_plt_list)
    gamma_plt_val_idx_list = [None] * len(alpha_plt_list)
    for alpha_idx, alpha in enumerate(alpha_plt_list):
        alpha_plt_train_idx_list = np.where(
            np.isclose(p_train_list[:, alpha_col_idx], alpha)
        )[0]
        gamma_plt_train_list = p_train_list[alpha_plt_train_idx_list, gamma_col_idx]
        gamma_plt_train_idx_list[alpha_idx] = [
            np.where(np.isclose(gamma_list, gamma_train))[0][0]
            for gamma_train in gamma_plt_train_list
        ]
        alpha_plt_val_idx_list = np.where(
            np.isclose(p_val_list[:, alpha_col_idx], alpha)
        )[0]
        gamma_plt_val_list = p_val_list[alpha_plt_val_idx_list, gamma_col_idx]
        gamma_plt_val_idx_list[alpha_idx] = [
            np.where(np.isclose(gamma_list, gamma_val))[0][0]
            for gamma_val in gamma_plt_val_list
        ]
    return gamma_plt_train_idx_list, gamma_plt_val_idx_list


def get_relative_error(y_true, y_pred):
    """Compute percent relative error"""
    diff = y_true - y_pred
    rel_err = 100 * np.abs(diff) / np.abs(y_true)
    return rel_err


def col_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2, axis=0)


def mse(y_true, y_pred):
    return np.sum(col_mse(y_true, y_pred))


def rmse(y_true, y_pred):
    return np.sum(np.sqrt(col_mse(y_true, y_pred)))


def nrmse(y_true, y_pred):
    col_maxmin = np.max(y_true, axis=0) - np.min(y_true, axis=0)
    col_rmse = np.sqrt(col_mse(y_true, y_pred))
    return np.sum(col_rmse / col_maxmin)


def error_plot(x, mean, std):
    plt.plot(x, mean, "-o", color="tab:blue")
    plt.fill_between(
        x, mean - std, mean + std, alpha=0.2, facecolor="tab:blue", antialiased=True
    )
    plt.yscale("log")
    plt.grid()


def main(args):
    print("Creating mesh.", flush=True)
    # mesh to choose training data from
    if args.p_var == "all":
        beta_list = np.arange(1.2, 2.9, 0.1)
        tau_list = np.arange(0.12, 0.29, 0.01)
    elif args.p_var == "beta":
        beta_list = np.arange(0.3, 5.6, 0.1)
        tau_list = np.array([0.2])
    elif args.p_var == "tau":
        beta_list = np.array([2.5])
        # tau_list = np.arange(0.12, 0.29, 0.01)
        tau_list = np.arange(0.05, 0.40, 0.01)

    beta_mesh, tau_mesh = np.meshgrid(beta_list, tau_list)
    p_mesh = np.hstack([beta_mesh.flatten()[:, None], tau_mesh.flatten()[:, None]])

    print("Loading pickled file.", flush=True)
    # load the pickled results from the hyperparameter search
    hyp_results, hyp_file = pp.unpickle_file(args.hyp_file_name)
    # hyp_results["train_idx_list"] = hyp_results["val_idx_list"]

    print("Creating path.", flush=True)
    # create path to put in the results plot
    results_path = f"src/results_new/{args.hyp_file_name.stem}/"
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    # get the washout data and true gradient
    print("Preparing washout and retrieving the true gradient", flush=True)
    # find out which variables were used for training in order to recreate the dataset
    if "input_var" in hyp_results["data_config"]:
        input_var = hyp_results["data_config"]["input_var"]
    elif (
        "train_var" in hyp_results["data_config"]
    ):  # old validation logs will have this tag
        # adding this bit so we can still handle the old logs
        if hyp_results["data_config"]["train_var"] == "gal":
            # Assumes we used 10 Galerkin modes!!
            if hyp_results["ESN_dict"]["dimension"] == 20:
                input_var = "eta_mu"
            elif hyp_results["ESN_dict"]["dimension"] == 30:
                input_var = "eta_mu_v"
        else:
            raise ValueError(
                "Can't find the gradient from other input variables than the Galerkin variables!"
            )

    (
        U_washout_grad,
        P_washout_grad,
        P_grad,
        J,
        dJ_dbeta,
        dJ_dtau,
    ) = get_washout_and_true_grad(
        p_mesh,
        dt=hyp_results["data_config"]["dt"],
        t_washout_len=hyp_results["data_config"]["t_washout_len"],
        input_var=input_var,
        p_var=args.p_var,
    )

    # reshape such that beta is in x-axis (columns) and tau is in y-axis(rows)
    J = J.reshape(len(tau_list), len(beta_list))
    dJ_dbeta = dJ_dbeta.reshape(len(tau_list), len(beta_list))
    dJ_dtau = dJ_dtau.reshape(len(tau_list), len(beta_list))

    print("Creating training dataset", flush=True)
    # create the same training set as the validation
    (
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
    ) = create_dataset(
        p_list=hyp_results["p_train_val_list"],
        dt=hyp_results["data_config"]["dt"],
        t_washout_len=hyp_results["data_config"]["t_washout_len"],
        t_train_len=hyp_results["data_config"]["t_train_len"],
        t_val_len=hyp_results["data_config"]["t_val_len"],
        grid_upsample=hyp_results["data_config"]["grid_upsample"],
        input_var=input_var,
        p_var=args.p_var,
    )

    # add noise to the data
    len_p_list = len(hyp_results["p_train_val_list"])
    U_washout_train_noisy = [None] * len_p_list
    U_train_noisy = [None] * len_p_list
    for p_idx in range(len_p_list):
        data_std = np.std(U_train[p_idx], axis=0)
        rnd = np.random.RandomState(70 + p_idx)
        mean = np.zeros(U_train[p_idx].shape[1])
        std = (hyp_results["noise_std"] / 100) * data_std
        U_washout_train_noisy[p_idx] = U_washout_train[p_idx] + rnd.normal(
            mean, std, U_washout_train[p_idx].shape
        )
        U_train_noisy[p_idx] = U_train[p_idx] + rnd.normal(
            mean, std, U_train[p_idx].shape
        )

    # choose which beta and tau to plot
    beta_plt_list = np.array([2.5])
    beta_plt_idx_list = [
        np.where(np.isclose(beta_list, beta_plt))[0][0] for beta_plt in beta_plt_list
    ]
    tau_plt_list = np.array([0.2])
    tau_plt_idx_list = [
        np.where(np.isclose(tau_list, tau_plt))[0][0] for tau_plt in tau_plt_list
    ]

    p_train_val_list = hyp_results["p_train_val_list"]
    p_train_list = p_train_val_list[hyp_results["train_idx_list"]]
    p_val_list = p_train_val_list[hyp_results["val_idx_list"]]
    beta_col_idx = 0
    tau_col_idx = 1
    tau_plt_train_idx_list, tau_plt_val_idx_list = get_train_val_plt_idx(
        tau_list, beta_plt_list, beta_col_idx, tau_col_idx, p_train_list, p_val_list
    )
    beta_plt_train_idx_list, beta_plt_val_idx_list = get_train_val_plt_idx(
        beta_list, tau_plt_list, tau_col_idx, beta_col_idx, p_train_list, p_val_list
    )

    # pool = mp.Pool(4)
    for param_idx in range(1):

        # create ESN objects using the hyperparameters
        n_ensemble = 10
        ESN_ensemble = [None] * n_ensemble
        J_esn_ensemble = [None] * n_ensemble
        dJ_dp_esn_ensemble = [None] * n_ensemble

        # Initialize error arrays
        train_nrmse = np.ones((n_ensemble, len_p_list))
        train_rmse = np.ones((n_ensemble, len_p_list))
        val_nrmse = np.ones((n_ensemble, len_p_list))
        val_rmse = np.ones((n_ensemble, len_p_list))

        for e_idx in range(n_ensemble):
            print(f"Training for {e_idx+1}/{n_ensemble}.", flush=True)
            while np.max(train_nrmse[e_idx, :]) >= 1:  # needs some work
                # initialize a base ESN object
                ESN_ensemble[e_idx] = ESN(
                    **hyp_results["ESN_dict"],
                    verbose=False,
                )
                # set the hyperparameters
                params = hyp_results["min_dict"]["params"][0][param_idx]
                set_ESN(
                    ESN_ensemble[e_idx],
                    hyp_results["hyp_param_names"],
                    params,
                )

                # train ESN
                if "tikh" not in hyp_results["min_dict"].keys():
                    tikh = 1e-3
                else:
                    tikh = hyp_results["min_dict"]["tikh"]

                ESN_ensemble[e_idx].train(
                    U_washout_train_noisy,
                    U_train_noisy,
                    Y_train,
                    tikhonov=tikh,
                    P_washout=P_washout_train,
                    P_train=P_train,
                    train_idx_list=hyp_results["train_idx_list"],
                    sample_weights=None,
                )
                # compute the error
                print(f"Computing error.", flush=True)
                for p_idx in range(len_p_list):
                    _, Y_pred_train_ = ESN_ensemble[e_idx].closed_loop_with_washout(
                        U_washout=U_washout_train[p_idx],
                        N_t=len(U_train[p_idx]),
                        P_washout=P_washout_train[p_idx],
                        P=P_train[p_idx],
                    )
                    Y_pred_train = Y_pred_train_[1:, :]
                    _, Y_pred_val_ = ESN_ensemble[e_idx].closed_loop_with_washout(
                        U_washout=U_washout_val[p_idx],
                        N_t=len(U_val[p_idx]),
                        P_washout=P_washout_val[p_idx],
                        P=P_val[p_idx],
                    )
                    Y_pred_val = Y_pred_val_[1:, :]
                    train_nrmse[e_idx, p_idx] = nrmse(Y_train[p_idx], Y_pred_train)
                    val_nrmse[e_idx, p_idx] = nrmse(Y_val[p_idx], Y_pred_val)
                    train_rmse[e_idx, p_idx] = rmse(Y_train[p_idx], Y_pred_train)
                    val_rmse[e_idx, p_idx] = rmse(Y_val[p_idx], Y_pred_val)

                    print("Train nrmse", train_nrmse[e_idx, p_idx])
                    print("Val nrmse", val_nrmse[e_idx, p_idx])

            J_esn = np.zeros((len(p_mesh),))
            dJ_dp_esn = np.zeros(
                (len(p_mesh), hyp_results["ESN_dict"]["parameter_dimension"])
            )

            def run_esn_grad(my_ESN, U_washout, N_t, P_washout, P):
                if args.method == "numerical":
                    return run_esn_grad_num(my_ESN, U_washout, N_t, P_washout, P, N_g=4)

            print(f"Computing gradient.", flush=True)
            for p_idx in range(len(p_mesh)):

                J_esn[p_idx], dJ_dp_esn[p_idx, :] = run_esn_grad(
                    ESN_ensemble[e_idx],
                    U_washout_grad[p_idx],
                    10000,
                    P_washout_grad[p_idx],
                    P_grad[p_idx],
                )
                # dJ_dp_esn[p_idx, :] = pool.apply(
                #    run_esn_grad,
                #    args=(
                #        ESN_ensemble[e_idx],
                #        U_washout_grad[p_idx],
                #        len(P_grad[p_idx]),
                #        P_washout_grad[p_idx],
                #        P_grad[p_idx],
                #    ),
                # )
            J_esn_ensemble[e_idx] = J_esn
            dJ_dp_esn_ensemble[e_idx] = dJ_dp_esn

        # Plot error
        train_nrmse_mean = np.mean(train_nrmse, axis=0)
        train_nrmse_std = np.std(train_nrmse, axis=0)
        train_rmse_mean = np.mean(train_rmse, axis=0)
        train_rmse_std = np.std(train_rmse, axis=0)

        val_nrmse_mean = np.mean(val_nrmse, axis=0)
        val_nrmse_std = np.std(val_nrmse, axis=0)
        val_rmse_mean = np.mean(val_rmse, axis=0)
        val_rmse_std = np.std(val_rmse, axis=0)

        tr_idx_list = hyp_results["train_idx_list"]
        if args.p_var == "beta":
            fig = plt.figure(figsize=(10, 10), constrained_layout=True)
            plt.subplot(2, 2, 1)
            error_plot(
                p_train_val_list[tr_idx_list, beta_col_idx],
                train_rmse_mean[tr_idx_list],
                train_rmse_std[tr_idx_list],
            )
            plt.xlabel("beta")
            plt.title("Train RMSE")

            plt.subplot(2, 2, 2)
            error_plot(
                p_train_val_list[tr_idx_list, beta_col_idx],
                train_nrmse_mean[tr_idx_list],
                train_nrmse_std[tr_idx_list],
            )
            plt.xlabel("beta")
            plt.title("Train NRMSE")

            plt.subplot(2, 2, 3)
            error_plot(
                p_train_val_list[tr_idx_list, beta_col_idx],
                val_rmse_mean[tr_idx_list],
                val_rmse_std[tr_idx_list],
            )
            plt.xlabel("beta")
            plt.title("Val RMSE")

            plt.subplot(2, 2, 4)
            error_plot(
                p_train_val_list[tr_idx_list, beta_col_idx],
                val_nrmse_mean[tr_idx_list],
                val_nrmse_std[tr_idx_list],
            )
            plt.xlabel("beta")
            plt.title("Val NRMSE")
            # save figure
            fig.savefig(results_path / f"error_{param_idx+1}.png")

            # Plot gradient
            if args.p_var == "all":
                dJ_dbeta_esn = [None] * n_ensemble
                dJ_dtau_esn = [None] * n_ensemble
            elif args.p_var == "beta":
                dJ_dbeta_esn = [None] * n_ensemble
            elif args.p_var == "tau":
                dJ_dtau_esn = [None] * n_ensemble

            if args.p_var == "beta":
                for e_idx in range(n_ensemble):
                    J_esn_ensemble[e_idx] = J_esn_ensemble[e_idx].reshape(
                        len(tau_list), len(beta_list)
                    )
                    dJ_dbeta_esn[e_idx] = dJ_dp_esn_ensemble[e_idx]
                    dJ_dbeta_esn[e_idx] = dJ_dbeta_esn[e_idx].reshape(
                        len(tau_list), len(beta_list)
                    )

                for tt, tau_plt_idx in enumerate(tau_plt_idx_list):
                    # GRADIENT FIGURE
                    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
                    dJ_dbeta_esn_mean = np.mean(np.array(dJ_dbeta_esn), axis=0)[
                        tau_plt_idx, :
                    ]
                    dJ_dbeta_esn_std = np.std(np.array(dJ_dbeta_esn), axis=0)[
                        tau_plt_idx, :
                    ]
                    rel_err_grad = get_relative_error(
                        dJ_dbeta[tau_plt_idx, :], dJ_dbeta_esn_mean
                    )

                    tau_name = f"{tau_plt_list[tt]:.2f}"
                    tau_name = tau_name.replace(".", "_")

                    # plot the gradients
                    plt.subplot(1, 2, 1)
                    plt.plot(beta_list, dJ_dbeta[tau_plt_idx, :], "-o", markersize=5)
                    plt.plot(beta_list, dJ_dbeta_esn_mean, "--+", markersize=4)
                    plt.fill_between(
                        beta_list,
                        dJ_dbeta_esn_mean - dJ_dbeta_esn_std,
                        dJ_dbeta_esn_mean + dJ_dbeta_esn_std,
                        alpha=0.2,
                        facecolor="tab:orange",
                        antialiased=True,
                    )
                    plt.plot(
                        beta_list[beta_plt_train_idx_list[tt]],
                        dJ_dbeta[tau_plt_idx, beta_plt_train_idx_list[tt]],
                        color="red",
                        linestyle="None",
                        marker="o",
                        markersize=10,
                        markerfacecolor="None",
                    )
                    plt.plot(
                        beta_list[beta_plt_val_idx_list[tt]],
                        dJ_dbeta[tau_plt_idx, beta_plt_val_idx_list[tt]],
                        color="green",
                        linestyle="None",
                        marker="s",
                        markersize=10,
                        markerfacecolor="None",
                    )
                    plt.xlabel("beta")
                    plt.ylabel("dJ/dbeta")
                    plt.title(f"tau = {tau_plt_list[tt]}")
                    plt.legend(["True", "ESN mean", "ESN std", "Train", "Val"])
                    plt.grid()
                    # plt.ylim([4.5, 6.5])
                    plt.ylim([-1, 10])

                    # plot the relative error
                    plt.subplot(1, 2, 2)
                    plt.plot(beta_list, rel_err_grad, linestyle="-", marker="o")
                    plt.plot(
                        beta_list[beta_plt_train_idx_list[tt]],
                        rel_err_grad[beta_plt_train_idx_list[tt]],
                        color="red",
                        linestyle="None",
                        marker="o",
                        markersize=10,
                        markerfacecolor="None",
                    )
                    plt.plot(
                        beta_list[beta_plt_val_idx_list[tt]],
                        rel_err_grad[beta_plt_val_idx_list[tt]],
                        color="green",
                        linestyle="None",
                        marker="s",
                        markersize=10,
                        markerfacecolor="None",
                    )
                    plt.xlabel("beta")
                    plt.ylabel("% Rel. Err. dJ/dbeta")
                    plt.title(f"tau = {tau_plt_list[tt]}")
                    plt.ylim([0, 100])
                    plt.legend(["Error", "Train", "Val"])
                    plt.grid()

                    # save figure
                    fig.savefig(
                        results_path / f"dJ_dbeta_tau_{tau_name}_{param_idx+1}.png"
                    )
                    plt.close()

        # pool.close()
    hyp_file.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests ESN on Rijke tube data")
    parser.add_argument(
        "--hyp_file_name",
        type=Path,
        default="src/results_new/val_run_20230601093348.pickle",
        help="file that contains the results of the hyperparameter search",
    )
    parser.add_argument(
        "--p_var",
        type=str,
        default="beta",
        help="which parameters to include",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="numerical",
        help="which method to use to calculate the gradient",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
