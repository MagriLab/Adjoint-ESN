import argparse
import os
import sys
from pathlib import Path

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
import multiprocessing as mp
import time

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
            # rescale the parameters according to the given scaling
            param_idx = param_idx_list[new_idx]
            new_param[new_idx] = params[param_idx]

        if len(param_idx_list) == 1:
            new_param = new_param[0]

        setattr(my_ESN, param_name, new_param)
    return


def get_washout_and_true_grad(p_mesh, dt, t_washout_len, p_var):
    len_p_mesh = len(p_mesh)
    U_washout = [None] * len_p_mesh
    P_washout = [None] * len_p_mesh
    P = [None] * len_p_mesh
    dJ_dbeta = np.zeros(len(p_mesh))
    dJ_dtau = np.zeros(len(p_mesh))
    for p_idx, params in enumerate(p_mesh):
        beta = params[0]
        tau = params[1]
        beta_name = f"{beta:.2f}"
        beta_name = beta_name.replace(".", "_")
        tau_name = f"{tau:.2f}"
        tau_name = tau_name.replace(".", "_")
        sim_str = f"src/data/rijke_kings_poly_beta_{beta_name}_tau_{tau_name}.h5"
        # load data
        data_dict = pp.read_h5(sim_str)

        # get the washout
        U_sim = data_dict["y"][:, 0 : 2 * data_dict["N_g"]]
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
        if p_var == "both":
            train_param_var = params
        elif p_var == "beta":
            train_param_var = beta
        elif p_var == "tau":
            train_param_var = tau
        P_washout[p_idx] = train_param_var * np.ones((len(U_washout[p_idx]), 1))
        P[p_idx] = train_param_var * np.ones((len(U[N_washout:]), 1))

        # get the gradients
        dJ_dbeta[p_idx] = data_dict["dJ_dbeta"]
        dJ_dtau[p_idx] = data_dict["dJ_dtau"]
    return U_washout, P_washout, P, dJ_dbeta, dJ_dtau


def run_esn_grad(my_ESN, U_washout, N_t, P_washout, P):
    # OBJECTIVE SQUARED L2 OF OUTPUT STATES (ACOUSTIC ENERGY)
    X_pred_grad, _ = my_ESN.closed_loop_with_washout(U_washout, N_t - 1, P_washout, P)

    # calculate gradient for a timeseries, adjoint method
    # time averaged objective
    X_pred_aug = np.hstack((X_pred_grad[N_t - 1, :], my_ESN.b_out))
    v_prev = (
        (1 / N_t)
        * 1
        / 2
        * np.dot(
            np.dot(X_pred_aug, my_ESN.W_out), my_ESN.W_out[: my_ESN.N_reservoir, :].T
        ).T
    )
    dJ_dp_adj = np.zeros(my_ESN.N_param_dim)
    for i in np.arange(N_t - 1, 0, -1):
        dJ_dp_adj += np.dot(my_ESN.drdp(X_pred_grad[i, :]).toarray().T, v_prev)
        X_pred_aug = np.hstack((X_pred_grad[i - 1, :], my_ESN.b_out))
        dJ_dr = (
            (1 / N_t)
            * 1
            / 2
            * np.dot(
                np.dot(X_pred_aug, my_ESN.W_out),
                my_ESN.W_out[: my_ESN.N_reservoir, :].T,
            ).T
        )
        v = np.dot(my_ESN.jac(X_pred_grad[i, :]).toarray().T, v_prev) + dJ_dr
        v_prev = v
    return dJ_dp_adj


def main(args):
    # Mesh
    beta_list = np.array([2.5])
    tau_list = np.arange(0.15, 0.26, 0.01)
    beta_mesh, tau_mesh = np.meshgrid(beta_list, tau_list)
    p_mesh = np.hstack([beta_mesh.flatten()[:, None], tau_mesh.flatten()[:, None]])

    # load the pickled results from the hyperparameter search
    hyp_results, hyp_file = pp.unpickle_file(args.hyp_file_name)
    p_var = "tau"

    # get the washout data and true gradient
    (
        U_washout_grad,
        P_washout_grad,
        P_grad,
        dJ_dbeta,
        dJ_dtau,
    ) = get_washout_and_true_grad(
        p_mesh,
        dt=hyp_results["data_config"]["dt"],
        t_washout_len=hyp_results["data_config"]["t_washout_len"],
        p_var=p_var,
    )

    # reshape such that beta is in x-axis (columns) and tau is in y-axis(rows)
    dJ_dbeta = dJ_dbeta.reshape(len(tau_list), len(beta_list))
    dJ_dtau = dJ_dtau.reshape(len(tau_list), len(beta_list))

    # create the same training set as the validation
    (
        U_washout_train,
        P_washout_train,
        U_train,
        P_train,
        Y_train,
        t_train,
        _,
    ) = create_dataset(
        hyp_results["p_train_val_list"], **hyp_results["data_config"], p_var=p_var
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

    # create ESN objects using the hyperparameters
    n_ensemble = 1  # len(hyp_results["min_dict"]["f"])
    ESN_ensemble = [None] * n_ensemble
    if p_var == "both":
        dJ_dbeta_esn = [None] * n_ensemble
        dJ_dtau_esn = [None] * n_ensemble
    elif p_var == "beta":
        dJ_dbeta_esn = [None] * n_ensemble
    elif p_var == "tau":
        dJ_dtau_esn = [None] * n_ensemble

    # choose which beta and tau to plot
    beta_plt_list = np.array([2.5])
    beta_plt_idx_list = np.where(beta_list == beta_plt_list)[0]
    tau_plt_list = np.array([0.2])
    tau_plt_idx_list = np.where(tau_list == tau_plt_list)[0]

    pool = mp.Pool(8)
    print(pool)
    for e_idx in range(n_ensemble):
        print(f"Calculating gradient for {e_idx+1}/{n_ensemble}.")
        # initialize a base ESN object
        ESN_ensemble[e_idx] = ESN(
            **hyp_results["ESN_dict"],
            input_seeds=hyp_results["min_dict"]["input_seeds"][e_idx],
            reservoir_seeds=hyp_results["min_dict"]["reservoir_seeds"][e_idx],
            verbose=False,
        )
        # set the hyperparameters
        params = hyp_results["min_dict"]["params"][e_idx]
        set_ESN(
            ESN_ensemble[e_idx],
            hyp_results["hyp_param_names"],
            params,
        )

        # train ESN
        ESN_ensemble[e_idx].train(
            U_washout_train_noisy,
            U_train_noisy,
            Y_train,
            tikhonov=hyp_results["min_dict"]["tikh"][e_idx],
            P_washout=P_washout_train,
            P_train=P_train,
            train_idx_list=hyp_results["train_idx_list"],
        )

        start = time.time()
        dJ_dp_esn = np.zeros(
            (len(p_mesh), hyp_results["ESN_dict"]["parameter_dimension"])
        )
        for p_idx in range(len(p_mesh)):
            dJ_dp_esn[p_idx, :] = pool.apply(
                run_esn_grad,
                args=(
                    ESN_ensemble[e_idx],
                    U_washout_grad[p_idx],
                    len(P_grad[p_idx]),
                    P_washout_grad[p_idx],
                    P_grad[p_idx],
                ),
            )
        end = time.time()
        print("Time for calculating the gradient", end - start)

        if p_var == "both":
            dJ_dbeta_esn[e_idx] = dJ_dp_esn[:, 0]
            dJ_dtau_esn[e_idx] = dJ_dp_esn[:, 1]
            dJ_dbeta_esn[e_idx] = dJ_dbeta_esn[e_idx].reshape(
                len(tau_list), len(beta_list)
            )
            dJ_dtau_esn[e_idx] = dJ_dtau_esn[e_idx].reshape(
                len(tau_list), len(beta_list)
            )
            for bb, beta_plt_idx in enumerate(beta_plt_idx_list):
                fig = plt.figure()
                plt.plot(tau_list, dJ_dbeta[:, beta_plt_idx])
                plt.plot(tau_list, dJ_dbeta_esn[e_idx][:, beta_plt_idx], "--")
                plt.xlabel("tau")
                plt.ylabel("dJ/dbeta")
                plt.title(f"beta = {beta_plt_list[bb]}")
                plt.legend(["True", "ESN"])
                fig.savefig(f"src/results/dJ_dbeta_beta_{beta_plt_list[bb]}.png")
                plt.close()

                fig = plt.figure()
                plt.plot(tau_list, dJ_dtau[:, beta_plt_idx])
                plt.plot(tau_list, dJ_dtau_esn[e_idx][:, beta_plt_idx], "--")
                plt.xlabel("tau")
                plt.ylabel("dJ/dtau")
                plt.title(f"beta = {beta_plt_list[bb]}")
                plt.legend(["True", "ESN"])
                fig.savefig(f"src/results/dJ_dtau_beta_{beta_plt_list[bb]}.png")
                plt.close()

            for tt, tau_plt_idx in enumerate(tau_plt_idx_list):
                fig = plt.figure()
                plt.plot(beta_list, dJ_dbeta[tau_plt_idx, :])
                plt.plot(beta_list, dJ_dbeta_esn[e_idx][tau_plt_idx, :], "--")
                plt.xlabel("beta")
                plt.ylabel("dJ/dbeta")
                plt.title(f"tau = {tau_plt_list[tt]}")
                plt.legend(["True", "ESN"])
                fig.savefig(f"src/results/dJ_dbeta_tau_{tau_plt_list[bb]}.png")
                plt.close()

                fig = plt.figure()
                plt.plot(beta_list, dJ_dtau[tau_plt_idx, :])
                plt.plot(beta_list, dJ_dtau_esn[e_idx][tau_plt_idx, :], "--")
                plt.xlabel("beta")
                plt.ylabel("dJ/dtau")
                plt.title(f"tau = {tau_plt_list[tt]}")
                plt.legend(["True", "ESN"])
                fig.savefig(f"src/results/dJ_dtau_tau_{tau_plt_list[tt]}.png")
                plt.close()

        elif p_var == "beta":
            dJ_dbeta_esn[e_idx] = dJ_dp_esn
            dJ_dbeta_esn[e_idx] = dJ_dbeta_esn[e_idx].reshape(
                len(tau_list), len(beta_list)
            )
            for tt, tau_plt_idx in enumerate(tau_plt_idx_list):
                fig = plt.figure()
                plt.plot(beta_list, dJ_dbeta[tau_plt_idx, :])
                plt.plot(beta_list, dJ_dbeta_esn[e_idx][tau_plt_idx, :], "--")
                plt.xlabel("beta")
                plt.ylabel("dJ/dbeta")
                plt.title(f"tau = {tau_plt_list[tt]}")
                plt.legend(["True", "ESN"])
                fig.savefig(f"src/results/dJ_dbeta_tau_{tau_plt_list[bb]}.png")
                plt.close()

        elif p_var == "tau":
            dJ_dtau_esn[e_idx] = dJ_dp_esn
            dJ_dtau_esn[e_idx] = dJ_dtau_esn[e_idx].reshape(
                len(tau_list), len(beta_list)
            )
            for bb, beta_plt_idx in enumerate(beta_plt_idx_list):
                fig = plt.figure()
                plt.plot(tau_list, dJ_dtau[:, beta_plt_idx])
                plt.plot(tau_list, dJ_dtau_esn[e_idx][:, beta_plt_idx], "--")
                plt.xlabel("tau")
                plt.ylabel("dJ/dtau")
                plt.title(f"beta = {beta_plt_list[bb]}")
                plt.legend(["True", "ESN"])
                fig.savefig(f"src/results/dJ_dtau_beta_{beta_plt_list[bb]}.png")
                plt.close()
    pool.close()
    hyp_file.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests ESN on Rijke tube data")
    parser.add_argument(
        "--hyp_file_name",
        type=Path,
        default="src/results/validation_run.pickle",
        help="file that contains the results of the hyperparameter search",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=3,
        help="number of test regimes",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
