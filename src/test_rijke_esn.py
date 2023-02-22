import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
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


def main(args):
    # Mesh
    beta_list = np.array([2.5])
    tau_list = np.arange(0.15, 0.26, 0.01)
    beta_mesh, tau_mesh = np.meshgrid(beta_list, tau_list)
    p_mesh = np.hstack([beta_mesh.flatten()[:, None], tau_mesh.flatten()[:, None]])

    # load the pickled results from the hyperparameter search
    hyp_results, hyp_file = pp.unpickle_file(args.hyp_file_name)

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
        hyp_results["p_train_val_list"], **hyp_results["data_config"], p_var="tau"
    )

    # create test set from different parameters in the mesh
    p_valid_test = []
    for p in p_mesh:
        matched_p = False
        for p_train in hyp_results["p_train_val_list"]:
            if all(p == p_train):
                matched_p = True
        if matched_p == False:
            p_valid_test.append(p)
    p_valid_test = np.array(p_valid_test)

    rnd = np.random.RandomState(seed=10)
    test_idx_list = rnd.choice(len(p_valid_test), size=args.n_test, replace=False)
    p_test_list = p_valid_test[test_idx_list, :]

    (
        U_washout_test,
        P_washout_test,
        U_test,
        P_test,
        Y_test,
        t_test,
        _,
    ) = create_dataset(p_test_list, **hyp_results["data_config"], p_var="tau")

    # add noise to the data
    len_p_list = len(hyp_results["p_train_val_list"])
    len_p_test_list = len(p_test_list)
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
    n_ensemble = len(hyp_results["min_dict"]["f"])
    ESN_ensemble = [None] * n_ensemble

    plt_idx = [0]

    for e_idx in range(n_ensemble):
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

        # plot prediction of train
        Y_pred_train = [None] * len_p_list
        for p_idx in range(len_p_list):
            _, Y_pred_train_ = ESN_ensemble[e_idx].closed_loop_with_washout(
                U_washout=U_washout_train[p_idx],
                N_t=len(U_train[p_idx]),
                P_washout=P_washout_train[p_idx],
                P=P_train[p_idx],
            )
            Y_pred_train[p_idx] = Y_pred_train_[1:, :]
            fig = plt.figure(figsize=(8, 2))
            for j in plt_idx:
                plt.plot(t_train[p_idx], Y_train[p_idx][:, j])
                plt.plot(t_train[p_idx], Y_pred_train[p_idx][:, j], "--")
                plt.title(f"Train & Val p = {hyp_results['p_train_val_list'][p_idx]}")
                plt.xlabel("t")
                plt.ylabel(f"q_{j}")
                plt.legend(["True", "ESN"])
            fig.savefig(f"src/results/train_val_ESN_{e_idx}_p_{p_idx}.png")
            plt.close()

        # plot prediction of test
        Y_pred_test = [None] * len_p_test_list
        for p_idx in range(len_p_test_list):
            _, Y_pred_test_ = ESN_ensemble[e_idx].closed_loop_with_washout(
                U_washout=U_washout_test[p_idx],
                N_t=len(U_test[p_idx]),
                P_washout=P_washout_test[p_idx],
                P=P_test[p_idx],
            )
            Y_pred_test[p_idx] = Y_pred_test_[1:, :]
            fig = plt.figure(figsize=(8, 2))
            for j in plt_idx:
                plt.plot(t_test[p_idx], Y_test[p_idx][:, j])
                plt.plot(t_test[p_idx], Y_pred_test[p_idx][:, j], "--")
                plt.title(f"Test p = {p_test_list[p_idx]}")
                plt.xlabel("t")
                plt.ylabel(f"q_{j}")
                plt.legend(["True", "ESN"])
            fig.savefig(f"src/results/test_ESN_{e_idx}_p_{p_idx}.png")
            plt.close()

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
