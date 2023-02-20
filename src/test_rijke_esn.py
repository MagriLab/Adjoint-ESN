import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
from adjoint_esn.esn import ESN
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.validation import set_ESN
from train_val_rijke_esn import create_dataset, input_norm_and_bias


def main(args):
    # load the pickled results from the hyperparameter search
    hyp_results, hyp_file = pp.unpickle_file(args.hyp_file_name)

    # create the same training set as the validation
    (
        U_washout_train,
        P_washout_train,
        U_train,
        P_train,
        Y_train,
        U_data,
    ) = create_dataset(hyp_results["p_train_val_list"], **hyp_results["data_config"])
    input_scale, input_bias = input_norm_and_bias(U_data)
    dim = U_train[0].shape[1]

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
    n_ensemble = len(hyp_results["min_dict"]["f"])
    ESN_ensemble = [None] * n_ensemble
    for e_idx in range(n_ensemble):
        # initialize a base ESN object
        ESN_ensemble[e_idx] = ESN(
            **hyp_results["ESN_dict"],
            input_seeds=hyp_results["min_dict"]["input_seeds"][e_idx],
            reservoir_seeds=hyp_results["min_dict"]["reservoir_seeds"][e_idx],
            verbose=False
        )
        # set the hyperparameters
        params = hyp_results["min_dict"]["params"][e_idx]
        set_ESN(
            ESN_ensemble[e_idx],
            hyp_results["param_names"],
            hyp_results["param_scales"],
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

    hyp_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests ESN on Rijke tube data")
    parser.add_argument(
        "--hyp_file_name",
        type=Path,
        default="src/results/validation_run.pickle",
        help="file that contains the results of the hyperparameter search",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
