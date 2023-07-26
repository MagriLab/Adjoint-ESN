from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import skopt
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_convergence
from skopt.space import Real

from adjoint_esn.esn import ESN
from adjoint_esn.rijke_esn import RijkeESN
from adjoint_esn.utils import errors, reverse_scalers


def set_ESN(my_ESN, param_names, param_scales, params):
    # set the ESN with the new parameters
    for param_name in set(param_names):
        # get the unique strings in the list with set
        # now the indices of the parameters with that name
        # (because ESN has attributes that are set as arrays when there are more than one parameters
        # and not single scalars)
        param_idx_list = np.where(np.array(param_names) == param_name)[0]

        new_param = np.zeros(len(param_idx_list))
        for new_idx in range(len(param_idx_list)):
            # rescale the parameters according to the given scaling
            param_idx = param_idx_list[new_idx]
            reverse_scaler = getattr(reverse_scalers, param_scales[param_idx])
            new_param[new_idx] = reverse_scaler(params[param_idx])

        if len(param_idx_list) == 1:
            new_param = new_param[0]

        setattr(my_ESN, param_name, new_param)
    return


def create_search_space(n_param, grid_range, param_names):
    search_space = [None] * n_param
    for param_idx in range(n_param):
        search_space[param_idx] = Real(
            *grid_range[param_idx], name=param_names[param_idx]
        )
    return search_space


def run_gp_optimization(val_fun, search_space, n_calls, n_initial_points, rand_state):
    # Bayesian Optimization
    res = skopt.gp_minimize(
        val_fun,  # the function to minimize
        search_space,  # the bounds on each dimension of params
        n_calls=n_calls,  # total number of evaluations of f
        n_initial_points=n_initial_points,  # the number of additional random initialization points
        random_state=rand_state,  # seed
        noise=1e-10,
        n_jobs=1,
    )
    return res


def loop(
    params,
    param_names,
    param_scales,
    ESN_dict,
    tikh,
    U_washout_train,
    U_train,
    U_val,
    Y_train,
    Y_val,
    P_washout_train,
    P_train,
    P_val,
    train_idx_list,
    val_idx_list,
    n_folds,
    n_realisations,
    N_washout,
    N_val,
    N_trans,
    p_list,
    ESN_type="standard",  # "standard" or "rijke"
    error_measure=errors.rmse,
):
    # initialize a base ESN object with unit input scaling and spectral radius
    # seeds not given, so the random generator creates a different seed each run
    global run_idx
    run_idx += 1
    print("--NEW RUN--", run_idx)

    realisation_error = np.zeros(n_realisations)
    for real_idx in range(n_realisations):
        print("Realisation:", real_idx)
        if ESN_type == "standard":
            my_ESN = ESN(
                **ESN_dict,
                verbose=False,
            )
        elif ESN_type == "rijke":
            my_ESN = RijkeESN(
                **ESN_dict,
                verbose=False,
            )

        for param_name, param, param_scale in zip(param_names, params, param_scales):
            reverse_scaler = getattr(reverse_scalers, param_scale)
            print(param_name, reverse_scaler(param))
            if not hasattr(my_ESN, param_name):
                raise ValueError(
                    f"Trying to set a non-existing hyperparameter, {param_name}"
                )

        print("\n")
        # avoid setting some non-existing hyperparameter
        # because setattr won't throw an error, but
        # will simply create a new attribute with that name

        # set the ESN with the given parameters
        set_ESN(my_ESN, param_names, param_scales, params)

        # train ESN
        my_ESN.train(
            U_washout_train,
            U_train,
            Y_train,
            tikhonov=tikh,
            P_washout=P_washout_train,
            P_train=P_train,
            train_idx_list=train_idx_list,
        )

        # divide test set in intervals and predict
        if val_idx_list is None:
            val_idx_list = range(len(U_val))

        val_error = np.zeros(len(val_idx_list))

        for val_idx_idx, val_idx in enumerate(val_idx_list):
            # set the time delay for rijke esn
            if ESN_type == "rijke" and p_list.shape[1] == 2:
                my_ESN.tau = p_list[val_idx, 1]
            print(my_ESN.tau)
            print("Val regime:", val_idx_idx)

            # validate with different folds
            fold_error = np.zeros(n_folds)

            for fold in range(n_folds):
                # select washout and validation
                # start_step = fold * (N_val-N_washout)
                start_step = np.random.randint(
                    len(U_val[val_idx]) - (N_washout + N_val)
                )
                U_washout_fold = U_val[val_idx][
                    start_step : start_step + N_washout
                ].copy()
                P_washout_fold = P_val[val_idx][
                    start_step : start_step + N_washout
                ].copy()
                Y_val_fold = Y_val[val_idx][
                    start_step + N_washout : start_step + N_washout + N_val
                ].copy()
                P_val_fold = P_val[val_idx][
                    start_step + N_washout : start_step + N_washout + N_val
                ].copy()

                # predict output validation in closed-loop
                _, Y_val_pred = my_ESN.closed_loop_with_washout(
                    U_washout=U_washout_fold,
                    N_t=N_val,
                    P_washout=P_washout_fold,
                    P=P_val_fold,
                )
                Y_val_pred = Y_val_pred[1:, :]

                # compute error
                fold_error[fold] = error_measure(
                    Y_val_fold[N_trans:], Y_val_pred[N_trans:]
                )
                # print("Fold:", fold,", fold error: ", fold_error[fold])
            # average over intervals
            val_error[val_idx_idx] = np.mean(fold_error)
            print("Val regime error:", val_error[val_idx_idx])
            # @todo: not only the smallest error, also the val errors should be close to each other
            # also consistent over the folds, can minimize mean and standard deviation or max-min
            # sum()+diff()
        # sum over validation regimes
        realisation_error[real_idx] = np.sum(val_error)
        print("Realisation error:", realisation_error[real_idx])
        print("\n")
    # average over realisations
    error = np.mean(realisation_error)
    print("Run", run_idx, "error:", error)
    print("\n")
    return error


def validate(
    grid_range,
    param_names,
    param_scales,
    n_calls,
    n_initial_points,
    ESN_dict,
    tikh,
    U_washout_train,
    U_train,
    U_val,
    Y_train,
    Y_val,
    P_washout_train,
    P_train,
    P_val,
    n_folds,
    n_realisations,
    N_washout_steps,
    N_val_steps,
    N_transient_steps,
    train_idx_list,
    val_idx_list,
    p_list,
    ESN_type="standard",
    random_seed=10,
    error_measure=errors.rmse,
):

    n_param = len(param_names)  # number of parameters

    # ranges for hyperparameters
    search_space = create_search_space(n_param, grid_range, param_names)

    # initialize dictionary to hold the minimum parameters and errors
    n_top = 5
    min_dict = {
        "f": np.zeros(n_top),
    }
    for param_name in param_names:
        min_dict[param_name] = np.zeros(n_top)

    global run_idx
    run_idx = 0
    # create the validation function
    # skopt minimize takes functions with only parameters as args
    # we create a partial function passing our ESN and other params
    # which we can then access for training/validation
    val_fun = partial(
        loop,
        param_names=param_names,
        param_scales=param_scales,
        ESN_dict=ESN_dict,
        tikh=tikh,
        U_washout_train=U_washout_train,
        U_train=U_train,
        U_val=U_val,
        Y_train=Y_train,
        Y_val=Y_val,
        P_washout_train=P_washout_train,
        P_train=P_train,
        P_val=P_val,
        train_idx_list=train_idx_list,
        val_idx_list=val_idx_list,
        n_folds=n_folds,
        n_realisations=n_realisations,
        N_washout=N_washout_steps,
        N_val=N_val_steps,
        N_trans=N_transient_steps,
        ESN_type=ESN_type,
        p_list=p_list,
        error_measure=error_measure,
    )

    res = run_gp_optimization(
        val_fun, search_space, n_calls, n_initial_points, rand_state=random_seed
    )
    # find the top 5 parameters
    min_idx_list = res.func_vals.argsort()[:n_top]

    # save the best parameters
    for j, min_idx in enumerate(min_idx_list):
        for param_idx, param_name in enumerate(param_names):
            # rescale the parameters according to the given scaling
            reverse_scaler = getattr(reverse_scalers, param_scales[param_idx])
            new_param = reverse_scaler(res.x_iters[min_idx][param_idx])
            min_dict[param_name][j] = new_param

        min_dict["f"][j] = res.func_vals[min_idx]
    min_dict["tikh"] = tikh
    print(min_dict)

    return min_dict
