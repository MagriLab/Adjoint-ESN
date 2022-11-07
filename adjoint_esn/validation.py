from functools import partial
from itertools import product

import numpy as np
import skopt
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.space import Real

from adjoint_esn.esn import ESN


def create_search_grid(n_param, n_grid, grid_range):
    """Generates a grid for the given grid ranges and number of grid points
    Works for any number of parameters
    Needs return a list of lists for skopt minimize
    Args:
        n_grid: number of grid points for each parameter
        grid_range: search range for each parameter
    Returns:
        search_grid
    """
    # initialise list of np.arrays that holds the range of each parameter
    param_grid = [None] * n_param
    for param_idx in range(n_param):
        param_grid[param_idx] = np.linspace(*grid_range[param_idx], n_grid[param_idx])

    # create the search grid from the parameter grids
    search_grid = product(*param_grid, repeat=1)

    # itertools.product returns a list of tuples
    # turn it to a list of lists
    search_grid = [list(search_list) for search_list in search_grid]

    return search_grid


def create_search_space(n_param, grid_range, param_names):
    search_space = [None] * n_param
    for param_idx in range(n_param):
        search_space[param_idx] = Real(
            *grid_range[param_idx], name=param_names[param_idx]
        )
    return search_space


def run_gp_optimization(
    gp_kernel, val_func, search_space, search_grid, n_total, n_initial
):
    # Gaussian Process reconstruction
    b_e = GPR(
        kernel=gp_kernel,
        normalize_y=True,  # if true mean assumed to be equal to the average of the obj function data, otherwise =0
        n_restarts_optimizer=3,  # number of random starts to find the gaussian process hyperparameters
        noise=1e-10,  # only for numerical stability
        random_state=10,
    )  # seed

    # Bayesian Optimization
    res = skopt.gp_minimize(
        val_func,  # the function to minimize
        search_space,  # the bounds on each dimension of params
        base_estimator=b_e,  # GP kernel
        acq_func="EI",  # the acquisition function
        n_calls=n_total,  # total number of evaluations of f
        x0=search_grid,  # initial grid search points to be evaluated at
        n_random_starts=n_initial,  # the number of additional random initialization points
        n_restarts_optimizer=3,  # number of tries for each acquisition
        random_state=10,  # seed
    )
    return res


def RVC(
    params,
    param_names,
    param_scales,
    my_ESN,
    U_washout,
    U,
    Y,
    data_scale,
    n_folds,
    N_init_steps,
    N_fwd_steps,
    N_washout_steps,
    N_val_steps,
    tikh_hist=None,
    print_flag=False,
):
    """Recycle cross validation method from
    Alberto Racca, Luca Magri:
    Robust Optimization and Validation of Echo State Networks for
    learning chaotic dynamics. Neural Networks 142: 252-268 (2021)
    """
    # set the ESN with the new parameters
    for param_idx, param_name in enumerate(param_names):
        # rescale the parameters according to the given scaling
        if param_scales[param_idx] == "uniform":
            new_param = params[param_idx]
        elif param_scales[param_idx] == "log10":
            new_param = 10 ** params[param_idx]

        setattr(my_ESN, param_name, new_param)

    # first train ESN with the complete data
    X_augmented = my_ESN.reservoir_for_train(U_washout, U, data_scale)

    # train for different tikhonov coefficients
    # since the input data will be the same,
    # we don't rerun the open loop multiple times just to train
    # with different tikhonov coefficients
    tikh_list = [1e-12, 1e-6]
    W_out_list = [None] * len(tikh_list)
    for tikh_idx, tikh in enumerate(tikh_list):
        W_out_list[tikh_idx] = my_ESN.solve_ridge(X_augmented, Y, tikh)

    # save the MSE error with each tikhonov coefficient over all folds
    mse_sum = np.zeros(len(tikh_list))

    # validate with different folds
    for fold in range(n_folds):
        # select washout and validation
        N_steps = N_init_steps + fold * N_fwd_steps
        U_washout_fold = U[N_steps : N_washout_steps + N_steps].copy()
        Y_val = U[
            N_washout_steps + N_steps : N_washout_steps + N_steps + N_val_steps
        ].copy()

        # run washout before closed loop
        x0_fold = my_ESN.run_washout(U_washout_fold, data_scale)

        for tikh_idx in range(len(tikh_list)):
            # set the output weights
            my_ESN.output_weights = W_out_list[tikh_idx]

            # predict output validation in closed-loop
            _, Y_val_pred = my_ESN.closed_loop(x0_fold, N_val_steps - 1, data_scale)

            # add the mse error with this tikh in log10 scale
            mse_sum[tikh_idx] += np.log10(np.mean((Y_val - Y_val_pred) ** 2))

    # find the mean mse
    mse_mean = mse_sum / n_folds

    # select the optimal tikh
    tikh_min_idx = np.argmin(mse_mean)
    tikh_min = tikh_list[tikh_min_idx]
    mse_mean_min = mse_mean[tikh_min_idx]

    # if a tikh hist is provided append to it
    if tikh_hist is not None:
        tikh_hist.append(tikh_min)

    if print_flag:
        for param_name in param_names:
            print(param_name, getattr(my_ESN, param_name))
        print("log10(MSE) = ", mse_mean_min)

    return mse_mean_min


def validate(
    n_grid,
    grid_range,
    param_names,
    param_scales,
    n_bo,
    n_initial,
    n_ensemble,
    ESN_dict,
    U_washout,
    U,
    Y,
    data_scale,
    n_folds,
    N_init_steps,
    N_fwd_steps,
    N_washout_steps,
    N_val_steps,
):

    n_param = len(param_names)  # number of parameters

    # ranges for hyperparameters
    search_space = create_search_space(n_param, grid_range, param_names)

    # grid points to start the search
    search_grid = create_search_grid(n_param, n_grid, grid_range)
    n_total = len(search_grid) + n_bo

    # ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
    gp_kernel = ConstantKernel(
        constant_value=1.0, constant_value_bounds=(1e-1, 3e0)
    ) * Matern(length_scale=[0.2] * n_param, nu=2.5, length_scale_bounds=(5e-2, 1e1))

    # initialize dictionary to hold the minimum parameters and errors
    min_dict = {
        "params": [[None] * n_param] * n_ensemble,
        "tikh": [None] * n_ensemble,
        "f": np.zeros(n_ensemble),
    }

    for i in range(n_ensemble):
        # set the seeds for each realization of ESN
        input_seeds = [4 * i, 4 * i + 1]
        reservoir_seeds = [4 * i + 2, 4 * i + 3]

        # initialize a base ESN object with unit input scaling and spectral radius
        my_ESN = ESN(
            **ESN_dict,
            input_seeds=input_seeds,
            reservoir_seeds=reservoir_seeds,
            verbose=False
        )

        # initialize a tikh history
        tikh_hist = []

        # create the validation function
        # skopt minimize takes functions with only parameters as args
        # we create a partial function passing our ESN and other params
        # which we can then access for training/validation
        val_func = partial(
            RVC,
            param_names=param_names,
            param_scales=param_scales,
            my_ESN=my_ESN,
            U_washout=U_washout,
            U=U,
            Y=Y,
            data_scale=data_scale,
            n_folds=n_folds,
            N_init_steps=N_init_steps,
            N_fwd_steps=N_fwd_steps,
            N_washout_steps=N_washout_steps,
            N_val_steps=N_val_steps,
            tikh_hist=tikh_hist,
        )

        res = run_gp_optimization(
            gp_kernel, val_func, search_space, search_grid, n_total, n_initial
        )

        # save the best parameters
        for param_idx in range(n_param):
            # rescale the parameters according to the given scaling
            if param_scales[param_idx] == "uniform":
                new_param = res.x[param_idx]
            elif param_scales[param_idx] == "log10":
                new_param = 10 ** res.x[param_idx]
            min_dict["params"][i][param_idx] = new_param

        min_iter = np.argmin(res.func_vals)
        min_dict["tikh"][i] = tikh_hist[min_iter]

        min_dict["f"][i] = res.fun

    return min_dict
