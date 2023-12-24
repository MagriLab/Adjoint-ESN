import os
import sys
from pathlib import Path

import numpy as np
from absl import app, flags
from ml_collections import config_flags

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime

import adjoint_esn.utils.flags as myflags
from adjoint_esn.utils import errors
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import scalers
from adjoint_esn.utils.dynamical_systems import Lorenz63
from adjoint_esn.validation import validate

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config")
_EXPERIMENT_PATH = myflags.DEFINE_path(
    "experiment_path", None, "Directory to store experiment results"
)
flags.mark_flags_as_required(["config"])


def save_config():
    with open(FLAGS.experiment_path / "config.yml", "w") as f:
        FLAGS.config.to_yaml(stream=f)


def check_config_errors(config):
    # check for errors in config before saving
    for param_var in config.model.param_vars:
        if param_var not in ["beta", "rho", "sigma"]:
            raise ValueError(f"Not valid parameter in {config.model.param_vars}")

    if set(config.val.hyperparameters.parameter_normalization_mean.keys()) != set(
        config.model.param_vars
    ) or set(config.val.hyperparameters.parameter_normalization_var.keys()) != set(
        config.model.param_vars
    ):
        raise ValueError(
            (
                "Parameters passed in the hyperparameter search for mean and variance "
                "are not the same as parameters passed in the model. "
                "In the current implementation, these must be the same, "
                "i.e., can not have beta, rho, sigma as parameters but only search for "
                "hyperparameters for one of them."
            )
        )

    if (
        config.model.input_only_mode
        and "spectral_radius" in config.val.hyperparameters.keys()
    ):
        raise ValueError(
            "If the model is in input_only_mode, "
            "spectral radius should not be a hyperparameter."
        )


def main(_):
    # setup random seed
    np.random.seed(FLAGS.config.random_seed)

    config = FLAGS.config
    check_config_errors(config)

    if not FLAGS.experiment_path:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        FLAGS.experiment_path = (
            Path.cwd() / "local_results" / "lorenz63" / f"run_{dt_string}"
        )

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    # redirect print to text file
    orig_stdout = sys.stdout
    f = open(FLAGS.experiment_path / "out.txt", "w")
    sys.stdout = f

    save_config()

    # Create the mesh to get the data from
    eParam = Lorenz63.get_eParamVar()
    param_mesh_input = [None] * 3
    param_mesh_input[eParam.beta] = config.simulation.beta_list
    param_mesh_input[eParam.rho] = config.simulation.rho_list
    param_mesh_input[eParam.sigma] = config.simulation.sigma_list
    param_mesh = pp.make_param_mesh(param_mesh_input)
    len_param_mesh = len(param_mesh)

    # choose the training and validation regimes
    if config.train.regime_selection == "all":
        list_train_idx = range(len_param_mesh)

    if config.val.regime_selection == "all":
        list_val_idx = range(len_param_mesh)

    if isinstance(config.train.regime_selection, int) and isinstance(
        config.val.regime_selection, int
    ):
        list_train_val_idx = np.random.choice(
            len_param_mesh,
            size=config.train.regime_selection + config.val.regime_selection,
            replace=False,
        )
        list_train_idx = list_train_val_idx[: config.train.regime_selection]
        list_val_idx = list_train_val_idx[config.train.regime_selection :]
    elif isinstance(config.train.regime_selection, int):
        list_train_idx = np.random.choice(
            len_param_mesh, size=config.train.regime_selection, replace=False
        )
    elif isinstance(config.val.regime_selection, int):
        list_val_idx = np.random.choice(
            len_param_mesh, size=config.val.regime_selection, replace=False
        )

    if isinstance(config.train.regime_selection, list):
        list_train_idx = list(set(config.train.regime_selection))

    if isinstance(config.val.regime_selection, list):
        list_val_idx = list(set(config.val.regime_selection))

    if config.val.validate_on_train:
        list_val_idx = np.hstack((list_train_idx, list_val_idx))
    # we only want to load the data in the training and validation
    set_train_val_idx = set(np.hstack((list_train_idx, list_val_idx)))
    # make a dictionary that holds the indices
    dict_train_val_idx = dict((k, i) for i, k in enumerate(list(set_train_val_idx)))

    # note: since we know for sure list_train_idx and list_val_idx are in set_train_val_idx,
    # we don't need the set.intersection
    new_list_train_idx = [dict_train_val_idx[inter] for inter in list_train_idx]
    new_list_val_idx = [dict_train_val_idx[inter] for inter in list_val_idx]

    param_list = param_mesh[list(set_train_val_idx)]

    train_param_list = param_list[new_list_train_idx]
    print("Training regimes:")
    for beta, rho, sigma in train_param_list[
        :, [eParam.beta, eParam.rho, eParam.sigma]
    ]:
        print(f"beta = {beta}, rho = {rho}, sigma = {sigma}")

    val_param_list = param_list[new_list_val_idx]
    print("Validation regimes:")
    for beta, rho, sigma in val_param_list[:, [eParam.beta, eParam.rho, eParam.sigma]]:
        print(f"beta = {beta}, rho = {rho}, sigma = {sigma}")

    print("Creating dataset.", flush=True)

    # create the dataset containing the train and validation regimes
    loop_names = ["train", "val"]
    loop_times = [config.train.time, config.val.time]

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

    for p_idx, p in enumerate(param_list):
        p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}
        my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
            Lorenz63,
            p_sim,
            sim_time=config.simulation.sim_time,
            sim_dt=config.simulation.sim_dt,
            random_seed=config.random_seed,
        )

        regime_data = pp.create_dataset_dyn_sys(
            my_sys,
            y_sim,
            t_sim,
            p_sim,
            network_dt=config.model.network_dt,
            transient_time=config.simulation.transient_time,
            washout_time=config.model.washout_time,
            loop_times=loop_times,
            loop_names=loop_names,
            input_vars=config.model.input_vars,
            param_vars=config.model.param_vars,
            noise_level=config.simulation.noise_level,
            random_seed=config.random_seed + p_idx,
        )

        for loop_name in loop_names:
            [
                DATA[loop_name][var].append(regime_data[loop_name][var])
                for var in DATA[loop_name].keys()
            ]

    # dimension of the inputs
    dim = DATA["train"]["u"][0].shape[1]

    print("Dimension", dim)
    print("Creating hyperparameter search range.", flush=True)

    # the hyperparameters related to the mean and normalization must be repeated
    repeated_names = ["parameter_normalization_mean", "parameter_normalization_var"]
    n_param = len(config.model.param_vars)

    hyp_param_names_ = [
        name for name in config.val.hyperparameters.keys() if name not in repeated_names
    ]
    hyp_param_names = hyp_param_names_[:]
    [
        hyp_param_names.extend([name] * n_param)
        for name in repeated_names
        if name in config.val.hyperparameters.keys()
    ]

    # scale for the hyperparameter range
    hyp_param_scales = [
        config.val.hyperparameters[name].scale for name in hyp_param_names_
    ]
    for name in repeated_names:
        if name in hyp_param_names:
            for param in config.model.param_vars:
                hyp_param_scales.extend([config.val.hyperparameters[name][param].scale])

    # range for hyperparameters
    grid_range = [
        [config.val.hyperparameters[name].min, config.val.hyperparameters[name].max]
        for name in hyp_param_names_
    ]
    for name in repeated_names:
        if name in hyp_param_names:
            for param in config.model.param_vars:
                grid_range.extend(
                    [
                        [
                            config.val.hyperparameters[name][param].min,
                            config.val.hyperparameters[name][param].max,
                        ]
                    ]
                )
    # scale the ranges
    for i in range(len(grid_range)):
        for j in range(2):
            scaler = getattr(scalers, hyp_param_scales[i])
            grid_range[i][j] = scaler(grid_range[i][j])

    N_washout = pp.get_steps(config.model.washout_time, config.model.network_dt)
    N_val = pp.get_steps(config.val.fold_time, config.model.network_dt)
    N_transient = 0

    ESN_dict = {
        "dimension": dim,
        "reservoir_size": config.model.reservoir_size,
        "parameter_dimension": n_param,
        "reservoir_connectivity": config.model.connectivity,
        "r2_mode": config.model.r2_mode,
        "input_only_mode": config.model.input_only_mode,
        "input_weights_mode": config.model.input_weights_mode,
        "reservoir_weights_mode": config.model.reservoir_weights_mode,
        "tikhonov": config.train.tikhonov,
    }

    print("Starting validation.", flush=True)
    min_dict = validate(
        grid_range,
        hyp_param_names,
        hyp_param_scales,
        n_calls=config.val.n_calls,
        n_initial_points=config.val.n_initial_points,
        ESN_dict=ESN_dict,
        ESN_type="standard",
        U_washout_train=DATA["train"]["u_washout"],
        U_train=DATA["train"]["u"],
        U_val=DATA["val"]["u"],
        Y_train=DATA["train"]["y"],
        Y_val=DATA["val"]["y"],
        P_washout_train=DATA["train"]["p_washout"],
        P_train=DATA["train"]["p"],
        P_val=DATA["val"]["p"],
        n_folds=config.val.n_folds,
        n_realisations=config.val.n_realisations,
        N_washout_steps=N_washout,
        N_val_steps=N_val,
        N_transient_steps=N_transient,
        train_idx_list=new_list_train_idx,
        val_idx_list=new_list_val_idx,
        p_list=param_list,
        random_seed=config.random_seed,
        error_measure=getattr(errors, config.val.error_measure),
    )

    results = {
        "training_parameters": train_param_list,
        "validation_parameters": val_param_list,
        **min_dict,
    }

    # datetime object containing current date and time
    print(f"Saving results to {FLAGS.experiment_path}.", flush=True)
    pp.pickle_file(FLAGS.experiment_path / "results.pickle", results)

    # close text
    sys.stdout = orig_stdout
    f.close()
    return


if __name__ == "__main__":
    app.run(main)
