import argparse
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
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar
from adjoint_esn.validation_v2 import validate

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file("config")
_EXPERIMENT_PATH = myflags.DEFINE_path(
    "experiment_path", None, "Directory to store experiment results"
)
_DATA_DIR = myflags.DEFINE_path(
    "data_dir", None, "Directory to store experiment results"
)
flags.mark_flags_as_required(["config", "experiment_path", "data_dir"])


def save_config():
    with open(FLAGS.experiment_path / "config.yml", "w") as f:
        FLAGS.config.to_yaml(stream=f)


def main(_):
    # setup random seed
    np.random.seed(FLAGS.config.random_seed)

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    config = FLAGS.config
    # check for errors in config before saving
    for param_var in config.model.param_vars:
        if param_var not in ["beta", "tau"]:
            raise ValueError(f"Not valid parameter in {config.model.param_vars}")
    if set(config.val.hyperparameters.parameter_normalization_mean.keys()) != set(
        config.model.param_vars
    ) or set(config.val.hyperparameters.parameter_normalization_var.keys()) != set(
        config.model.param_vars
    ):
        raise ValueError(
            (
                "Parameters passed in the hyperparameter search for mean and variance "
                "are not the same as parameters passed in the model. In the current implementation,"
                "these must be the same, i.e., can't have beta and tau as parameters but only search for "
                "hyperparameters for one of them."
            )
        )

    # Make sure not to get the indices of beta and tau mixed up!
    if len(config.model.param_vars) == 2:
        config.model.param_vars[eParam.beta] = "beta"
        config.model.param_vars[eParam.tau] = "tau"

    save_config()

    # Create the mesh to get the data from
    param_mesh_input = [None] * 2
    param_mesh_input[eParam.beta] = config.simulation.beta_list
    param_mesh_input[eParam.tau] = config.simulation.tau_list
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
    for beta, tau in train_param_list[:, [eParam.beta, eParam.tau]]:
        print(f"beta = {beta}, tau = {tau}")

    val_param_list = param_list[new_list_val_idx]
    print("Validation regimes:")
    for beta, tau in val_param_list[:, [eParam.beta, eParam.tau]]:
        print(f"beta = {beta}, tau = {tau}")

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

    for p in param_list:
        p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
        y_sim, t_sim = pp.load_data(
            beta=p_sim["beta"],
            tau=p_sim["tau"],
            x_f=0.2,
            N_g=config.simulation.N_g,
            sim_time=config.simulation.sim_time,
            sim_dt=config.simulation.sim_dt,
            data_dir=FLAGS.data_dir,
        )

        regime_data = pp.create_dataset(
            y_sim,
            t_sim,
            p_sim,
            network_dt=config.model.network_dt,
            transient_time=config.simulation.transient_time,
            washout_time=config.model.washout_time,
            loop_times=loop_times,
            loop_names=loop_names,
            input_vars=config.model.input_vars,
            output_vars=config.model.output_vars,
            param_vars=config.model.param_vars,
            N_g=config.simulation.N_g,
            u_f_order=config.model.u_f_order,
        )

        for loop_name in loop_names:
            [
                DATA[loop_name][var].append(regime_data[loop_name][var])
                for var in DATA[loop_name].keys()
            ]

    # dimension of the inputs
    dim = DATA["train"]["u"][0].shape[1]

    # how to scale the inputs (applied apart from the input_scaling)
    scale = [None] * 2
    scale[0] = np.zeros(dim)
    scale[1] = np.ones(dim)

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

    # range for hyperparameters
    grid_range = [
        [config.val.hyperparameters[hyp].min, config.val.hyperparameters[hyp].max]
        for hyp in hyp_param_names_
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

    print(grid_range)

    # # n_grid = [4] * len(grid_range)
    # N_washout = int(np.round(data_config["t_washout_len"] / data_config["dt"]))
    # t_val_len = 8
    # N_val = int(np.round(t_val_len / data_config["dt"]))
    # # N_fwd = int(np.round(2 * data_config["t_washout_len"] / data_config["dt"]))
    # N_transient = 0
    # noise_std = args.noise_std
    # n_folds = args.n_folds
    # n_realisations = args.n_realisations
    # ESN_dict = {
    #     "reservoir_size": args.reservoir_size,
    #     "dimension": dim,
    #     "parameter_dimension": n_param,
    #     "reservoir_connectivity": args.connectivity,
    #     "input_normalization": input_scale,
    #     "input_bias": input_bias,
    #     "output_bias": output_bias,
    #     "r2_mode": r2_mode,
    # }
    # for refine in range(args.n_refinements):
    #     print("Starting validation.", flush=True)
    #     print("Grid_range", grid_range, flush=True)
    #     min_dict = validate(
    #         grid_range,
    #         hyp_param_names,
    #         hyp_param_scales,
    #         n_calls=5,
    #         n_initial_points=5,
    #         n_ensemble=1,
    #         ESN_dict=ESN_dict,
    #         tikh=args.tikh,
    #         U_washout_train=U_washout_train,
    #         U_train=U_train,
    #         U_val=U_val,
    #         Y_train=Y_train,
    #         Y_val=Y_val,
    #         P_washout_train=P_washout_train,
    #         P_train=P_train,
    #         P_val=P_val,
    #         n_folds=n_folds,
    #         n_realisations=n_realisations,
    #         N_washout_steps=N_washout,
    #         N_val_steps=N_val,
    #         N_transient_steps=N_transient,
    #         train_idx_list=train_idx_list,
    #         val_idx_list=val_idx_list,
    #     )

    #     results = {
    #         "data_config": data_config,
    #         "p_train_val_list": p_train_val_list,
    #         "train_idx_list": train_idx_list,
    #         "val_idx_list": val_idx_list,
    #         "hyp_param_names": hyp_param_names,
    #         "hyp_param_scales": hyp_param_scales,
    #         "hyp_grid_range": grid_range,
    #         "ESN_dict": ESN_dict,
    #         "N_washout": N_washout,
    #         "N_val": N_val,
    #         "N_transient": N_transient,
    #         "n_folds": n_folds,
    #         "noise_std": noise_std,
    #         "min_dict": min_dict,
    #     }
    #     # datetime object containing current date and time
    #     now = datetime.now()
    #     dt_string = now.strftime("%Y%m%d%H%M%S")
    #     save_path = Path(f"src/results/val_run_{dt_string}.pickle")
    #     print(save_path, flush=True)
    #     pp.pickle_file(str(save_path.absolute()), results)

    #     # REFINEMENT
    #     # range for hyperparameters (spectral radius and input scaling)
    #     spec_in = (spec_in + min_dict["params"][0][0][0]) / 2
    #     spec_end = (spec_end + min_dict["params"][0][0][0]) / 2
    #     in_scal_in = (in_scal_in + np.log10(min_dict["params"][0][0][1])) / 2
    #     in_scal_end = (in_scal_end + np.log10(min_dict["params"][0][0][1])) / 2
    #     leak_in = (leak_in + min_dict["params"][0][0][2]) / 2
    #     leak_end = (leak_end + min_dict["params"][0][0][2]) / 2
    #     p_norm_mean_in = (
    #         p_norm_mean_in + min_dict["params"][0][0][3 : 3 + n_param]
    #     ) / 2
    #     p_norm_mean_end = (
    #         p_norm_mean_end + min_dict["params"][0][0][3 : 3 + n_param]
    #     ) / 2
    #     p_norm_mean = np.array([p_norm_mean_in, p_norm_mean_end]).T
    #     p_norm_var_in = (
    #         p_norm_var_in
    #         + np.log10(min_dict["params"][0][0][3 + n_param : 3 + 2 * n_param])
    #     ) / 2
    #     p_norm_var_end = (
    #         p_norm_var_end
    #         + np.log10(min_dict["params"][0][0][3 + n_param : 3 + 2 * n_param])
    #     ) / 2
    #     p_norm_var = np.array([p_norm_var_in, p_norm_var_end]).T
    #     grid_range = [
    #         [spec_in, spec_end],
    #         [in_scal_in, in_scal_end],
    #         [leak_in, leak_end],
    #     ]
    #     grid_range.extend(p_norm_mean.tolist())
    #     grid_range.extend(p_norm_var.tolist())
    return


if __name__ == "__main__":
    app.run(main)
