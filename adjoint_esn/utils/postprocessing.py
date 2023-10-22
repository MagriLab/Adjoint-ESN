import numpy as np
import yaml

from adjoint_esn.esn import ESN
from adjoint_esn.rijke_esn import RijkeESN
from adjoint_esn.validation import set_ESN


def load_config(experiment_path):
    with open(experiment_path / "config.yml", "r") as file:
        config = yaml.unsafe_load(file)
        return config


def create_ESN(ESN_dict, model_type, hyp_param_names, hyp_param_scales, hyp_params):
    if model_type == "standard":
        my_ESN = ESN(**ESN_dict)
    elif model_type == "rijke":
        my_ESN = RijkeESN(**ESN_dict)
    set_ESN(my_ESN, hyp_param_names, hyp_param_scales, hyp_params)
    return my_ESN


def get_ESN_properties_from_results(config, results, dim):
    # which system parameter is passed to the ESN
    param_vars = config.model.param_vars

    ESN_dict = {
        "reservoir_size": config.model.reservoir_size,
        "parameter_dimension": len(param_vars),
        "reservoir_connectivity": config.model.connectivity,
        "r2_mode": config.model.r2_mode,
        "input_only_mode": config.model.input_only_mode,
        "input_weights_mode": config.model.input_weights_mode,
        "reservoir_weights_mode": config.model.reservoir_weights_mode,
        "tikhonov": config.train.tikhonov,
    }
    if config.model.type == "standard":
        ESN_dict["dimension"] = dim
    elif config.model.type == "rijke":
        ESN_dict["N_g"] = config.simulation.N_g
        ESN_dict["x_f"] = 0.2
        ESN_dict["dt"] = config.model.network_dt
        ESN_dict["u_f_order"] = config.model.u_f_order

    print("System dimension: ", dim)

    top_idx = 0
    hyp_param_names = []
    hyp_params = []
    for name in results.keys():
        if name not in ["training_parameters", "validation_parameters", "f", "tikh"]:
            if results[name].ndim == 2:
                hyp_param_names.extend([name] * results[name].shape[1])
                hyp_params.extend(results[name][top_idx])
            elif results[name].ndim == 1:
                hyp_param_names.extend([name])
                hyp_params.extend([results[name][top_idx]])

    hyp_param_scales = ["uniform"] * len(hyp_param_names)
    return ESN_dict, hyp_param_names, hyp_param_scales, hyp_params
