import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    config.random_seed = 71

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.N_g = 4
    config.simulation.beta_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    config.simulation.tau_list = [0.1, 0.15, 0.2, 0.25, 0.3]
    config.simulation.sim_time = 800
    config.simulation.sim_dt = 0.001
    config.simulation.transient_time = 300

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.network_dt = 0.01
    config.model.washout_time = 10
    config.model.input_vars = "eta_mu_v_tau"
    config.model.output_vars = "eta_mu"
    config.model.param_vars = ["beta", "tau"]
    config.model.u_f_order = 1

    # training configuration
    config.train = ml_collections.ConfigDict()

    config.train.regime_selection = (
        "all"  # 'all', integer if random, list if fixed indices
    )
    config.train.time = 200

    # validation configuration
    config.val = ml_collections.ConfigDict()

    config.val.regime_selection = 5
    config.val.time = 100

    config.val.hyperparameters = ml_collections.ConfigDict()

    config.val.hyperparameters.spectral_radius = ml_collections.ConfigDict()
    config.val.hyperparameters.spectral_radius.min = 0.1
    config.val.hyperparameters.spectral_radius.max = 1.0
    config.val.hyperparameters.spectral_radius.scale = "uniform"

    config.val.hyperparameters.input_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.input_scaling.min = 0.1
    config.val.hyperparameters.input_scaling.max = 1.0
    config.val.hyperparameters.input_scaling.scale = "uniform"

    config.val.hyperparameters.leak_factor = ml_collections.ConfigDict()
    config.val.hyperparameters.leak_factor.min = 0.1
    config.val.hyperparameters.leak_factor.max = 1.0
    config.val.hyperparameters.leak_factor.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_mean = (
        ml_collections.ConfigDict()
    )

    config.val.hyperparameters.parameter_normalization_mean.beta = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.beta.min = 0.3
    config.val.hyperparameters.parameter_normalization_mean.beta.max = 30.0
    config.val.hyperparameters.parameter_normalization_mean.beta.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_mean.tau = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.tau.min = 0.03
    config.val.hyperparameters.parameter_normalization_mean.tau.max = 3.0
    config.val.hyperparameters.parameter_normalization_mean.tau.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var = ml_collections.ConfigDict()

    config.val.hyperparameters.parameter_normalization_var.beta = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.beta.min = 0.1
    config.val.hyperparameters.parameter_normalization_var.beta.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.beta.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.tau = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.tau.min = 0.01
    config.val.hyperparameters.parameter_normalization_var.tau.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.tau.scale = "uniform"

    return config
