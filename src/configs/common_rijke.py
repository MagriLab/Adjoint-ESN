import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.random_seed = 41

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.N_g = 4
    config.simulation.beta_list = [6.0, 6.5, 7.0, 7.5, 8.0]
    config.simulation.tau_list = [0.1, 0.15, 0.2, 0.25, 0.3]
    config.simulation.sim_time = 800
    config.simulation.sim_dt = 1e-3
    config.simulation.transient_time = 200
    config.simulation.noise_level = 0.0  # percent

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.network_dt = 1e-2
    config.model.washout_time = 4
    config.model.input_vars = "eta_mu_v_tau"
    config.model.output_vars = "eta_mu"
    config.model.param_vars = ["beta"]
    config.model.u_f_order = 1

    config.model.type = "rijke"
    config.model.reservoir_size = 1200
    config.model.connectivity = 20
    config.model.r2_mode = False
    config.model.input_only_mode = False
    config.model.input_weights_mode = "sparse_grouped_rijke"
    config.model.reservoir_weights_mode = "erdos_renyi1"

    # training configuration
    config.train = ml_collections.ConfigDict()

    config.train.regime_selection = (
        20  # 'all', integer if random, list if fixed indices
    )
    config.train.time = 8
    config.train.tikhonov = 1e-3

    # validation configuration
    config.val = ml_collections.ConfigDict()

    config.val.validate_on_train = False
    config.val.regime_selection = 5
    config.val.time = 200
    config.val.fold_time = 4
    config.val.n_folds = 5
    config.val.n_realisations = 5
    config.val.n_calls = 250
    config.val.n_initial_points = 100
    config.val.error_measure = "rel_L2"

    # Ranges for the hyperparameters

    # WARNING: when passing a scale other than uniform,
    # the min and max should be the original min and max you want
    # the scaling is done in the script using the scalers in the utils

    config.val.hyperparameters = ml_collections.ConfigDict()

    # SPECTRAL RADIUS
    config.val.hyperparameters.spectral_radius = ml_collections.ConfigDict()
    config.val.hyperparameters.spectral_radius.min = 0.01
    config.val.hyperparameters.spectral_radius.max = 1.0
    config.val.hyperparameters.spectral_radius.scale = "log10"

    # INPUT SCALING
    config.val.hyperparameters.input_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.input_scaling.min = 0.01
    config.val.hyperparameters.input_scaling.max = 5.0
    config.val.hyperparameters.input_scaling.scale = "log10"

    # U_F_SCALING
    config.val.hyperparameters.u_f_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.u_f_scaling.min = 0.01
    config.val.hyperparameters.u_f_scaling.max = 5.0
    config.val.hyperparameters.u_f_scaling.scale = "log10"

    # LEAK FACTOR
    config.val.hyperparameters.leak_factor = ml_collections.ConfigDict()
    config.val.hyperparameters.leak_factor.min = 0.01
    config.val.hyperparameters.leak_factor.max = 1.0
    config.val.hyperparameters.leak_factor.scale = "log10"

    # TIKHONOV
    config.val.hyperparameters.tikhonov = ml_collections.ConfigDict()
    config.val.hyperparameters.tikhonov.min = 1e-8
    config.val.hyperparameters.tikhonov.max = 1e-1
    config.val.hyperparameters.tikhonov.scale = "log10"

    # PARAMETER NORMALIZATION
    config.val.hyperparameters.parameter_normalization_mean = (
        ml_collections.ConfigDict()
    )

    config.val.hyperparameters.parameter_normalization_var = ml_collections.ConfigDict()

    # BETA
    config.val.hyperparameters.parameter_normalization_mean.beta = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.beta.min = -10.0
    config.val.hyperparameters.parameter_normalization_mean.beta.max = 10.0
    config.val.hyperparameters.parameter_normalization_mean.beta.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.beta = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.beta.min = 0.01
    config.val.hyperparameters.parameter_normalization_var.beta.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.beta.scale = "log10"

    return config
