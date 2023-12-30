import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.random_seed = 41

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.beta_list = [4 / 3, 8 / 3, 16 / 3]
    config.simulation.rho_list = [30, 35, 40, 45, 50]
    config.simulation.sigma_list = [10]
    config.simulation.sim_time = 300
    config.simulation.sim_dt = 1e-3
    config.simulation.transient_time = 20
    config.simulation.integrator = "rk4"
    config.simulation.noise_level = 0  # percent

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.network_dt = 1e-2
    config.model.washout_time = 4
    config.model.input_vars = ["x", "y", "z"]
    config.model.param_vars = ["beta", "rho"]

    config.model.reservoir_size = 300
    config.model.connectivity = 3
    config.model.r2_mode = False
    config.model.input_only_mode = False
    config.model.input_weights_mode = "sparse_grouped"
    config.model.reservoir_weights_mode = "erdos_renyi1"

    # training configuration
    config.train = ml_collections.ConfigDict()

    config.train.regime_selection = (
        10  # 'all', integer if random, list if fixed indices
    )
    config.train.time = 10
    config.train.tikhonov = 1e-3

    # validation configuration
    config.val = ml_collections.ConfigDict()

    config.val.validate_on_train = False
    config.val.regime_selection = 5
    config.val.time = 100
    config.val.fold_time = 4
    config.val.n_folds = 5
    config.val.n_realisations = 5
    config.val.n_calls = 10
    config.val.n_initial_points = 5
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
    config.val.hyperparameters.input_scaling.max = 10.0
    config.val.hyperparameters.input_scaling.scale = "log10"

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
    config.val.hyperparameters.parameter_normalization_mean.beta.min = -5.0
    config.val.hyperparameters.parameter_normalization_mean.beta.max = 5.0
    config.val.hyperparameters.parameter_normalization_mean.beta.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.beta = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.beta.min = 0.01
    config.val.hyperparameters.parameter_normalization_var.beta.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.beta.scale = "log10"

    # RHO
    config.val.hyperparameters.parameter_normalization_mean.rho = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.rho.min = -10.0
    config.val.hyperparameters.parameter_normalization_mean.rho.max = 10.0
    config.val.hyperparameters.parameter_normalization_mean.rho.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.rho = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.rho.min = 0.01
    config.val.hyperparameters.parameter_normalization_var.rho.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.rho.scale = "log10"

    return config
