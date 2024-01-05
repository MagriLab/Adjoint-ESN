import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.random_seed = 41

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.p_list = [2, 4, 6, 8, 10]
    config.simulation.sim_time = 300
    config.simulation.sim_dt = 1e-2
    config.simulation.transient_time = 50
    config.simulation.integrator = "rk4"
    config.simulation.noise_level = 0  # percent

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.network_dt = 1e-2
    config.model.washout_time = 4
    config.model.input_vars = [f"x{i+1}" for i in range(10)]
    config.model.param_vars = ["p"]

    config.model.reservoir_size = 300
    config.model.connectivity = 3
    config.model.r2_mode = False
    config.model.input_only_mode = False
    config.model.input_weights_mode = "sparse_grouped"
    config.model.reservoir_weights_mode = "erdos_renyi1"

    # training configuration
    config.train = ml_collections.ConfigDict()

    config.train.regime_selection = [
        0,
        2,
        4,
    ]  # 'all', integer if random, list if fixed indices
    config.train.time = 10
    config.train.tikhonov = 1e-3

    # validation configuration
    config.val = ml_collections.ConfigDict()

    config.val.validate_on_train = False
    config.val.regime_selection = [1, 3]
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
    config.val.hyperparameters.input_scaling.min = 0.001
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

    # P
    config.val.hyperparameters.parameter_normalization_mean.p = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.p.min = -10.0
    config.val.hyperparameters.parameter_normalization_mean.p.max = 10.0
    config.val.hyperparameters.parameter_normalization_mean.p.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.p = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.p.min = 0.001
    config.val.hyperparameters.parameter_normalization_var.p.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.p.scale = "log10"

    return config
