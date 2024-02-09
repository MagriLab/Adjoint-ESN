import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.random_seed = 41

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.beta_list = [1, 1.5, 2, 2.5, 3]
    config.simulation.rho_list = [30, 35, 40, 45, 50]
    config.simulation.sigma_list = [8, 10, 12, 14, 16]
    config.simulation.sim_time = 200
    config.simulation.sim_dt = 1e-2
    config.simulation.transient_time = 20
    config.simulation.integrator = "rk4"
    config.simulation.noise_level = 0.0  # percent

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.network_dt = 1e-2
    config.model.washout_time = 4
    config.model.input_vars = ["x", "y", "z"]
    config.model.param_vars = ["beta", "rho", "sigma"]

    config.model.reservoir_size = 300
    config.model.connectivity = 3
    config.model.r2_mode = False
    config.model.input_only_mode = False
    config.model.input_weights_mode = "sparse_grouped"
    config.model.reservoir_weights_mode = "erdos_renyi1"
    config.model.normalize_input = "mean_std"
    config.model.add_output_bias = True

    # training configuration
    config.train = ml_collections.ConfigDict()

    config.train.regime_selection = 5  # 'all', integer if random, list if fixed indices
    config.train.time = 10
    config.train.tikhonov = 1e-3

    # validation configuration
    config.val = ml_collections.ConfigDict()

    config.val.validate_on_train = False
    config.val.regime_selection = 3
    config.val.time = 100
    config.val.fold_time = 20
    config.val.n_folds = 1
    config.val.n_realisations = 5
    config.val.n_calls = 5
    config.val.n_initial_points = 2
    config.val.error_measure = "rel_L2"

    # Ranges for the hyperparameters

    # WARNING: when passing a scale other than uniform,
    # the min and max should be the original min and max you want
    # the scaling is done in the script using the scalers in the utils

    config.val.hyperparameters = ml_collections.ConfigDict()

    # SPECTRAL RADIUS
    config.val.hyperparameters.spectral_radius = ml_collections.ConfigDict()
    config.val.hyperparameters.spectral_radius.min = 0.1
    config.val.hyperparameters.spectral_radius.max = 1.0
    config.val.hyperparameters.spectral_radius.scale = "uniform"

    # INPUT SCALING
    config.val.hyperparameters.input_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.input_scaling.min = 0.01
    config.val.hyperparameters.input_scaling.max = 0.1
    config.val.hyperparameters.input_scaling.scale = "log10"

    # LEAK FACTOR
    config.val.hyperparameters.leak_factor = ml_collections.ConfigDict()
    config.val.hyperparameters.leak_factor.min = 0.1
    config.val.hyperparameters.leak_factor.max = 1.0
    config.val.hyperparameters.leak_factor.scale = "uniform"

    # TIKHONOV
    config.val.hyperparameters.tikhonov = ml_collections.ConfigDict()
    config.val.hyperparameters.tikhonov.min = 1e-10
    config.val.hyperparameters.tikhonov.max = 1e-6
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
    config.val.hyperparameters.parameter_normalization_mean.beta.min = 40.0
    config.val.hyperparameters.parameter_normalization_mean.beta.max = 100.0
    config.val.hyperparameters.parameter_normalization_mean.beta.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.beta = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.beta.min = 0.01
    config.val.hyperparameters.parameter_normalization_var.beta.max = 0.1
    config.val.hyperparameters.parameter_normalization_var.beta.scale = "log10"

    # RHO
    config.val.hyperparameters.parameter_normalization_mean.rho = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.rho.min = 40.0
    config.val.hyperparameters.parameter_normalization_mean.rho.max = 100.0
    config.val.hyperparameters.parameter_normalization_mean.rho.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.rho = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.rho.min = 0.001
    config.val.hyperparameters.parameter_normalization_var.rho.max = 0.01
    config.val.hyperparameters.parameter_normalization_var.rho.scale = "log10"

    # # SIGMA
    config.val.hyperparameters.parameter_normalization_mean.sigma = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_mean.sigma.min = 40.0
    config.val.hyperparameters.parameter_normalization_mean.sigma.max = 100.0
    config.val.hyperparameters.parameter_normalization_mean.sigma.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var.sigma = (
        ml_collections.ConfigDict()
    )
    config.val.hyperparameters.parameter_normalization_var.sigma.min = 0.001
    config.val.hyperparameters.parameter_normalization_var.sigma.max = 0.01
    config.val.hyperparameters.parameter_normalization_var.sigma.scale = "log10"

    return config
