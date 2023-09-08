import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.random_seed = 41

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.N_g = 4
    config.simulation.beta_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    config.simulation.tau_list = [0.1, 0.15, 0.2, 0.25, 0.3]
    config.simulation.sim_time = 800
    config.simulation.sim_dt = 1e-3
    config.simulation.transient_time = 300
    config.simulation.noise_level = 0  # percent

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.network_dt = 1e-2
    config.model.washout_time = 10
    config.model.input_vars = "eta_mu_v_tau"
    config.model.output_vars = "eta_mu"
    config.model.param_vars = ["beta"]
    config.model.u_f_order = 1

    config.model.type = "rijke"
    config.model.reservoir_size = 1200
    config.model.connectivity = 0
    config.model.r2_mode = False
    config.model.input_only_mode = True
    config.model.input_weights_mode = "sparse_grouped_rijke_dense"

    # training configuration
    config.train = ml_collections.ConfigDict()

    config.train.regime_selection = (
        10  # 'all', integer if random, list if fixed indices
    )
    config.train.time = 200
    config.train.tikhonov = 1e-3

    # validation configuration
    config.val = ml_collections.ConfigDict()

    config.val.validate_on_train = False
    config.val.regime_selection = 5
    config.val.time = 100
    config.val.fold_time = 8
    config.val.n_folds = 2
    config.val.n_realisations = 1
    config.val.n_calls = 10
    config.val.n_initial_points = 5
    config.val.error_measure = "rel_L2"

    # Ranges for the hyperparameters

    # WARNING: when passing a scale other than uniform,
    # the min and max should be the original min and max you want
    # the scaling is done in the script using the scalers in the utils

    config.val.hyperparameters = ml_collections.ConfigDict()

    # SPECTRAL RADIUS
    # config.val.hyperparameters.spectral_radius = ml_collections.ConfigDict()
    # config.val.hyperparameters.spectral_radius.min = 0.1
    # config.val.hyperparameters.spectral_radius.max = 1.0
    # config.val.hyperparameters.spectral_radius.scale = 'uniform'

    # INPUT SCALING
    config.val.hyperparameters.input_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.input_scaling.min = 0.01
    config.val.hyperparameters.input_scaling.max = 1.0
    config.val.hyperparameters.input_scaling.scale = "log10"

    # U_F_SCALING
    config.val.hyperparameters.u_f_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.u_f_scaling.min = 0.5
    config.val.hyperparameters.u_f_scaling.max = 2.5
    config.val.hyperparameters.u_f_scaling.scale = "uniform"

    # LEAK FACTOR
    config.val.hyperparameters.leak_factor = ml_collections.ConfigDict()
    config.val.hyperparameters.leak_factor.min = 0.1
    config.val.hyperparameters.leak_factor.max = 1.0
    config.val.hyperparameters.leak_factor.scale = "uniform"

    # TIKHONOV
    config.val.hyperparameters.tikhonov = ml_collections.ConfigDict()
    config.val.hyperparameters.tikhonov.min = 1e-4
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
    config.val.hyperparameters.parameter_normalization_var.beta.min = 1.0
    config.val.hyperparameters.parameter_normalization_var.beta.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.beta.scale = "uniform"

    # TAU
    # config.val.hyperparameters.parameter_normalization_mean.tau = (
    #     ml_collections.ConfigDict()
    # )
    # config.val.hyperparameters.parameter_normalization_mean.tau.min = -0.2
    # config.val.hyperparameters.parameter_normalization_mean.tau.max = 0.2
    # config.val.hyperparameters.parameter_normalization_mean.tau.scale = 'uniform'

    # config.val.hyperparameters.parameter_normalization_var.tau = (
    #     ml_collections.ConfigDict()
    # )
    # config.val.hyperparameters.parameter_normalization_var.tau.min = 1.0
    # config.val.hyperparameters.parameter_normalization_var.tau.max = 10.0
    # config.val.hyperparameters.parameter_normalization_var.tau.scale = 'uniform'

    return config
