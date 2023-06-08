import numpy as np

from adjoint_esn.esn import ESN
from adjoint_esn.rijke_galerkin.solver import Rijke


class RijkeESN(ESN):
    """Creates a specialised Echo State Network for the Rijke tube system
    Args:
        N_g: number of Galerkin modes,
            the inputs must be passed such that the first N_g inputs
            are the velocity amplitudes
    Returns:
        RijkeESN object
    """

    def __init__(
        self,
        reservoir_size,
        N_g,
        x_f,
        tau,
        dt,
        reservoir_connectivity,
        input_normalization,
        parameter_dimension=0,
        parameter_normalization=[np.array([0.0]), np.array([1.0])],
        input_scaling=1.0,
        spectral_radius=1.0,
        leak_factor=1.0,
        input_bias=np.array([]),
        output_bias=np.array([1.0]),
        input_seeds=[None, None, None],
        reservoir_seeds=[None, None],
        verbose=True,
        r2_mode=False,
        input_only_mode=False,
        input_weights_mode="sparse2",
        reservoir_weights_mode="erdos_renyi2",
    ):

        self.verbose = verbose
        self.r2_mode = r2_mode
        self.input_only_mode = input_only_mode

        ## Hyperparameters
        # these should be fixed during initialization and not changed since they affect
        # the matrix dimensions, and the matrices can become incompatible
        self.N_reservoir = reservoir_size

        self.N_g = N_g
        self.x_f = x_f

        self.dt = dt
        self.tau = tau

        self.N_dim = 2 * self.N_g  # output dimension
        self.N_param_dim = parameter_dimension

        self.reservoir_connectivity = reservoir_connectivity

        self.leak_factor = leak_factor

        ## Biases
        self.input_bias = input_bias
        self.output_bias = output_bias

        ## Input normalization
        self.input_normalization = input_normalization
        self.parameter_normalization = parameter_normalization

        ## Weights
        # the object should also store the seeds for reproduction
        # initialise input weights
        self.W_in_seeds = input_seeds
        self.W_in_shape = (
            self.N_reservoir,
            self.N_dim + len(self.input_bias) + 1 + self.N_param_dim,
        )
        # N_dim+length of input bias because we augment the inputs with a bias
        # if no bias, then this will be + 0
        self.input_weights_mode = input_weights_mode
        self.input_weights = self.generate_input_weights()
        self.input_scaling = input_scaling
        # input weights are automatically scaled if input scaling is updated

        # initialise reservoir weights
        self.W_seeds = reservoir_seeds
        self.W_shape = (self.N_reservoir, self.N_reservoir)
        self.reservoir_weights_mode = reservoir_weights_mode
        self.reservoir_weights = self.generate_reservoir_weights()
        self.spectral_radius = spectral_radius
        # reservoir weights are automatically scaled if spectral radius is updated

        # initialise output weights
        self.W_out_shape = (self.N_reservoir + len(self.output_bias), self.N_dim)
        # N_reservoir+length of output bias because we augment the outputs with a bias
        # if no bias, then this will be + 0
        self.output_weights = np.zeros(self.W_out_shape)

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, new_tau):
        self._tau = new_tau
        self.N_tau = int(self.tau / self.dt)
        return

    def closed_loop(self, X_tau, N_t, P=None):
        """Advances ESN in closed-loop.
        Args:
            N_t: number of time steps
            x0: initial reservoir state
            P: parameter time series (N_t x N_param_dim)
        Returns:
            X: time series of the reservoir states (N_t x N_reservoir)
            Y: time series of the output (N_t x N_dim)
        """
        # create an empty matrix to hold the reservoir states in time
        X = np.empty((self.N_tau + N_t + 1, self.N_reservoir))
        # create an empty matrix to hold the output states in time
        Y = np.empty((self.N_tau + N_t + 1, self.N_dim))

        for t in range(0, self.N_tau + 1):
            # initialize with the given initial reservoir states
            X[t, :] = X_tau[t, :].copy()

            # augment the reservoir states with the bias
            if self.r2_mode:
                x_tau_2 = X_tau[t, :].copy()
                x_tau_2[1::2] = x_tau_2[1::2] ** 2
                x_tau_augmented = np.hstack((x_tau_2, self.b_out))
            else:
                x_tau_augmented = np.hstack((X_tau[t, :], self.b_out))

            # initialise with the calculated output states
            Y[t, :] = np.dot(x_tau_augmented, self.W_out)

        # step in time
        for n in range(self.N_tau + 1, self.N_tau + N_t + 1):
            # update the reservoir with the feedback from the output
            y_tau = Y[n - 1 - self.N_tau, :]
            eta_tau = y_tau[0 : self.N_g]
            velocity_f_tau = Rijke.toVelocity(
                N_g=self.N_g, eta=eta_tau, x=np.array([self.x_f])
            )
            y_augmented = np.hstack((Y[n - 1, :], velocity_f_tau))

            if self.N_param_dim > 0:
                X[n, :] = self.step(X[n - 1, :], y_augmented, P[n - 1, :])
            else:
                X[n, :] = self.step(X[n - 1, :], y_augmented)

            # augment the reservoir states with bias
            # replaces r with r^2 if even, r otherwise
            if self.r2_mode:
                X2 = X[n, :].copy()
                X2[1::2] = X2[1::2] ** 2
                x_augmented = np.hstack((X2, self.b_out))
            else:
                x_augmented = np.hstack((X[n, :], self.b_out))

            # update the output with the reservoir states
            Y[n, :] = np.dot(x_augmented, self.W_out)

        return X[self.N_tau :, :], Y[self.N_tau :, :]

    def closed_loop_with_washout(self, U_washout, N_t, P_washout=None, P=None):
        # Wash-out phase to get rid of the effects of reservoir states initialised as zero
        # initialise the reservoir states before washout
        x0_washout = np.zeros(self.N_reservoir)

        # let the ESN run in open-loop for the wash-out
        # get the initial reservoir to start the actual open/closed-loop,
        # which is the last reservoir state
        X_tau = self.open_loop(x0=x0_washout, U=U_washout, P=P_washout)[
            -self.N_tau - 1 :, :
        ]
        P = np.vstack((P_washout[-self.N_tau - 1 :, :], P))
        X, Y = self.closed_loop(X_tau, N_t=N_t, P=P)
        return X, Y
