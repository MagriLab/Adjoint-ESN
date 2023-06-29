import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge

import adjoint_esn.generate_input_weights as generate_input_weights
from adjoint_esn.input_only_esn import InputOnlyESN
from adjoint_esn.rijke_galerkin.solver import Rijke


class RijkeInputOnlyESN(InputOnlyESN):
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
        input_normalization,
        parameter_dimension=0,
        parameter_normalization=[np.array([0.0]), np.array([1.0])],
        input_scaling=1.0,
        u_f_scaling=1.0,
        u_f_order=1,
        leak_factor=1.0,
        input_bias=np.array([]),
        output_bias=np.array([1.0]),
        input_seeds=[None, None, None],
        reservoir_seeds=[None, None],
        verbose=True,
        r2_mode=False,
        input_weights_mode="sparse_grouped",
        reservoir_weights_mode="erdos_renyi2",
        step_mode="step1",
    ):

        self.verbose = verbose
        self.r2_mode = r2_mode
        self.step_mode = step_mode

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
        self.u_f_order = u_f_order

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
            self.N_dim + len(self.input_bias) + self.u_f_order + self.N_param_dim,
        )
        # N_dim+length of input bias because we augment the inputs with a bias
        # if no bias, then this will be + 0
        self.input_weights_mode = input_weights_mode
        self.input_weights = self.generate_input_weights()
        self.input_scaling = input_scaling
        self.u_f_scaling = u_f_scaling
        # input weights are automatically scaled if input scaling is updated

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

    @property
    def x_f(self):
        return self._x_f

    @x_f.setter
    def x_f(self, new_x_f):
        self._x_f = new_x_f
        return

    def generate_input_weights(self):
        if self.input_weights_mode == "sparse_grouped_rijke":
            return generate_input_weights.sparse_grouped_rijke(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds, self.u_f_order
            )
        elif self.input_weights_mode == "sparse_grouped_rijke_dense":
            return generate_input_weights.sparse_grouped_rijke_dense(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds, self.u_f_order
            )
        else:
            super().generate_input_weights()

    @property
    def input_scaling(self):
        return self.sigma_in

    @input_scaling.setter
    def input_scaling(self, new_input_scaling):
        """Setter for the input scaling, if new input scaling is given,
        then the input weight matrix is also updated
        """
        if hasattr(self, "sigma_in"):
            # rescale the input matrix
            self.W_in[:, : -self.N_param_dim - self.u_f_order] = (
                1 / self.sigma_in
            ) * self.W_in[:, : -self.N_param_dim - self.u_f_order]
        # set input scaling
        self.sigma_in = new_input_scaling
        if self.verbose:
            print("Input weights are rescaled with the new input scaling.")
        self.W_in[:, : -self.N_param_dim - self.u_f_order] = (
            self.sigma_in * self.W_in[:, : -self.N_param_dim - self.u_f_order]
        )
        return

    @property
    def u_f_scaling(self):
        return self.sigma_u_f

    @u_f_scaling.setter
    def u_f_scaling(self, new_u_f_scaling):
        """Setter for the u_f(t-tau) scaling, if new u_f(t-tau) scaling is given,
        then the input weight matrix is also updated
        """
        if hasattr(self, "sigma_u_f"):
            # rescale the input matrix
            self.W_in[:, -self.N_param_dim - self.u_f_order : -self.N_param_dim] = (
                1 / self.sigma_u_f
            ) * self.W_in[:, -self.N_param_dim - self.u_f_order : -self.N_param_dim]
        # set input scaling
        self.sigma_u_f = new_u_f_scaling
        if self.verbose:
            print(
                "Input weights of u_f(t-tau) are rescaled with the new input scaling."
            )
        self.W_in[:, -self.N_param_dim - self.u_f_order : -self.N_param_dim] = (
            self.sigma_u_f
            * self.W_in[:, -self.N_param_dim - self.u_f_order : -self.N_param_dim]
        )
        return

    def closed_loop(self, X_tau, Y_tau, N_t, P=None):
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
            if self.step_mode == "step1":
                Y[t, :] = np.dot(x_tau_augmented, self.W_out)
            elif self.step_mode == "step2":
                Y[t, :] = Y_tau[t, :] + np.dot(x_tau_augmented, self.W_out)

        # step in time
        for n in range(self.N_tau + 1, self.N_tau + N_t + 1):
            # update the reservoir with the feedback from the output
            y_tau = Y[n - 1 - self.N_tau, :]
            eta_tau = y_tau[0 : self.N_g]
            velocity_f_tau = Rijke.toVelocity(
                N_g=self.N_g, eta=eta_tau, x=np.array([self.x_f])
            )
            y_augmented = Y[n - 1, :]
            for order in range(self.u_f_order):
                y_augmented = np.hstack((y_augmented, velocity_f_tau ** (order + 1)))

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
            if self.step_mode == "step1":
                Y[n, :] = np.dot(x_augmented, self.W_out)
            elif self.step_mode == "step2":
                Y[n, :] = Y[n - 1, :] + np.dot(x_augmented, self.W_out)

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
        Y_tau = U_washout[-self.N_tau - 1 :, : 2 * self.N_g]
        P = np.vstack((P_washout[-self.N_tau - 1 :, :], P))
        X, Y = self.closed_loop(X_tau, Y_tau, N_t=N_t, P=P)
        return X, Y

    # def solve_ridge(self, X, Y, tikh, sample_weights = None):
    #     # solve ridge for each row, mask the uncoupled modes
    #     # works only for the sparse case and u_f_order = 1
    #     W_out = np.zeros(self.W_out_shape)
    #     for y_idx in range(Y.shape[1]):
    #         reg = Ridge(alpha=tikh, fit_intercept=False)
    #         if y_idx < self.N_g:
    #             eta_j_idx = np.where(self.W_in[:,y_idx].toarray()  != 0)[0]
    #             mu_j_idx = np.where(self.W_in[:,self.N_g+y_idx].toarray()  != 0)[0]
    #             unmasked_idx = np.hstack((eta_j_idx,mu_j_idx))
    #         elif y_idx >= self.N_g:
    #             eta_j_idx = np.where(self.W_in[:,y_idx-self.N_g].toarray() != 0)[0]
    #             mu_j_idx = np.where(self.W_in[:,y_idx].toarray() != 0)[0]
    #             u_f_idx =  np.where(self.W_in[:,-self.N_param_dim-1].toarray() != 0)[0]
    #             unmasked_idx = np.hstack((eta_j_idx,mu_j_idx))
    #             unmasked_idx = np.hstack((unmasked_idx,u_f_idx))

    #         X_new = X[:,unmasked_idx]
    #         reg.fit(X_new, Y[:, y_idx], sample_weight=None)
    #         W_out[unmasked_idx, y_idx
    #             ] = (
    #                 reg.coef_.T
    #             )

    #     return W_out
