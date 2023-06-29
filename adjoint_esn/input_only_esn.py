import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge

import adjoint_esn.generate_input_weights as generate_input_weights


class InputOnlyESN:
    def __init__(
        self,
        dimension,
        reservoir_size,
        input_normalization,
        parameter_dimension=0,
        parameter_normalization=[np.array([0.0]), np.array([1.0])],
        input_scaling=1.0,
        leak_factor=1.0,
        input_bias=np.array([]),
        output_bias=np.array([]),
        input_seeds=[None, None, None],
        input_weights_mode="sparse_grouped",
        verbose=True,
        r2_mode=False,
        step_mode="step1",
    ):
        """Creates an Input Only Echo State Network with the given parameters
        Args:
            dimension: dimension of the state space of the input and output
                #@todo: separate input and output dimensions
            input_normalization: normalization applied to the input before activation
                tuple with (mean, norm) such that u is updated as (u-mean)/norm
            parameter_dimension: dimension of the system's bifurcation parameters
            input_scaling: scaling applied to the input weights matrix
            leak_factor: factor for the leaky integrator
                if set to 1 (default), then no leak is applied
            input_bias: bias that is augmented to the input vector
            output_bias: bias that is augmented to the output vector
            input_seeds: seeds to generate input weights matrix
            input_weights_mode: how to generate the input weights
            verbose: whether the ESN prints statements when changing hyperparameters
            r2_mode: whether the even rows will be squared in the output
        Returns:
            Input only ESN object

        """
        self.verbose = verbose
        self.r2_mode = r2_mode
        self.step_mode = step_mode

        ## Hyperparameters
        # the dimensions should be fixed during initialization and not changed since they affect
        # the matrix dimensions, and the matrices can become incompatible
        self.N_reservoir = reservoir_size
        self.N_dim = dimension
        self.N_param_dim = parameter_dimension

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
            self.N_dim + len(self.input_bias) + self.N_param_dim,
        )
        # N_dim+length of input bias because we augment the inputs with a bias
        # if no bias, then this will be + 0
        self.input_weights_mode = input_weights_mode
        self.input_weights = self.generate_input_weights()
        self.input_scaling = input_scaling
        # input weights are automatically scaled if input scaling is updated

        # initialise output weights
        self.W_out_shape = (self.N_reservoir + len(self.output_bias), self.N_dim)
        # N_reservoir+length of output bias because we augment the outputs with a bias
        # if no bias, then this will be + 0
        self.output_weights = np.zeros(self.W_out_shape)

    @property
    def leak_factor(self):
        return self.alpha

    @leak_factor.setter
    def leak_factor(self, new_leak_factor):
        # set leak factor
        if new_leak_factor < 0 or new_leak_factor > 1:
            raise ValueError("Leak factor must be between 0 and 1 (including).")
        self.alpha = new_leak_factor
        return

    @property
    def input_normalization(self):
        return self.norm_in

    @input_normalization.setter
    def input_normalization(self, new_input_normalization):
        self.norm_in = new_input_normalization
        if self.verbose:
            print("Input normalization is changed, training must be done again.")

    @property
    def parameter_normalization_mean(self):
        return self.norm_p[0]

    @parameter_normalization_mean.setter
    def parameter_normalization_mean(self, new_parameter_normalization_mean):
        self.norm_p[0] = new_parameter_normalization_mean
        if self.verbose:
            print("Parameter normalization is changed, training must be done again.")

    @property
    def parameter_normalization_var(self):
        return self.norm_p[1]

    @parameter_normalization_var.setter
    def parameter_normalization_var(self, new_parameter_normalization_var):
        self.norm_p[1] = new_parameter_normalization_var
        if self.verbose:
            print("Parameter normalization is changed, training must be done again.")

    @property
    def parameter_normalization(self):
        return self.norm_p

    @parameter_normalization.setter
    def parameter_normalization(self, new_parameter_normalization):
        self.norm_p = new_parameter_normalization
        if self.verbose:
            print("Parameter normalization is changed, training must be done again.")

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
            self.W_in = (1 / self.sigma_in) * self.W_in
        # set input scaling
        self.sigma_in = new_input_scaling
        if self.verbose:
            print("Input weights are rescaled with the new input scaling.")
        self.W_in = self.sigma_in * self.W_in
        return

    @property
    def input_weights(self):
        return self.W_in

    @input_weights.setter
    def input_weights(self, new_input_weights):
        # first check the dimensions
        if new_input_weights.shape != self.W_in_shape:
            raise ValueError(
                f"The shape of the provided input weights does not match with the network, {new_input_weights.shape} != {self.W_in_shape}"
            )

        # set the new input weights
        self.W_in = new_input_weights

        # set the input scaling to 1.0
        if self.verbose:
            print("Input scaling is set to 1, set it separately if necessary.")
        self.sigma_in = 1.0
        return

    @property
    def input_bias(self):
        return self.b_in

    @input_bias.setter
    def input_bias(self, new_input_bias):
        self.b_in = new_input_bias
        return

    @property
    def output_bias(self):
        return self.b_out

    @output_bias.setter
    def output_bias(self, new_output_bias):
        self.b_out = new_output_bias
        return

    @property
    def output_weights(self):
        return self.W_out

    @output_weights.setter
    def output_weights(self, new_output_weights):
        # first check the dimensions
        if new_output_weights.shape != self.W_out_shape:
            raise ValueError(
                f"The shape of the provided output weights does not match with the network,"
                f"{new_output_weights.shape} != {self.W_out_shape}"
            )
        # set the new reservoir weights
        self.W_out = new_output_weights
        return

    @property
    def tikhonov(self):
        return self.tikh

    @tikhonov.setter
    def tikhonov(self, new_tikhonov):
        # set tikhonov coefficient
        if new_tikhonov <= 0:
            raise ValueError("Tikhonov coefficient must be greater than 0.")
        self.tikh = new_tikhonov
        return

    def generate_input_weights(self):
        if self.input_weights_mode == "sparse_random":
            return generate_input_weights.sparse_random(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds
            )
        elif self.input_weights_mode == "sparse_grouped":
            return generate_input_weights.sparse_grouped(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds
            )
        elif self.input_weights_mode == "dense":
            return generate_input_weights.dense(self.W_in_shape, self.W_in_seeds)

    def get_next_step(self, u_augmented):
        x_tilde = np.tanh(self.W_in.dot(u_augmented))
        return x_tilde

    def step1(self, x_prev, u, p=None):
        """Advances ESN time step.
        Args:
            x_prev: reservoir state in the previous time step (n-1)
            u: input in this time step (n)
            p: systems bifucation parameters vector
        Returns:
            x_next: reservoir state in this time step (n)
        """
        # normalise the input
        u_norm = (u - self.norm_in[0]) / self.norm_in[1]
        # we normalize here, so that the input is normalised
        # in closed-loop run too?

        # augment the input with the input bias
        u_augmented = np.hstack((u_norm, self.b_in))

        # augment the input with the parameters
        if self.N_param_dim > 0:
            u_augmented = np.hstack(
                (u_augmented, (p - self.norm_p[0]) / self.norm_p[1])
            )

        # update the reservoir
        x_tilde = self.get_next_step(u_augmented)

        # apply the leaky integrator
        x = (1 - self.alpha) * x_prev + self.alpha * x_tilde
        return x

    def step2(self, x_prev, u, p=None):
        """Advances ESN time step.
        Args:
            x_prev: reservoir state in the previous time step (n-1)
            u: input in this time step (n)
            p: systems bifucation parameters vector
        Returns:
            x_next: reservoir state in this time step (n)
        """
        # normalise the input
        u_norm = (u - self.norm_in[0]) / self.norm_in[1]
        # we normalize here, so that the input is normalised
        # in closed-loop run too?

        # augment the input with the input bias
        u_augmented = np.hstack((u_norm, self.b_in))

        # augment the input with the parameters
        if self.N_param_dim > 0:
            u_augmented = np.hstack(
                (u_augmented, (p - self.norm_p[0]) / self.norm_p[1])
            )

        # update the reservoir
        x_tilde = self.get_next_step(u_augmented)

        # apply the leaky integrator
        x = x_tilde
        return x

    def step(self, x_prev, u, p):
        if self.step_mode == "step1":
            return self.step1(x_prev, u, p)
        elif self.step_mode == "step2":
            return self.step2(x_prev, u, p)

    def open_loop(self, x0, U, P=None):
        """Advances ESN in open-loop.
        Args:
            x0: initial reservoir state
            U: input time series in matrix form (N_t x N_dim)
            P: parameter time series (N_t x N_param_dim)
        Returns:
            X: time series of the reservoir states (N_t x N_reservoir)
        """
        N_t = U.shape[0]  # number of time steps

        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t + 1, self.N_reservoir))
        # N_t+1 because at t = 0, we don't have input

        # initialise with the given initial reservoir states
        X[0, :] = x0
        # X = [x0]
        # step in time

        for n in np.arange(1, N_t + 1):
            # update the reservoir
            if self.N_param_dim > 0:
                X[n] = self.step(X[n - 1, :], U[n - 1, :], P[n - 1, :])
                # X.append(self.step(X[n - 1], U[n - 1], P[n - 1]))
            else:
                X[n] = self.step(X[n - 1, :], U[n - 1, :])
                # X.append(self.step(X[n - 1], U[n - 1]))
        # X = np.array(X)
        return X

    def closed_loop(self, x0, y0, N_t, P=None):
        # @todo: make it an option to hold X or just x in memory
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
        X = np.empty((N_t + 1, self.N_reservoir))
        # create an empty matrix to hold the output states in time
        Y = np.empty((N_t + 1, self.N_dim))

        # initialize with the given initial reservoir states
        X[0, :] = x0

        # augment the reservoir states with the bias
        if self.r2_mode:
            x0_2 = x0.copy()
            x0_2[1::2] = x0_2[1::2] ** 2
            x0_augmented = np.hstack((x0_2, self.b_out))
        else:
            x0_augmented = np.hstack((x0, self.b_out))

        # initialise with the calculated output states
        if self.step_mode == "step1":
            Y[0, :] = np.dot(x0_augmented, self.W_out)
        elif self.step_mode == "step2":
            Y[0, :] = y0 + np.dot(x0_augmented, self.W_out)

        # step in time
        for n in range(1, N_t + 1):
            # update the reservoir with the feedback from the output
            if self.N_param_dim > 0:
                X[n, :] = self.step(X[n - 1, :], Y[n - 1, :], P[n - 1, :])
            else:
                X[n, :] = self.step(X[n - 1, :], Y[n - 1, :])

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
                Y[n, :] = Y[n - 1, : self.N_dim] + np.dot(x_augmented, self.W_out)
        return X, Y

    def run_washout(self, U_washout, P_washout=None):
        # Wash-out phase to get rid of the effects of reservoir states initialised as zero
        # initialise the reservoir states before washout
        x0_washout = np.zeros(self.N_reservoir)

        # let the ESN run in open-loop for the wash-out
        # get the initial reservoir to start the actual open/closed-loop,
        # which is the last reservoir state
        x0 = self.open_loop(x0=x0_washout, U=U_washout, P=P_washout)[-1, :]
        y0 = U_washout[-1, :]
        return x0, y0

    def open_loop_with_washout(self, U_washout, U, P_washout=None, P=None):
        x0, _ = self.run_washout(U_washout, P_washout)
        X = self.open_loop(x0=x0, U=U, P=P)
        return X

    def closed_loop_with_washout(self, U_washout, N_t, P_washout=None, P=None):
        x0, y0 = self.run_washout(U_washout, P_washout)
        X, Y = self.closed_loop(x0=x0, y0=y0, N_t=N_t, P=P)
        return X, Y

    def solve_ridge(self, X, Y, tikh, sample_weights):
        """Solves the ridge regression problem
        Args:
            X: input data
            Y: output data
            tikh: tikhonov coefficient that regularises L2 norm
        """
        # @todo:
        # can set the method for ridge regression, compare the methods
        # scikit recommends minibatch sgd method for large scale data
        # Alberto implements the closed-form solution because he doesn't want to recalculate
        # the matmuls for each tikhonov parameter?
        if sample_weights is not None and len(sample_weights) == Y.shape[1]:
            W_out = np.zeros((X.shape[1], Y.shape[1]))
            for y_idx in range(Y.shape[1]):
                reg = Ridge(alpha=tikh, fit_intercept=False)
                reg.fit(X, Y[:, y_idx], sample_weight=sample_weights[y_idx])
                W_out[
                    :, y_idx
                ] = (
                    reg.coef_.T
                )  # we take the transpose because of how the closed loop is structured
        else:
            reg = Ridge(alpha=tikh, fit_intercept=False)
            reg.fit(X, Y, sample_weight=sample_weights)
            W_out = reg.coef_.T
        # W_out = np.dot(np.dot(Y.T, X), np.linalg.inv((np.dot(X.T, X)+tikh*np.identity(self.N_reservoir))))
        # W_out = W_out.T
        return W_out

    def reservoir_for_train(self, U_washout, U_train, P_washout=None, P_train=None):
        X_train = self.open_loop_with_washout(U_washout, U_train, P_washout, P_train)

        # X_train is one step longer than U_train and Y_train, we discard the initial state
        X_train = X_train[1:, :]

        # augment with the bias
        N_t = X_train.shape[0]  # number of time steps

        if self.r2_mode:
            X_train2 = X_train.copy()
            X_train2[:, 1::2] = X_train2[:, 1::2] ** 2
            X_train_augmented = np.hstack((X_train2, self.b_out * np.ones((N_t, 1))))
        else:
            X_train_augmented = np.hstack((X_train, self.b_out * np.ones((N_t, 1))))

        return X_train_augmented

    def train(
        self,
        U_washout,
        U_train,
        Y_train,
        P_washout=None,
        P_train=None,
        tikhonov=1e-12,
        train_idx_list=None,
        sample_weights=None,
    ):
        """Trains ESN and sets the output weights.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            Y_train: training output time series
            (list of time series if more than one trajectories)
            P_washout: parameters in washout
            P_train: parameters in training
            tikhonov: regularization coefficient
            train_idx_list: if list of time series, then which ones to use in training
                if not specified, all are used
        """
        # get the training input
        # this is the reservoir states augmented with the bias after a washout phase
        if isinstance(U_train, list):
            X_train_augmented = np.empty((0, self.W_out_shape[0]))
            if train_idx_list is None:
                train_idx_list = range(len(U_train))
            if self.step_mode == "step2":
                for train_idx in train_idx_list:
                    X_train_augmented_ = self.reservoir_for_train(
                        U_washout[train_idx],
                        U_train[train_idx],
                        P_washout[train_idx],
                        P_train[train_idx],
                    )
                    X_train_augmented = np.vstack(
                        (X_train_augmented, X_train_augmented_[1:, :])
                    )
                Y_train = [
                    Y_train[train_idx][1:, :] - Y_train[train_idx][0:-1, :]
                    for train_idx in train_idx_list
                ]
                Y_train = np.vstack(Y_train)
            elif self.step_mode == "step1":
                for train_idx in train_idx_list:
                    X_train_augmented_ = self.reservoir_for_train(
                        U_washout[train_idx],
                        U_train[train_idx],
                        P_washout[train_idx],
                        P_train[train_idx],
                    )
                    X_train_augmented = np.vstack(
                        (X_train_augmented, X_train_augmented_)
                    )
                Y_train = [Y_train[train_idx] for train_idx in train_idx_list]
                Y_train = np.vstack(Y_train)
        else:
            X_train_augmented = self.reservoir_for_train(
                U_washout, U_train, P_washout, P_train
            )

        # solve for W_out using ridge regression
        self.tikhonov = tikhonov  # set the tikhonov during training
        self.output_weights = self.solve_ridge(
            X_train_augmented, Y_train, tikhonov, sample_weights
        )
        return