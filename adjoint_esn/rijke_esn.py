from functools import partial

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge

import adjoint_esn.generate_input_weights as generate_input_weights
from adjoint_esn.esn import ESN
from adjoint_esn.rijke_galerkin.solver import Rijke
from adjoint_esn.utils.discretizations import finite_differences


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
        dt,
        tau=0.0,
        reservoir_connectivity=0,
        parameter_dimension=0,
        input_normalization=None,
        parameter_normalization=None,
        input_scaling=1.0,
        u_f_scaling=1.0,
        u_f_order=1,
        spectral_radius=1.0,
        leak_factor=1.0,
        input_bias=np.array([]),
        output_bias=np.array([]),
        input_seeds=[None, None, None],
        reservoir_seeds=[None, None],
        tikhonov=None,
        verbose=True,
        r2_mode=False,
        input_only_mode=False,
        input_weights_mode="sparse_grouped_rijke_dense",
        reservoir_weights_mode="erdos_renyi2",
    ):
        self.verbose = verbose
        if r2_mode == True:
            print("r2 mode not implemented, setting to False")
        self.r2_mode = False

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
        self.u_f_order = u_f_order

        self.leak_factor = leak_factor

        ## Biases
        self.input_bias = input_bias
        self.output_bias = output_bias

        ## Input normalization
        if not input_normalization:
            input_normalization = [None] * 2
            input_normalization[0] = np.zeros(self.N_dim + self.u_f_order)
            input_normalization[1] = np.ones(self.N_dim + self.u_f_order)

        self.input_normalization = input_normalization

        if not parameter_normalization:
            parameter_normalization = [None] * 2
            parameter_normalization[0] = np.zeros(self.N_param_dim)
            parameter_normalization[1] = np.ones(self.N_param_dim)

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

        # initialise reservoir weights
        if not self.input_only_mode:
            self.reservoir_connectivity = reservoir_connectivity
            self.W_seeds = reservoir_seeds
            self.W_shape = (self.N_reservoir, self.N_reservoir)
            self.reservoir_weights_mode = reservoir_weights_mode
            valid_W = False
            while not valid_W:
                try:
                    self.reservoir_weights = self.generate_reservoir_weights()
                    valid_W = True
                except:
                    # perturb the seeds
                    for i in range(2):
                        if self.W_seeds[i]:
                            self.W_seeds[i] = self.W_seeds[i] + 1
                    valid_W = False
                    print("Not valid reservoir encountered, changing seed.")
            self.spectral_radius = spectral_radius
            # reservoir weights are automatically scaled if spectral radius is updated

        # tikhonov coefficient
        if tikhonov:
            self.tikhonov = tikhonov

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
        self.N_tau = int(np.round(self.tau / self.dt))
        return

    @property
    def x_f(self):
        return self._x_f

    @x_f.setter
    def x_f(self, new_x_f):
        self._x_f = new_x_f
        return

    @property
    def u_f_order(self):
        return self._u_f_order

    @u_f_order.setter
    def u_f_order(self, new_u_f_order):
        if new_u_f_order <= 0:
            raise ValueError("Order of the delayed velocity must be greater than 0.")
        self._u_f_order = new_u_f_order
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

    @property
    def dfdx_tau_const(self):
        if not hasattr(self, "_dfdx_tau_const"):
            j = np.arange(1, self.N_g + 1)
            modes = np.cos(j * np.pi * self.x_f)
            modes = np.hstack((modes, np.zeros(self.N_g)))
            W_out_modes = np.dot(self.W_out[: self.N_reservoir, :], modes)[:, None].T
            self._dfdx_tau_const = W_out_modes
        return self._dfdx_tau_const

    def jac_tau(self, dtanh, u_f_tau):
        """Jacobian of the reservoir states with respect to the tau delayed
        reservoir states
        Args:
        dtanh: derivative of tanh at x(i+1), x(i)

        Returns:
        dfdx_tau: jacobian of the reservoir states
        """
        # derivative of delayed velocity
        du_f_tau = np.zeros(self.u_f_order)
        for order in range(self.u_f_order):
            du_f_tau[order] = (order + 1) * u_f_tau**order

        dfdu_f_tau = self.dfdu_f_tau_const.dot(du_f_tau)[:, None]

        # derivative with respect to delayed reservoir states
        dfdx_tau = np.matmul(dfdu_f_tau, self.dfdx_tau_const)

        dfdx_tau = np.multiply(dfdx_tau, dtanh)
        return dfdx_tau

    @property
    def dfdu_f_tau_const(self):
        if not hasattr(self, "_dfdu_f_tau_const"):
            self._dfdu_f_tau_const = self.alpha * self.W_in[
                :, -self.N_param_dim - self.u_f_order : -self.N_param_dim
            ].multiply(
                1.0
                / self.norm_in[1][
                    -self.N_param_dim - self.u_f_order : -self.N_param_dim
                ]
            )
        return self._dfdu_f_tau_const

    def dfdu_f_tau(self, dtanh, u_f_tau):
        """Gradient of the reservoir states with respect to delayed velocity
        Args:
        dtanh:

        Returns:
        dfdu_f_tau
        """
        du_f_tau = np.zeros(self.u_f_order)
        for order in range(self.u_f_order):
            du_f_tau[order] = (order + 1) * u_f_tau**order

        dfdu_f_tau_ = self.dfdu_f_tau_const.dot(du_f_tau)[:, None]

        dfdu_f_tau = np.multiply(dfdu_f_tau_, dtanh)

        return dfdu_f_tau

    def reset_grad_attrs(self):
        # reset the attributes
        attr_list = [
            "_dfdu_const",
            "_dudx_const",
            "_dfdu_dudx_const",
            "_dfdx_x_const",
            "_dfdx_u",
            "_dfdp_const",
            "_dydf",
            "_dfdx_tau_const",
            "_dfdu_f_tau_const",
            "_dfdx_const",
        ]
        for attr in attr_list:
            if hasattr(self, attr):
                delattr(self, attr)

    def direct_sensitivity(
        self, X, Y, N, X_past, method="central", dJdy_fun=None, fast_jac=False
    ):
        """Sensitivity of the ESN with respect to the parameters
        Calculated using DIRECT method
        Objective is squared L2 of the 2*N_g output states, i.e. acoustic energy

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            P: parameter
            N: number of steps
            N_g: number of galerkin modes,
                assuming outputs are ordered such that the first 2*N_g correspond to the
                Galerkin amplitudes

        Returns:
            dJdp: direct sensitivity to parameters
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # if the objective is not defined the default is the acoustic energy
        if dJdy_fun is None:
            dJdy_fun = partial(self.dacoustic_energy, N_g=self.N_g)

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize direct variables, dx(i+1)/dp
        # dJ_dp doesn't depend on the initial reservoir state, i.e. q[0] = 0
        # we add time delay, tau as a parameter
        q = np.zeros((N + 1, self.N_reservoir, self.N_param_dim + 1))

        # initialize sensitivity,
        dJdp = np.zeros(self.N_param_dim + 1)

        # stack with the past
        XX = np.vstack((X_past[-self.N_tau - 1 : -1, :], X))

        for i in np.arange(1, N + 1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # partial derivative with respect to parameters
            dfdbeta = self.dfdp(dtanh)

            # get tau sensitivity via finite difference
            x_aug_left = self.before_readout(XX[i, :])
            x_aug = self.before_readout(XX[i - 1, :])

            eta_tau_left = np.dot(x_aug_left, self.W_out)[0 : self.N_g]
            eta_tau = np.dot(x_aug, self.W_out)[0 : self.N_g]

            u_f_tau_left = Rijke.toVelocity(
                N_g=self.N_g, eta=eta_tau_left, x=np.array([self.x_f])
            )
            u_f_tau = Rijke.toVelocity(
                N_g=self.N_g, eta=eta_tau, x=np.array([self.x_f])
            )
            if method == "central" and i > 1:
                x_aug_right = self.before_readout(XX[i - 2, :])
                eta_tau_right = np.dot(x_aug_right, self.W_out)[0 : self.N_g]
                u_f_tau_right = Rijke.toVelocity(
                    N_g=self.N_g, eta=eta_tau_right, x=np.array([self.x_f])
                )
                # gradient of the delayed velocity with respect to tau
                # central finite difference
                du_f_tau_dtau = (u_f_tau_right - u_f_tau_left) / (2 * self.dt)
            else:
                # gradient of the delayed velocity with respect to tau
                # backward finite difference
                du_f_tau_dtau = (u_f_tau - u_f_tau_left) / self.dt

            dfdu_f_tau = self.dfdu_f_tau(dtanh, u_f_tau)

            # tau sensitivity given by chain rule
            dfdtau = dfdu_f_tau * du_f_tau_dtau

            # stack the parameter sensitivities
            dfdp = np.hstack((dfdbeta, dfdtau))

            # integrate direct variables forwards in time
            jac = jac_fun(dtanh, X[i - 1, :])

            if i <= self.N_tau:
                q[i] = dfdp + np.dot(jac, q[i - 1])
            # depends on the past
            elif i > self.N_tau:
                jac_tau = self.jac_tau(dtanh, u_f_tau)
                q[i] = (
                    dfdp
                    + np.dot(jac, q[i - 1])
                    + np.dot(jac_tau, q[i - self.N_tau - 1])
                )

            # get objective with respect to output states
            dJdy = dJdy_fun(Y[i])

            # gradient of objective with respect to reservoir states
            dydf = self.dydf(X[i, :])
            dJdf = (1 / N) * np.dot(dJdy, dydf)

            # sensitivity to parameters
            dJdp += np.dot(dJdf, q[i])

        return dJdp

    def adjoint_sensitivity(
        self, X, Y, N, X_past, method="central", dJdy_fun=None, fast_jac=False
    ):
        """Sensitivity of the ESN with respect to the parameters
        Calculated using ADJOINT method
        Objective is squared L2 of the 2*N_g output states, i.e. acoustic energy

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            P: parameter
            N: number of steps
            N_g: number of galerkin modes,
                assuming outputs are ordered such that the first 2*N_g correspond to the
                Galerkin amplitudes

        Returns:
            dJdp: adjoint sensitivity to parameters
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # if the objective is not defined the default is the acoustic energy
        if dJdy_fun is None:
            dJdy_fun = partial(self.dacoustic_energy, N_g=self.N_g)

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize adjoint variables
        v = np.zeros((N + 1, self.N_reservoir))

        # initialize sensitivity
        dJdp = np.zeros(self.N_param_dim + 1)

        # stack with the past
        XX = np.vstack((X_past[-self.N_tau - 1 : -1, :], X))

        # integrate backwards
        # objective function at the terminal state
        dJdy = dJdy_fun(Y[N])

        # terminal condition,
        # i.e. gradient of the objective at the terminal state
        dydf = self.dydf(X[N, :])
        v[N] = (1 / N) * np.dot(dJdy, dydf).T

        for i in np.arange(N, 0, -1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # partial derivative with respect to parameters
            dfdbeta = self.dfdp(dtanh)

            # get tau sensitivity via finite difference
            x_aug_left = self.before_readout(XX[i, :])
            x_aug = self.before_readout(XX[i - 1, :])

            eta_tau_left = np.dot(x_aug_left, self.W_out)[0 : self.N_g]
            eta_tau = np.dot(x_aug, self.W_out)[0 : self.N_g]

            u_f_tau_left = Rijke.toVelocity(
                N_g=self.N_g, eta=eta_tau_left, x=np.array([self.x_f])
            )
            u_f_tau = Rijke.toVelocity(
                N_g=self.N_g, eta=eta_tau, x=np.array([self.x_f])
            )

            if method == "central" and i > 1:
                x_aug_right = self.before_readout(XX[i - 2, :])
                eta_tau_right = np.dot(x_aug_right, self.W_out)[0 : self.N_g]
                u_f_tau_right = Rijke.toVelocity(
                    N_g=self.N_g, eta=eta_tau_right, x=np.array([self.x_f])
                )
                # gradient of the delayed velocity with respect to tau
                # central finite difference
                du_f_tau_dtau = (u_f_tau_right - u_f_tau_left) / (2 * self.dt)
            else:
                # gradient of the delayed velocity with respect to tau
                # backward finite difference
                du_f_tau_dtau = (u_f_tau - u_f_tau_left) / self.dt

            dfdu_f_tau = self.dfdu_f_tau(dtanh, u_f_tau)

            # tau sensitivity given by chain rule
            dfdtau = dfdu_f_tau * du_f_tau_dtau

            # stack the parameter sensitivities
            dfdp = np.hstack((dfdbeta, dfdtau))

            # sensitivity to parameters
            dJdp += np.dot(v[i], dfdp)

            # get the derivative of the objective with respect to the outputs
            dJdy = dJdy_fun(Y[i - 1])

            # gradient of objective with respect to reservoir states
            dydf = self.dydf(X[i - 1, :])
            dJdf = (1 / N) * np.dot(dJdy, dydf).T

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            if i <= N - self.N_tau:
                # need tau "advanced" velocity (delayed becomes advanced in adjoint)
                eta_tau_future = Y[i - 1, 0 : self.N_g]
                u_f_tau_future = Rijke.toVelocity(
                    N_g=self.N_g, eta=eta_tau_future, x=np.array([self.x_f])
                )

                dtanh_future = self.dtanh(
                    X[i + self.N_tau, :], X[i + self.N_tau - 1, :]
                )[:, None]
                jac_tau = self.jac_tau(dtanh_future, u_f_tau_future)
                v[i - 1] = (
                    np.dot(jac.T, v[i]) + np.dot(jac_tau.T, v[i + self.N_tau]) + dJdf
                )
            else:
                v[i - 1] = np.dot(jac.T, v[i]) + dJdf

        return dJdp

    def finite_difference_sensitivity(
        self, X, Y, X_tau, P, N, h=1e-5, method="central", J_fun=None
    ):
        """Sensitivity of the ESN with respect to the parameters
        Calculated using FINITE DIFFERENCES
        Objective is squared L2 of the 2*N_g output states, i.e. acoustic energy

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            X_tau: tau steps past (initial conditions)
            P: parameter
            N: number of steps
            N_g: number of galerkin modes,
                assuming outputs are ordered such that the first 2*N_g correspond to the
                Galerkin amplitudes
            h: perturbation
            method: finite difference method, "forward","backward" or "central"

        Returns:
            dJdp: numerical sensitivity to parameters
        """
        # initialize sensitivity
        dJdp = np.zeros((self.N_param_dim + 1))

        if J_fun is None:
            J_fun = partial(self.acoustic_energy, N_g=self.N_g)

        # compute the energy of the base
        J = J_fun(Y[1:])

        # define which finite difference method to use
        finite_difference = partial(finite_differences, method=method)

        # central finite difference
        # perturbed by h
        for i in range(self.N_param_dim):
            P_left = P.copy()
            P_left[:, i] -= h
            P_right = P.copy()
            P_right[:, i] += h
            _, Y_left = self.closed_loop(X_tau[-self.N_tau - 1 :, :], N, P_left)
            _, Y_right = self.closed_loop(X_tau[-self.N_tau - 1 :, :], N, P_right)
            J_left = J_fun(Y_left[1:])
            J_right = J_fun(Y_right[1:])
            dJdp[i] = finite_difference(J, J_right, J_left, h)

        # tau sensitivity
        orig_tau = self.tau
        h_tau = self.dt
        self.tau = orig_tau - h_tau
        _, Y_tau_left = self.closed_loop(X_tau[-self.N_tau - 1 :, :], N, P)
        J_tau_left = J_fun(Y_tau_left[1:])

        self.tau = orig_tau + h_tau
        _, Y_tau_right = self.closed_loop(X_tau[-self.N_tau - 1 :, :], N, P)
        J_tau_right = J_fun(Y_tau_right[1:])
        dJdp[-1] = finite_difference(J, J_tau_right, J_tau_left, h_tau)

        # set tau back to the original value
        self.tau = orig_tau

        return dJdp
