import numpy as np
from scipy.integrate import odeint


class Rijke:
    def __init__(self, N_g, N_c, c_1, c_2, beta, x_f, tau, heat_law, damping):
        """Create Rijke system instance with the given parameters
        Uses Galerkin decomposition method

        Args:
            N_g: number of Galerkin modes
            N_c: number of Chebyshev points
            c_1, c_2: damping coefficients
            beta: heat release strength
            x_f: heat source location
            tau: time delay in heat release
            heat_law: heat release law, 'kings' or 'sigmoid'
            damping: damping term, 'modal' or 'constant'
        """
        self.N_g = N_g
        self.N_c = N_c
        self.c_1 = c_1
        self.c_2 = c_2
        self.beta = beta
        self.x_f = x_f
        self.tau = tau
        self.heat_law = heat_law
        self.damping = damping

    @property
    def N_dim(self):
        # Total number of Galerkin states
        return 2 * self.N_g + self.N_c

    @property
    def j(self):
        # Array for mode indices
        if not hasattr(self, "_j"):
            self._j = np.arange(1, self.N_g + 1)
        return self._j

    @property
    def jpi(self):
        if not hasattr(self, "_jpi"):
            self._jpi = self.j * np.pi
        return self._jpi

    @property
    def zeta(self):
        # Damping for each mode
        if not hasattr(self, "_zeta"):
            if self.damping == "modal":
                self._zeta = self.c_1 * self.j**2 + self.c_2 * self.j**0.5
            elif self.damping == "constant":
                self._zeta = self.c_1 * np.ones(self.N_g)
        return self._zeta

    @property
    def cosjpixf(self):
        if not hasattr(self, "_cosjpixf"):
            self._cosjpixf = np.cos(self.jpi * self.x_f)
        return self._cosjpixf

    @property
    def sinjpixf(self):
        if not hasattr(self, "_sinjpixf"):
            self._sinjpixf = np.sin(self.jpi * self.x_f)
        return self._sinjpixf

    @property
    def chebdiff(self):
        # Chebyshev differential matrix
        # adapted from Trefethen 2000
        if not hasattr(self, "_chebdiff"):
            x = -np.cos(np.pi * np.arange(self.N_c + 1) / self.N_c)
            x = np.vstack(x)

            c = np.hstack([2, np.ones(self.N_c - 1), 2]) * (-1) ** np.arange(
                self.N_c + 1
            )
            c = np.vstack(c)

            X = np.tile(x, (1, self.N_c + 1))
            dX = X - X.T

            D = c * np.hstack(1 / c) / (dX + np.eye(self.N_c + 1))
            D = D - np.diag(np.sum(D, axis=1))
            self._chebdiff = D
        return self._chebdiff

    @staticmethod
    def toPressure(N_g, mu, x):
        """Find pressure at the given locations from Galerkin variables

        p(x,t) = -\sum_{j = 1}^N_g \mu_j(t)\sin(j\pi x)

        Args:
            N_g: number of Galerkin modes
            mu: Galerkin variables associated with pressure
            x: Spatial grid
        """
        j = np.arange(1, N_g + 1)

        p = -np.matmul(mu, np.sin(np.outer(j * np.pi, x)))
        return p

    @staticmethod
    def toPressureDot(N_g, mu_dot, x):
        """Find time derivative of pressure at the given locations from Galerkin variables

        dp(x,t)/dt = -\sum_{j = 1}^N_g \dot{\mu}_j(t)\sin(j\pi x)

        Args:
            N_g: number of Galerkin modes
            mu_dot: time derivative of Galerkin variables associated with pressure
            x: Spatial grid
        """
        j = np.arange(1, N_g + 1)

        p_dot = -np.matmul(mu_dot, np.sin(np.outer(j * np.pi, x)))
        return p_dot

    @staticmethod
    def toPressurePrime(N_g, mu, x):
        """Find spatial derivative of pressure at the given locations from Galerkin variables

        dp(x,t)/dx = -\sum_{j = 1}^N_g \mu_j(t)j\pi\cos(j\pi x)

        Args:
            N_g: number of Galerkin modes
            mu: Galerkin variables associated with pressure
            x: Spatial grid
        """
        j = np.arange(1, N_g + 1)

        p_prime = -np.matmul(mu * j * np.pi, np.cos(np.outer(j * np.pi, x)))
        return p_prime

    @staticmethod
    def toVelocity(N_g, eta, x):
        """Find velocity at the given locations from Galerkin variables

        u(x,t) = \sum_{j = 1}^N_g \eta_j(t)\cos(j\pi x)

        Args:
            N_g: number of Galerkin modes
            eta: Galerkin variables associated with velocity
            x: Spatial grid
        """
        j = np.arange(1, N_g + 1)

        u = np.matmul(eta, np.cos(np.outer(j * np.pi, x)))
        return u

    @staticmethod
    def toVelocityDot(N_g, eta_dot, x):
        """Find time derivative of velocity at the given locations from Galerkin variables

        du(x,t)/dt = \sum_{j = 1}^N_g \dot{\eta}_j(t)\cos(j\pi x)

        Args:
            N_g: number of Galerkin modes
            eta_dot: time derivative of Galerkin variables associated with velocity
            x: Spatial grid
        """
        j = np.arange(1, N_g + 1)

        u_dot = np.matmul(eta_dot, np.cos(np.outer(j * np.pi, x)))
        return u_dot

    @staticmethod
    def toVelocityPrime(N_g, eta, x):
        """Find spatial derivative of velocity at the given locations from Galerkin variables

        du(x,t)/dx = -\sum_{j = 1}^N_g \eta_j(t)j\pi\sin(j\pi x)

        Args:
            N_g: number of Galerkin modes
            eta: Galerkin variables associated with velocity
            x: Spatial grid
        """
        j = np.arange(1, N_g + 1)

        u_prime = -np.matmul(eta * j * np.pi, np.sin(np.outer(j * np.pi, x)))
        return u_prime

    def ode(self, t, y):
        """Rijke system ode

        Galerkin expansion
        u(x,t) = sum_{j=1}^{N_g} \eta_j(t)\cos(j\pi x)
        p(x,t) = -sum_{j=1}^{N_g} \mu_j(t)\sin(j\pi x)

        d\eta_j/dt = j\pi\mu_j
        d\mu_j/dt = -j\pi\eta_j-\zeta_j*\mu_j-2*dq/dt*sin(j\pi x_f)
        dq/dt = \beta/(1+\exp(-u_f(t-\tau)))

        Dummy variable v to track u_f(t-\tau)
        \partial v/\partial t = -1/\tau*\partial v/\partial X, 0<= X <= 1
        v(X = 0,t) = u_f(t), v(X = 1,t) = u_f(t-\tau)

        Integrate the instantenous acoustic energy \tilde{J} simultaneously

        Args:
            t: time
            y: state vector, [\eta(t),\mu(t),v(t),\int_0^t \tilde{J}]
        Returns:
            dy/dt = [d\eta/dt,d\mu/dt,d\v/dt,\tilde{J}]
        """
        # write the states
        eta = y[0 : self.N_g]
        mu = y[self.N_g : 2 * self.N_g]
        v = y[2 * self.N_g : 2 * self.N_g + self.N_c]

        # deta_j/dt
        eta_dot = self.jpi * mu

        # velocity at the heat source
        u_f = np.dot(eta, self.cosjpixf)

        # dummy variable to track delayed velocity
        v = np.hstack([u_f, v])
        v_dot = -2 / self.tau * np.dot(self.chebdiff, v)[1:]  # multiply by 2
        # chebyshev grid [-1,1], X = [0,1], so the speed becomes 2/\tau (from -1 to 1)

        # find heat release
        u_f_tau = v[-1]
        if self.heat_law == "kings":
            q_dot = self.beta * (np.sqrt(np.abs(1 + u_f_tau)) - 1)
        elif self.heat_law == "sigmoid":
            q_dot = self.beta / (1 + np.exp(-u_f_tau))

        # dmu_j/dt
        mu_dot = -self.jpi * eta - self.zeta * mu - 2 * q_dot * self.sinjpixf

        # calculate J = \int_0^T \tilde{J} simultaneously
        J_tilde = 1 / 4 * np.sum(eta**2 + mu**2)

        dydt = np.hstack([eta_dot, mu_dot, v_dot, q_dot, J_tilde])
        return dydt

    # Following derivatives are used in the calculation of direct and adjoint eqns
    # f = [d\eta/dt,d\mu/dt,dv/dt]
    @property
    def df1_deta(self):
        if not hasattr(self, "_df1_deta"):
            self._df1_deta = np.zeros([self.N_g, self.N_g])
        return self._df1_deta

    @property
    def df1_dmu(self):
        if not hasattr(self, "_df1_dmu"):
            self._df1_dmu = np.diag(self.jpi)
        return self._df1_dmu

    @property
    def df1_dv(self):
        if not hasattr(self, "_df1_dv"):
            self._df1_dv = np.zeros([self.N_g, self.N_c])
        return self._df1_dv

    @property
    def df2_deta(self):
        if not hasattr(self, "_df2_deta"):
            self._df2_deta = -np.diag(self.jpi)
        return self._df2_deta

    @property
    def df2_dmu(self):
        if not hasattr(self, "_df2_dmu"):
            self._df2_dmu = -np.diag(self.zeta)
        return self._df2_dmu

    @property
    def df3_deta(self):
        if not hasattr(self, "_df3_deta"):
            self._df3_deta = (
                -2 / self.tau * np.outer(self.chebdiff[:, 0], self.cosjpixf)
            )
            self._df3_deta = self._df3_deta[1:, :]
        return self._df3_deta

    @property
    def df3_dmu(self):
        if not hasattr(self, "_df3_dmu"):
            self._df3_dmu = np.zeros([self.N_c, self.N_g])
        return self._df3_dmu

    @property
    def df3_dv(self):
        if not hasattr(self, "_df3_dv"):
            self._df3_dv = -2 / self.tau * self.chebdiff[1:, 1:]
        return self._df3_dv

    def dFdy_u_f_tau_bar(self, u_f_tau_bar):
        """Linearized system around u_f(t-\tau) = \bar{u_f}(t-\tau)
        F(dy/dt,y) = 0, dy/dt = f(y)
        df/dy = [[df_1/d\eta, df_1/d\mu, df_1/dv],
                  [df_2/d\eta, df_2/d\mu, df_2/dv],
                  [df_3/d\eta, df_3/d\mu, df_3/dv]]
        dF/dy = -df/dy
        """
        df1_dy = np.hstack([self.df1_deta, self.df1_dmu, self.df1_dv])

        # The only nonlinear term that needs to be linearized
        df2_dv = np.zeros([self.N_g, self.N_c])
        if self.heat_law == "kings":
            # not defined for \bar{u_f}(t-\tau) = -1
            dq_dot_du_f_tau = (
                self.beta
                * (1 + u_f_tau_bar)
                / (2 * (np.abs(1 + u_f_tau_bar)) ** (3 / 2))
            )
        elif self.heat_law == "sigmoid":
            dq_dot_du_f_tau = (
                self.beta * np.exp(-u_f_tau_bar) / ((1 + np.exp(-u_f_tau_bar)) ** 2)
            )
        df2_dv[:, -1] = -2 * dq_dot_du_f_tau * self.sinjpixf

        df2_dy = np.hstack([self.df2_deta, self.df2_dmu, df2_dv])

        df3_dy = np.hstack([self.df3_deta, self.df3_dmu, self.df3_dv])

        dfdy = np.vstack([df1_dy, df2_dy, df3_dy])
        dFdy = -dfdy
        return dFdy

    def dFdp_p_bar(self, eta, v, u_f_tau):
        """Derivative of the system with respect to the parameters p_bar = [beta,tau]"""
        df1_dbeta = np.zeros(self.N_g)
        df2_dbeta = -2 * self.sinjpixf / (1 + np.exp(-u_f_tau))
        df3_dbeta = np.zeros(self.N_c)

        df1_dtau = np.zeros(self.N_g)
        df2_dtau = np.zeros(self.N_g)

        u_f = np.dot(eta, self.cosjpixf)
        v = np.hstack([u_f, v])
        df3_dtau = (
            2 / (self.tau**2) * np.dot(self.chebdiff, v)[1:]
        )  # d/dtau(1/tau) = -1/tau^2

        dfdbeta = np.hstack([df1_dbeta, df2_dbeta, df3_dbeta])
        dfdtau = np.hstack([df1_dtau, df2_dtau, df3_dtau])

        dFdbeta = -dfdbeta
        dFdtau = -dfdtau
        return dFdbeta, dFdtau

    def equi_interp(self, xq, x, inv_dx, v):
        """Interpolate between uniformly spaced points with known spacing
        (speeds up interpolation of the base solution)

        Args:
            xq: queried points
            x: known points
            inv_dx: inverse of spacing 1/dx
            v: known values

        Returns:
            vq: values at the queried points
        """
        j = int(xq * inv_dx)
        if j >= len(x) - 1:
            vq = v[-1, :]
        else:
            xx = (xq - x[j]) * inv_dx
            vq = v[j, :] * (1 - xx) + v[j + 1, :] * xx
        return vq

    def direct_ode(self, t, dir, t_bar, inv_dt, y_bar):
        """Solve the direct problem to find the gradient dJ/dp

        dq/dt = -dF/dy*q-dF/dp
        d\tilde{J}/dp = d\tilde{J}/dx*q

        Args:
            t: time
            dir: [q(t), dJ/d\beta(t), dJ/d\tau(t)]
                 q = dy/dp, the direct variables
                 dJ/dp are integrated simultaneously
            t_bar, y_bar: base solution the system is linearized around
            inv_dt: inverse of temporal spacing, 1/dt

        Returns:
            ddir_dt = [dq/dt, d(dJ/d\beta)/dt, d(dJ/d\tau)/dt]
        """
        # number of direct variables increase with the parameters
        q_beta = dir[0 : self.N_dim]
        q_tau = dir[self.N_dim : 2 * self.N_dim]

        # interpolate to get the base solution at time t
        # need interpolation since ode solver is variable step size
        y_bar_t = self.equi_interp(t, t_bar, inv_dt, y_bar)
        eta_bar = y_bar_t[0 : self.N_g]
        mu_bar = y_bar_t[self.N_g : 2 * self.N_g]
        v_bar = y_bar_t[2 * self.N_g : self.N_dim]

        u_f_tau_bar = v_bar[-1]

        # linearize system around \bar{u_f}(t-\tau)
        dFdy = self.dFdy_u_f_tau_bar(u_f_tau_bar)
        dFdy_q_beta = np.dot(dFdy, q_beta)
        dFdy_q_tau = np.dot(dFdy, q_tau)

        # derivatives with respect to \beta and \tau
        dFdbeta, dFdtau = self.dFdp_p_bar(eta_bar, v_bar, u_f_tau_bar)

        # Direct equations
        dq_beta_dt = -dFdy_q_beta - dFdbeta
        dq_tau_dt = -dFdy_q_tau - dFdtau

        dJdy = 1 / 2 * np.hstack([eta_bar, mu_bar, np.zeros(self.N_c)])

        dJ_dbeta_dt = np.dot(dJdy, q_beta)
        dJ_dtau_dt = np.dot(dJdy, q_tau)

        dqdt = np.hstack([dq_beta_dt, dq_tau_dt])
        ddir_dt = np.hstack([dqdt, dJ_dbeta_dt, dJ_dtau_dt])
        return ddir_dt

    def adjoint_ode(self, t, adj, t_bar, inv_dt, y_bar):
        """Solve the adjoint problem to find the gradient dJ/dp
        dq^+/dt = q^+'*dF/dy-dJ/dy
        dJ/dp = 1/T int_0^T (-q^+ dF/dp)dt

        Args:
            t: time
            adj: [q^+(t), dL/d\beta(t), dL/d\tau(t)]
                 q^+, the adjoint variables
                 dL/dp are integrated simultaneously
                 L, Lagrangian
            t_bar, y_bar: base solution the system is linearized around
            inv_dt: inverse of temporal spacing, 1/dt

        Returns:
            dadj_dt = [dq^+/dt, d(dL/d\beta)/dt, d(dL/d\tau)/dt]
        """
        q_plus = adj[0 : self.N_dim]

        # interpolate to get the base solution at time t
        y_bar_t = self.equi_interp(t, t_bar, inv_dt, y_bar)
        eta_bar = y_bar_t[0 : self.N_g]
        mu_bar = y_bar_t[self.N_g : 2 * self.N_g]
        v_bar = y_bar_t[2 * self.N_g : self.N_dim]

        u_f_tau_bar = v_bar[-1]

        # linearize system around \bar{u_f}(t-\tau)
        dFdy = self.dFdy_u_f_tau_bar(u_f_tau_bar)
        q_plus_dFdy = np.dot(q_plus, dFdy)

        dJdy = 1 / 2 * np.hstack([eta_bar, mu_bar, np.zeros(self.N_c)])

        # Adjoint equations
        dq_plus_dt = q_plus_dFdy - dJdy

        # derivatives with respect to \beta and \tau
        dFdbeta, dFdtau = self.dFdp_p_bar(eta_bar, v_bar, u_f_tau_bar)

        dL_dbeta_dt = np.dot(
            q_plus, dFdbeta
        )  # change sign since we are integrating backwards
        dL_dtau_dt = np.dot(
            q_plus, dFdtau
        )  # change sign since we are integrating backwards

        dadj_dt = np.hstack([dq_plus_dt, dL_dbeta_dt, dL_dtau_dt])
        return dadj_dt
