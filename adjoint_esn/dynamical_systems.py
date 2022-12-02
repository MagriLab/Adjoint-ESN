import numpy as np


class Lorenz63:
    def __init__(self, beta, rho, sigma, t_lyap=None):
        """Create a Lorenz63 system instance with the given parameters"""
        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        self.t_lyap = t_lyap
        self.N_dim = 3

    @property
    def params(self):
        """Returns a dictionary containing only the system parameters"""
        return {"beta": self.beta, "rho": self.rho, "sigma": self.sigma}

    def ode(self, u, t):
        """Lorenz63 system ode
        t: time
        u: state vector, contains x, y, z
        """
        x, y, z = u
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        dudt = np.array([dxdt, dydt, dzdt])
        return dudt

    def jac(self, u):
        """Jacobian of the Lorenz63 system around the given state vector
        u: state vector, contains x, y, z
        """
        x, y, z = u
        dFdu = np.array(
            [[-self.sigma, self.sigma, 0], [self.rho - z, -1, -x], [y, x, -self.beta]]
        )
        return dFdu


class Lorenz96:
    def __init__(self, p, t_lyap=None):
        self.p = p
        self.t_lyap = t_lyap

    @property
    def params(self):
        """Returns a dictionary containing only the system parameters"""
        return {"p": self.p}

    def ode(self, x, t):
        """Lorenz96 system ode
        t: time
        x: state vector
        """
        dxdt = np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + self.p
        return dxdt

    def jac(self, x):
        """Jacobian of the Lorenz96 system around the given state vector
        x: state vector
        """
        D = len(x)
        dFdx = np.zeros((D, D), dtype="float")
        for i in range(D):
            dFdx[i, (i - 1) % D] = x[(i + 1) % D] - x[(i - 2) % D]
            dFdx[i, (i + 1) % D] = x[(i - 1) % D]
            dFdx[i, (i - 2) % D] = -x[(i - 1) % D]
            dFdx[i, i] = -1.0
        return dFdx
