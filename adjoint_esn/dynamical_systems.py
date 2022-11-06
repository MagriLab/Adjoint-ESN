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
