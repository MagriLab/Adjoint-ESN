from enum import IntEnum

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

    @staticmethod
    def get_eVar():
        var_list = ["x", "y", "z"]
        eVar = IntEnum("eVar", var_list, start=0)
        return eVar

    @staticmethod
    def get_eParamVar():
        var_list = ["beta", "rho", "sigma"]
        eParamVar = IntEnum("eParamVar", var_list, start=0)
        return eParamVar

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


class RoesslerLorenz:
    def __init__(self, a, b, c, mu, r, d, epsilon, t_lyap=None):
        """Create a coupled Roessler-Lorenz system instance with the given parameters"""
        self.a = a
        self.b = b
        self.c = c
        self.mu = mu
        self.r = r
        self.d = d
        self.epsilon = epsilon
        self.t_lyap = t_lyap
        self.N_dim = 6

    @property
    def params(self):
        """Returns a dictionary containing only the system parameters"""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "mu": self.mu,
            "r": self.r,
            "d": self.d,
            "epsilon": self.epsilon,
        }

    def ode(self, u, t):
        """Coupled Roessler-Lorenz system ode
        t: time
        u: state vector, contains x, y, z
        """
        x1, x2, x3, y1, y2, y3 = u
        dx1dt = -x2 - x3
        dx2dt = x1 + self.a * x2 + self.epsilon * (y2 - x2)
        dx3dt = self.b + x3 * (x1 - self.c)
        dy1dt = self.mu * (y2 - y1)
        dy2dt = -y1 * y3 - y2 + self.r * y1 + self.epsilon * (x2 - y2)
        dy3dt = y1 * y2 - self.d * y3

        dudt = np.array([dx1dt, dx2dt, dx3dt, dy1dt, dy2dt, dy3dt])
        return dudt


class VanDerPol:
    def __init__(self, mu):
        self.mu = mu
        self.N_dim = 2

    @property
    def params(self):
        """Returns a dictionary containing only the system parameters"""
        return {
            "mu": self.mu,
        }

    def ode(self, u, t):
        x1, x2 = u
        dx1dt = x2
        dx2dt = self.mu * (1 - x1**2) * x2 - x1

        dudt = np.array([dx1dt, dx2dt])
        return dudt


class MultiStable:
    def __init__(self, alpha, beta, epsilon, k, t_lyap=None):
        """Create a coupled Roessler-Lorenz system instance with the given parameters"""
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.k = k
        self.N_dim = 4

    @property
    def params(self):
        """Returns a dictionary containing only the system parameters"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "k": self.k,
        }

    def ode(self, q, t):
        """Multistable system ode from Roy 2022
        t: time
        q: state vector, contains x, y, z, u
        """
        x, y, z, u = q
        dxdt = y
        dydt = z
        dzdt = -y + 3 * y**2 - x**2 - x * z + self.alpha + self.epsilon * u
        dudt = -self.k * u - self.epsilon * (z - self.beta)

        dudt = np.array([dxdt, dydt, dzdt, dudt])
        return dudt
