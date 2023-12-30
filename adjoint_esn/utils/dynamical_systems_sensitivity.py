import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from functools import partial

import numpy as np

from adjoint_esn.utils import solve_ode
from adjoint_esn.utils.discretizations import finite_differences


def equi_interp(xq, x, inv_dx, v):
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


def direct_ode(dir, t, t_bar, inv_dt, y_bar, sys, dobjective_fun):
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
    # reshape the direct variables
    q = np.reshape(dir[: -sys.N_param], ((sys.N_dim, sys.N_param)))
    # interpolate to get the base solution at time t
    # need interpolation since ode solver is variable step size
    y_bar_t = equi_interp(t, t_bar, inv_dt, y_bar)

    # linearize system
    dFdy = -sys.jac(y_bar_t)
    dFdy_q = np.matmul(dFdy, q)

    # derivatives with respect to parameters
    dFdp = -sys.dfdp(y_bar_t)

    # direct equations
    dqdt = -dFdy_q - dFdp

    dJdy = dobjective_fun(y_bar_t)

    dJdp_dt = np.matmul(dJdy, q)

    dqdt_ = np.reshape(dqdt, (sys.N_dim * sys.N_param,))
    ddir_dt = np.hstack([dqdt_, dJdp_dt])
    return ddir_dt


def true_direct_sensitivity(my_sys, t_bar, y_bar, dobjective_fun, integrator="odeint"):
    dt = t_bar[1] - t_bar[0]
    # tangent linear, direct problem
    # the direct variable's dimension grows with the number of parameters
    dir0 = np.zeros(my_sys.N_param * my_sys.N_dim + my_sys.N_param)
    my_dir_ode = partial(direct_ode, sys=my_sys, dobjective_fun=dobjective_fun)
    dir = solve_ode.integrate(
        my_dir_ode,
        dir0,
        t_bar,
        integrator=integrator,
        args=(t_bar, 1 / dt, y_bar),
    )
    dJdp = 1 / t_bar[-1] * dir[-1, -my_sys.N_param :]
    return dJdp


def adjoint_ode(adj, t, t_bar, inv_dt, y_bar, sys, dobjective_fun):
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
    q_plus = adj[0 : sys.N_dim]

    # interpolate to get the base solution at time t
    y_bar_t = equi_interp(t, t_bar, inv_dt, y_bar)

    # linearize system
    dFdy = -sys.jac(y_bar_t)
    q_plus_dFdy = np.dot(q_plus, dFdy)

    dJdy = dobjective_fun(y_bar_t)

    # Adjoint equations
    # we reverse the sign because integrating backwards
    dq_plus_dt = q_plus_dFdy - dJdy

    # derivatives with respect to parameters
    dFdp = -sys.dfdp(y_bar_t)

    dLdp_dt = np.dot(q_plus, dFdp)

    dadj_dt = np.hstack([dq_plus_dt, dLdp_dt])
    return dadj_dt


def true_adjoint_sensitivity(my_sys, t_bar, y_bar, dobjective_fun, integrator="odeint"):
    dt = t_bar[1] - t_bar[0]
    # adjoint problem
    # the adjoint variable's dimension is always equal to the
    # dimension of the state vector
    adjT = np.zeros(my_sys.N_dim + my_sys.N_param)
    adj_ode = partial(adjoint_ode, sys=my_sys, dobjective_fun=dobjective_fun)
    adj = solve_ode.integrate(
        adj_ode,
        adjT,
        np.flip(t_bar),
        integrator=integrator,
        args=(t_bar, 1 / dt, y_bar),
    )
    dJdp = 1 / t_bar[-1] * adj[-1, -my_sys.N_param :]
    return dJdp


def true_finite_difference_sensitivity(
    my_sys, t_bar, y_bar, h, objective_fun, method, integrator="odeint"
):
    # Calculate numerically
    # Find perturbed solutions
    dJdp = np.zeros((my_sys.N_param,))

    # define which finite difference method to use
    finite_difference = partial(finite_differences, method=method)

    eParam = my_sys.get_eParamVar()

    J = objective_fun(y_bar)
    for param_idx in range(my_sys.N_param):
        param_name = eParam(param_idx).name
        current_param = getattr(my_sys, param_name)

        # perturb from the left
        setattr(my_sys, param_name, current_param - h)
        y_bar_left = solve_ode.integrate(
            my_sys.ode, y_bar[0], t_bar, integrator=integrator
        )
        J_left = objective_fun(y_bar_left[1:])

        # perturb from the right
        setattr(my_sys, param_name, current_param + h)
        y_bar_right = solve_ode.integrate(
            my_sys.ode, y_bar[0], t_bar, integrator=integrator
        )
        J_right = objective_fun(y_bar_right[1:])

        # compute the gradient by finite difference
        dJdp[param_idx] = finite_difference(J, J_right, J_left, h)

        # set the current parameter back
        setattr(my_sys, param_name, current_param)

    return dJdp
