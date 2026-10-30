# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Total derivatives through nonlinear DAE initial states and shared controls.

The positive branch satisfies xdot=k*z, qdot=x, z*z=a*x+u. One design
variable sets k, x0=k*k and every interval amplitude u_j=k*k*c_j.
Requires CasADi; neither ASL nor an optimization solver is used.
"""

import math

import casadi as ca
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator


def run_example():
    """Check total direct and adjoint cost derivatives against exact phase formulas."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=0.8)
    model.a = pyo.Param(initialize=1.4)
    model.x, model.q, model.z = pyo.Var(model.t), pyo.Var(model.t), pyo.Var(model.t)
    model.u = pyo.Var(model.t)
    model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)
    model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.k * m.z[t])
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    model.alg = pyo.Constraint(
        model.t, rule=lambda m, t: m.z[t] ** 2 == m.a * m.x[t] + m.u[t]
    )
    bounds, amplitudes = [0, 0.2, 0.7, 1.2], [0.1, 0.4, -0.2]
    flow = Simulator(model, package='casadi').compile(
        parameters=[model.k],
        controls=[model.u],
        control_times=bounds,
        times=bounds,
        integrator='idas',
        integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
    )
    theta = ca.MX.sym('theta')
    u = theta**2 * ca.DM([amplitudes])
    states = flow(x0=ca.vertcat(theta**2, 0), p=theta, z0=1, u=u)
    cost = (
        2 * states['xf'][0, -1]
        + 3 * states['xf'][1, -1]
        + 4 * states['zf'][0, -1]
        + theta**2
        + 0.2
        * sum(
            (end - start) * u[0, j] ** 2
            for j, (start, end) in enumerate(zip(bounds, bounds[1:]))
        )
    )
    objective = ca.Function('objective', [theta], [cost], ['theta'], ['J'])
    point = 0.8
    value = objective(theta=point)['J']
    direct = objective.factory('direct', ['theta', 'fwd:theta'], ['fwd:J'])(
        theta=point, fwd_theta=1
    )['fwd_J']
    adjoint = objective.factory('adjoint', ['theta', 'adj:J'], ['adj:theta'])(
        theta=point, adj_J=1
    )['adj_theta']

    # Exact scaled phase solution: x=theta^2*X, q=theta^2*Q, z=theta*Z.
    x, q, energy, a = 1, 0, 0, 1.4
    for c, start, end in zip(amplitudes, bounds, bounds[1:]):
        d = end - start
        z = math.sqrt(a * x + c)
        q += x * d + z * d * d / 2 + a * d**3 / 12
        x += z * d + a * d * d / 4
        energy += d * c * c
    z = math.sqrt(a * x + amplitudes[-1])
    expected_value = (
        point**2 * (2 * x + 3 * q + 1) + 4 * point * z + 0.2 * point**4 * energy
    )
    expected_gradient = (
        2 * point * (2 * x + 3 * q + 1) + 4 * z + 0.8 * point**3 * energy
    )
    for actual, reference in (
        (value, expected_value),
        (direct, expected_gradient),
        (adjoint, expected_gradient),
    ):
        assert math.isclose(float(actual), reference, rel_tol=3e-7, abs_tol=3e-7)
    return value, direct, adjoint


if __name__ == '__main__':
    value, direct, adjoint = run_example()
    print('Cost:', value)
    print('Direct total derivative:', direct)
    print('Adjoint total derivative:', adjoint)
