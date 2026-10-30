# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Compiled ODE simulation with direct and adjoint parameter sensitivities.

Requires CasADi and NumPy. No optimization solver or ASL extension is needed.
The extra differential state q integrates x; J=2*x(T)+3*q(T)+k**2.
"""

import math

import casadi as ca
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator


def create_model():
    """Define exponential decay and its integral without numerical IC values."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=-0.7)
    model.x = pyo.Var(model.t)
    model.q = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.dq = DerivativeVar(model.q)
    model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.k * m.x[t])
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    return model


def run_example():
    """Evaluate primal states, a direct parameter direction, and a cost adjoint."""
    model = create_model()
    flow = Simulator(model, package='casadi').compile(
        parameters=[model.k],
        times=[0, 0.2, 0.7, 1.2],
        integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
    )
    trajectory = flow(x0=[1.3, 0], p=[-0.7])['xf']
    direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
        x0=[1.3, 0], p=[-0.7], fwd_p=[1]
    )['fwd_xf']

    # Keep explicit parameter dependence in the symbolic terminal objective.
    x0, parameter = ca.MX.sym('x0', 2), ca.MX.sym('parameter')
    final = flow(x0=x0, p=parameter)['xf'][:, -1]
    objective = ca.Function(
        'objective',
        [x0, parameter],
        [2 * final[0] + 3 * final[1] + parameter**2],
        ['x0', 'p'],
        ['J'],
    )
    adjoint = objective.factory('adjoint', ['x0', 'p', 'adj:J'], ['adj:x0', 'adj:p'])(
        x0=[1.3, 0], p=[-0.7], adj_J=1
    )

    k, T, a = -0.7, 1.2, 1.3
    e = math.exp(k * T)
    dx = T * a * e
    dq = a * ((k * T - 1) * e + 1) / k**2
    assert math.isclose(float(trajectory[0, -1]), a * e, rel_tol=3e-7)
    assert math.isclose(float(direct[0, -1]), dx, rel_tol=3e-7)
    assert math.isclose(float(adjoint['adj_p']), 2 * dx + 3 * dq + 2 * k, rel_tol=3e-7)
    return trajectory, direct, adjoint


if __name__ == '__main__':
    trajectory, direct, adjoint = run_example()
    print('Final state and integral:', trajectory[:, -1])
    print('Direct parameter sensitivities:', direct[:, -1])
    print('Adjoint objective gradient:', adjoint)
