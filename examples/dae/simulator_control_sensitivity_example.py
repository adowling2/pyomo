# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Fixed control amplitudes and total derivatives through x0=k**2.

Requires CasADi and NumPy, without an optimization solver or ASL extension.
The model is x_dot=k*x+u, q_dot=x, with J=2*x(T)+3*q(T)+k**2.
"""

import math

import casadi as ca
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator


def create_model():
    """Define controlled dynamics and an ordinary integral state."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=-0.7)
    model.u = pyo.Var(model.t)
    model.x = pyo.Var(model.t)
    model.q = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.dq = DerivativeVar(model.q)
    model.ode = pyo.Constraint(
        model.t, rule=lambda m, t: m.dx[t] == m.k * m.x[t] + m.u[t]
    )
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    return model


def run_example():
    """Differentiate a terminal/integral cost through controls and a nonlinear IC."""
    model = create_model()
    flow = Simulator(model, package='casadi').compile(
        parameters=[model.k],
        controls=[model.u],
        times=[0, 0.2, 0.5, 0.7, 1.2],
        control_times=[0, 0.2, 0.7, 1.2],
        integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
    )
    theta = ca.MX.sym('theta')
    amplitudes = ca.MX.sym('amplitudes', 1, 3)
    initial = ca.vertcat(theta**2, 0)
    states = flow(x0=initial, p=theta, u=amplitudes)['xf']
    cost = 2 * states[0, -1] + 3 * states[1, -1] + theta**2
    objective = ca.Function(
        'objective', [theta, amplitudes], [cost], ['theta', 'u'], ['J']
    )

    point = -0.7
    controls = ca.DM([[0.1, 0.4, -0.2]])
    trajectory = flow(x0=[point**2, 0], p=point, u=controls)['xf']
    direct = objective.factory('direct', ['theta', 'u', 'fwd:theta'], ['fwd:J'])(
        theta=point, u=controls, fwd_theta=1
    )['fwd_J']
    adjoint = objective.factory(
        'adjoint', ['theta', 'u', 'adj:J'], ['adj:theta', 'adj:u']
    )(theta=point, u=controls, adj_J=1)

    # Exact state, cost, and total rate derivative from affine phase formulas.
    x, q, dx, dq = point**2, 0.0, 2 * point, 0.0
    for duration, amplitude in zip((0.2, 0.5, 0.5), (0.1, 0.4, -0.2)):
        e = math.exp(point * duration)
        g = math.expm1(point * duration) / point
        gp = ((point * duration - 1) * e + 1) / point**2
        h = (g - duration) / point
        hp = (gp - h) / point
        q, dq = q + g * x + amplitude * h, dq + g * dx + gp * x + amplitude * hp
        x, dx = e * x + amplitude * g, e * dx + duration * e * x + amplitude * gp
    expected = 2 * dx + 3 * dq + 2 * point
    assert math.isclose(float(trajectory[0, -1]), x, rel_tol=3e-7)
    assert math.isclose(
        float(objective(theta=point, u=controls)['J']),
        2 * x + 3 * q + point**2,
        rel_tol=3e-7,
    )
    assert math.isclose(float(direct), expected, rel_tol=3e-7)
    assert math.isclose(float(adjoint['adj_theta']), expected, rel_tol=3e-7)
    return trajectory, direct, adjoint


if __name__ == '__main__':
    trajectory, direct, adjoint = run_example()
    print('Final state and integral:', trajectory[:, -1])
    print('Direct total rate derivative:', direct)
    print('Adjoint total derivatives:', adjoint)
