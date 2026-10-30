# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Total direct/adjoint derivatives of a nonlinear index-one DAE cost.

The positive branch satisfies xdot=k*z, qdot=x, z*z=a*x, x0=k*k.
The objective is 2*x(T)+3*q(T)+4*z(T)+k*k+0.2*a*a. Requires CasADi
and NumPy, without ASL or an optimization solver.
"""

import math

import casadi as ca
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator


def create_model():
    """Build the original implicit equations without eliminating the algebraic state."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=0.3)
    model.a = pyo.Param(mutable=True, initialize=1.4)
    model.x = pyo.Var(model.t)
    model.q = pyo.Var(model.t)
    model.z = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.dq = DerivativeVar(model.q)
    model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.k * m.z[t])
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] ** 2 == m.a * m.x[t])
    return model


def run_example():
    """Compose nonlinear ICs and an algebraic/integral cost, checking exact derivatives."""
    model = create_model()
    flow = Simulator(model, package='casadi').compile(
        parameters=[model.k, model.a],
        times=[0, 0.2, 0.7, 1.2],
        integrator='idas',
        integrator_options={'reltol': 1e-10, 'abstol': 1e-12},
    )
    theta = ca.MX.sym('theta', 2)
    result = flow(x0=ca.vertcat(theta[0] ** 2, 0), p=theta, z0=[0.3])
    cost = (
        2 * result['xf'][0, -1]
        + 3 * result['xf'][1, -1]
        + 4 * result['zf'][0, -1]
        + theta[0] ** 2
        + 0.2 * theta[1] ** 2
    )
    objective = ca.Function('objective', [theta], [cost], ['theta'], ['J'])
    point = [0.3, 1.4]
    value = objective(theta=point)['J']
    direct = objective.factory('direct', ['theta', 'fwd:theta'], ['fwd:J'])(
        theta=point, fwd_theta=ca.DM.eye(2)
    )['fwd_J']
    adjoint = objective.factory('adjoint', ['theta', 'adj:J'], ['adj:theta'])(
        theta=point, adj_J=1
    )['adj_theta']
    trajectory = flow(x0=[point[0] ** 2, 0], p=point, z0=0.3)

    # Independent closed-form reduction, used only to check the numerical results.
    k, a = point
    horizon = 1.2
    c = math.sqrt(a) + a * horizon / 2
    b = horizon + math.sqrt(a) * horizon**2 / 2 + a * horizon**3 / 12
    ca_derivative = 1 / (2 * math.sqrt(a)) + horizon / 2
    ba_derivative = horizon**2 / (4 * math.sqrt(a)) + horizon**3 / 12
    expected_state = k * k * c * c / a
    expected_integral = k * k * b
    expected_algebraic = k * c
    expected_cost = (
        2 * expected_state
        + 3 * expected_integral
        + 4 * expected_algebraic
        + k * k
        + 0.2 * a * a
    )
    expected = [
        2 * k * (2 * c * c / a + 3 * b + 1) + 4 * c,
        k * k * (4 * c * ca_derivative / a - 2 * c * c / a**2 + 3 * ba_derivative)
        + 4 * k * ca_derivative
        + 0.4 * a,
    ]
    for actual, reference in (
        (value, expected_cost),
        (trajectory['xf'][0, -1], expected_state),
        (trajectory['xf'][1, -1], expected_integral),
        (trajectory['zf'][0, -1], expected_algebraic),
    ):
        assert math.isclose(float(actual), reference, rel_tol=3e-7, abs_tol=3e-7)
    for index, reference in enumerate(expected):
        assert math.isclose(float(direct[index]), reference, rel_tol=3e-7, abs_tol=3e-7)
        assert math.isclose(
            float(adjoint[index]), reference, rel_tol=3e-7, abs_tol=3e-7
        )
    return trajectory, value, direct, adjoint


if __name__ == '__main__':
    trajectory, value, direct, adjoint = run_example()
    print('Final differential states:', trajectory['xf'][:, -1])
    print('Final consistent algebraic state:', trajectory['zf'][:, -1])
    print('Cost:', value)
    print('Direct total gradient:', direct)
    print('Adjoint total gradient:', adjoint)
