# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""ODE closeout checks with nonlinear controls and six orders of state scaling."""

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available, check_min_version
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.simulator import casadi as ca, casadi_available


def scaled_model():
    """Build rational decay in two state units without assigning any state values."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=0.4)
    model.u = pyo.Var(model.t)
    model.scale = pyo.Param([0, 1], initialize={0: 1, 1: 1e6})
    model.x = pyo.Var(model.t, [0, 1])
    model.dx = DerivativeVar(model.x, wrt=model.t)
    model.ode = pyo.Constraint(
        model.t,
        [0, 1],
        rule=lambda m, t, i: m.dx[t, i]
        == -(m.k + m.u[t]) * m.x[t, i] ** 2 / m.scale[i],
    )
    return model


def rational_reference(times, boundaries, amplitudes, rate, initial):
    """Return exact rational states and independent x0/p/u partial derivatives."""
    scales = np.array([1, 1e6])
    lengths = np.maximum(
        0,
        np.minimum(np.asarray(times)[:, None], np.asarray(boundaries[1:]))
        - np.asarray(boundaries[:-1]),
    )
    exposure = rate * np.asarray(times) + lengths @ np.asarray(amplitudes)
    normalized = np.asarray(initial) / scales
    denominator = 1 + normalized[:, None] * exposure
    states = np.asarray(initial)[:, None] / denominator
    coefficient = -((states / scales[:, None]) ** 2) * scales[:, None]
    return {
        'xf': states,
        'x0': 1 / denominator**2,
        'p': coefficient * np.asarray(times),
        'u': coefficient[:, :, None] * lengths[None, :, :],
    }


def check_scaled_controls():
    """Measure scaled primal/JVP/VJP errors and identities over tolerance refinement."""
    times, boundaries = [0, 0.1, 0.2, 0.5, 0.7, 1.2], [0, 0.2, 0.7, 1.2]
    initial, rate, u = [0.8, 1.3e6], 0.4, np.array([[-0.15, 0.2, 0.05]])
    scales = np.array([[1], [1e6]])
    reference = rational_reference(times, boundaries, u[0], rate, initial)
    seeds_x0 = np.array([[0.2, -0.1], [0.3e6, -0.2e6]])
    seeds_p = np.array([[0.3, -0.2]])
    seeds_u = np.array([[0.1, -0.2, 0.3, -0.3, 0.1, 0.2]])
    weights = (
        np.array(
            [
                [0.1, -0.2, 0.4, 0.3, -0.1, 0.5, 0.2, 0.3, -0.1, 0.4, 0.1, -0.2],
                [0.3, 0.1, -0.2, 0.4, 0.2, -0.1, -0.1, 0.2, 0.3, -0.2, 0.5, 0.1],
            ]
        )
        / scales
    )
    model = scaled_model()
    simulator = Simulator(model, package='casadi')
    rows = []
    for tolerance in (1e-3, 1e-7, 1e-11):
        flow = simulator.compile(
            parameters=[model.k],
            times=times,
            controls=[model.u],
            control_times=boundaries,
            integrator_options={'reltol': tolerance, 'abstol': tolerance / 100},
        )
        value = flow(x0=initial, p=rate, u=u)['xf']
        direct = flow.forward(2)(
            x0=initial,
            p=rate,
            u=u,
            out_xf=value,
            fwd_x0=seeds_x0,
            fwd_p=seeds_p,
            fwd_u=seeds_u,
        )['fwd_xf']
        reverse = flow.reverse(2)(x0=initial, p=rate, u=u, out_xf=value, adj_xf=weights)
        row = {
            'reltol': tolerance,
            'abstol': tolerance / 100,
            'primal_scaled_error': float(
                np.max(abs((np.asarray(value) - reference['xf']) / scales))
            ),
            'direct_scaled_error': 0.0,
            'adjoint_scaled_error': 0.0,
            'duality_error': 0.0,
        }
        for i in range(2):
            direction_u = seeds_u[0, 3 * i : 3 * (i + 1)]
            expected_direct = (
                reference['x0'] * seeds_x0[:, i, None]
                + reference['p'] * seeds_p[0, i]
                + np.einsum('itj,j->it', reference['u'], direction_u)
            )
            block = np.asarray(direct)[:, 6 * i : 6 * (i + 1)]
            row['direct_scaled_error'] = max(
                row['direct_scaled_error'],
                float(np.max(abs((block - expected_direct) / scales))),
            )
            weight = weights[:, 6 * i : 6 * (i + 1)]
            expected_x0 = np.sum(weight * reference['x0'], axis=1)
            expected_p = np.sum(weight * reference['p'])
            expected_u = np.einsum('it,itj->j', weight, reference['u'])
            errors = [
                np.max(
                    abs(
                        (np.asarray(reverse['adj_x0'])[:, i] - expected_x0)
                        * scales[:, 0]
                    )
                ),
                abs(float(reverse['adj_p'][0, i]) - expected_p),
                np.max(
                    abs(
                        np.asarray(reverse['adj_u'])[0, 3 * i : 3 * (i + 1)]
                        - expected_u
                    )
                ),
            ]
            row['adjoint_scaled_error'] = max(
                row['adjoint_scaled_error'], float(max(errors))
            )
            for j in range(2):
                left = np.sum(weight * np.asarray(direct)[:, 6 * j : 6 * (j + 1)])
                right = (
                    seeds_x0[:, j] @ np.asarray(reverse['adj_x0'])[:, i]
                    + seeds_p[0, j] * float(reverse['adj_p'][0, i])
                    + seeds_u[0, 3 * j : 3 * (j + 1)]
                    @ np.asarray(reverse['adj_u'])[0, 3 * i : 3 * (i + 1)]
                )
                row['duality_error'] = max(
                    row['duality_error'], abs(float(left - right))
                )
        rows.append(row)
    return rows


def check_scaled_total_derivative():
    """Compare a shared IC/control composition with exact gradients and a FD sweep."""
    times, boundaries = [0, 0.2, 0.5, 0.7, 1.2], [0, 0.2, 0.7, 1.2]
    model = scaled_model()
    flow = Simulator(model, package='casadi').compile(
        parameters=[model.k],
        times=times,
        controls=[model.u],
        control_times=boundaries,
        integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
    )
    theta = ca.MX.sym('theta', 3)
    controls = ca.horzcat(theta[1], theta[2], theta[1] + theta[2])
    initial = ca.vertcat(1 + theta[0] ** 2, 1e6 * (1 + 0.2 * theta[1] ** 2))
    states = flow(x0=initial, p=theta[0], u=controls)['xf']
    weights = np.array(
        [[0.1, -0.2, 0.3, 0.4, 0.5], [0.2, 0.1, -0.3, 0.2, 0.4]]
    ) / np.array([[1], [1e6]])
    cost = (
        ca.dot(ca.vec(ca.DM(weights)), ca.vec(states))
        + theta[0] ** 2
        + 0.1 * ca.sumsqr(controls)
    )
    objective = ca.Function('objective', [theta], [cost], ['theta'], ['J'])
    point = np.array([0.4, -0.15, 0.2])
    u = np.array([point[1], point[2], point[1] + point[2]])
    x0 = [1 + point[0] ** 2, 1e6 * (1 + 0.2 * point[1] ** 2)]
    reference = rational_reference(times, boundaries, u, point[0], x0)
    gx0 = np.sum(weights * reference['x0'], axis=1)
    gu = np.einsum('it,itj->j', weights, reference['u']) + 0.2 * u
    expected = np.array(
        [
            np.sum(weights * reference['p']) + 2 * point[0] * (1 + gx0[0]),
            gu[0] + gu[2] + gx0[1] * 0.4e6 * point[1],
            gu[1] + gu[2],
        ]
    )
    value = objective(theta=point)['J']
    direct = objective.forward(3)(theta=point, out_J=value, fwd_theta=np.eye(3))[
        'fwd_J'
    ]
    reverse = objective.reverse(1)(theta=point, out_J=value, adj_J=1)['adj_theta']
    differences = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5):
        derivative = np.array(
            [
                float(
                    (objective(point + step * d) - objective(point - step * d))
                    / (2 * step)
                )
                for d in np.eye(3)
            ]
        )
        differences.append(
            {'step': step, 'max_error': float(np.max(abs(derivative - expected)))}
        )
    return {
        'analytic_gradient': expected.tolist(),
        'direct_error': float(np.max(abs(np.asarray(direct).ravel() - expected))),
        'adjoint_error': float(np.max(abs(np.asarray(reverse).ravel() - expected))),
        'finite_differences': differences,
    }


@unittest.skipUnless(
    numpy_available and casadi_available and check_min_version(ca, '3.8.1'),
    'NumPy and CasADi 3.8.1 or later are required',
)
class TestODEQualification(unittest.TestCase):
    """Qualify nonlinear controls, state scaling and total derivative composition."""

    def test_scaled_nonlinear_controls(self):
        """Both multi-direction modes converge to exact scaled state derivatives."""
        rows = check_scaled_controls()
        for key in (
            'primal_scaled_error',
            'direct_scaled_error',
            'adjoint_scaled_error',
            'duality_error',
        ):
            with self.subTest(quantity=key):
                self.assertLess(rows[-1][key], 3e-7)
                self.assertLess(rows[-1][key], rows[0][key] / 10)

    def test_scaled_total_derivatives(self):
        """Nonlinear shared IC/control derivatives match analytic and discrete checks."""
        row = check_scaled_total_derivative()
        self.assertLess(row['direct_error'], 3e-7)
        self.assertLess(row['adjoint_error'], 3e-7)
        errors = [record['max_error'] for record in row['finite_differences']]
        self.assertLess(min(errors), 1e-6)
        self.assertLess(min(errors[1:]), errors[0] / 10)


if __name__ == '__main__':
    unittest.main()
