# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Compiled piecewise controls and total initial-condition sensitivity checks."""

from copy import deepcopy
import math

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available, check_min_version
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import casadi as ca, casadi_available


def controlled_model(start=0, end=1.2, kind='var', integral=True, multiple=False):
    """Build affine controlled dynamics and an optional integral state."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(start, end))
    model.k = pyo.Param(mutable=True, initialize=-0.7)
    model.x = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    if kind == 'var':
        model.u = pyo.Var(model.t)
    else:
        model.u = pyo.Param(model.t, default=99, mutable=kind == 'mutable')
    if multiple:
        model.v = pyo.Param(model.t, default=0, mutable=True)
    model.ode = pyo.Constraint(
        model.t,
        rule=lambda m, t: m.dx[t]
        == m.k * m.x[t] + m.u[t] + (2 * m.v[t] if multiple else 0),
    )
    if integral:
        model.q = pyo.Var(model.t)
        model.dq = DerivativeVar(model.q)
        model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    return model


def affine_reference(times, boundaries, amplitudes, rate, initial=(0.3, 0)):
    """Integrate exact affine phase formulas and their independent derivatives."""
    values, parameters, initials, controls = [], [], [], []
    for time in times:
        x, q = initial
        xp, qp = 0.0, 0.0
        xx, qx = 1.0, 0.0
        xu, qu = np.zeros(len(amplitudes)), np.zeros(len(amplitudes))
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if time <= start:
                break
            duration = min(time, end) - start
            e = math.exp(rate * duration)
            if rate:
                g = math.expm1(rate * duration) / rate
                gp = ((rate * duration - 1) * e + 1) / rate**2
                h = (g - duration) / rate
                hp = (gp - h) / rate
            else:
                g, gp, h, hp = (
                    duration,
                    duration**2 / 2,
                    duration**2 / 2,
                    duration**3 / 6,
                )
            q, qp, qx, qu = (
                q + g * x + amplitudes[index] * h,
                qp + g * xp + gp * x + amplitudes[index] * hp,
                qx + g * xx,
                qu + g * xu,
            )
            qu[index] += h
            x, xp, xx, xu = (
                e * x + amplitudes[index] * g,
                e * xp + duration * e * x + amplitudes[index] * gp,
                e * xx,
                e * xu,
            )
            xu[index] += g
        values.append([x, q])
        parameters.append([xp, qp])
        initials.append([[xx, 0], [qx, 1]])
        controls.append([xu, qu])
    return {
        'xf': np.asarray(values).T,
        'p': np.asarray(parameters).T,
        'x0': np.asarray(initials).transpose(1, 0, 2),
        'u': np.asarray(controls).transpose(1, 0, 2),
    }


@unittest.skipUnless(
    numpy_available and casadi_available and check_min_version(ca, '3.8.1'),
    'NumPy and CasADi 3.8.1 or later are required',
)
class TestCompiledControls(unittest.TestCase):
    """Check phase composition through the public compile interface."""

    def compile(self, model, times, boundaries, **kwds):
        """Compile with fixed observation and control grids and tight tolerances."""
        kwds.setdefault('parameters', [model.k])
        kwds.setdefault('controls', [model.u])
        kwds.setdefault('integrator_options', {'reltol': 1e-11, 'abstol': 1e-13})
        return Simulator(model, package='casadi').compile(
            times=times, control_times=boundaries, **kwds
        )

    def assert_close(self, actual, expected):
        """Check dimensions and finite analytic agreement."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)

    def test_affine_controls_and_integral(self):
        """Check state, integral, parameter and amplitude derivatives at switches."""
        boundaries = [0, 0.2, 0.7, 1.2]
        times = [0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.2]
        amplitudes = np.array([[0.1, 0.4, -0.2]])
        weights = np.array(
            [
                [0.2, -0.1, 0.3, 0.1, -0.2, 0.4, 0.7],
                [-0.3, 0.1, 0.4, 0.2, 0.1, -0.1, 0.5],
            ]
        )
        for rate in (0, -0.7):
            with self.subTest(rate=rate):
                model = controlled_model()
                flow = self.compile(model, times, boundaries)
                reference = affine_reference(times, boundaries, amplitudes[0], rate)
                value = flow(x0=[0.3, 0], p=rate, u=amplitudes)['xf']
                self.assertEqual(flow.name_in(), ['x0', 'p', 'u'])
                self.assertEqual(flow.size_in('u'), (1, 3))
                self.assert_close(value, reference['xf'])
                direct = flow.factory('direct', ['x0', 'p', 'u', 'fwd:p'], ['fwd:xf'])(
                    x0=[0.3, 0], p=rate, u=amplitudes, fwd_p=1
                )['fwd_xf']
                self.assert_close(direct, reference['p'])
                for index in range(3):
                    seed = np.eye(3)[index : index + 1]
                    tangent = flow.factory(
                        'control', ['x0', 'p', 'u', 'fwd:u'], ['fwd:xf']
                    )(x0=[0.3, 0], p=rate, u=amplitudes, fwd_u=seed)['fwd_xf']
                    self.assert_close(tangent, reference['u'][:, :, index])
                reverse = flow.factory(
                    'adjoint', ['x0', 'p', 'u', 'adj:xf'], ['adj:x0', 'adj:p', 'adj:u']
                )(x0=[0.3, 0], p=rate, u=amplitudes, adj_xf=weights)
                self.assert_close(
                    reverse['adj_p'], [[np.sum(weights * reference['p'])]]
                )
                self.assert_close(
                    reverse['adj_x0'],
                    np.einsum('it,itp->p', weights, reference['x0']).reshape(2, 1),
                )
                self.assert_close(
                    reverse['adj_u'],
                    np.einsum('it,itp->p', weights, reference['u']).reshape(1, 3),
                )

    def test_sparse_and_initial_only_observations(self):
        """Unobserved phases propagate state; future phases have zero derivatives."""
        boundaries, amplitudes = [0, 0.2, 0.7, 1.2], np.array([[0.1, 0.4, -0.2]])
        for times in (
            [0],
            [0.1],
            [1.2],
            [0.2, 0.7],
            [0, 0.2, 0.7, 1.2],
            [np.nextafter(0.2, 1), 0.5, 1.2],
            [np.nextafter(0.2, 0), 0.5, 1.2],
            [np.nextafter(0.2, 0), 0.2, np.nextafter(0.2, 1), 1.2],
        ):
            with self.subTest(times=times):
                flow = self.compile(controlled_model(), times, boundaries)
                reference = affine_reference(times, boundaries, amplitudes[0], -0.7)
                self.assert_close(
                    flow(x0=[0.3, 0], p=-0.7, u=amplitudes)['xf'], reference['xf']
                )
                weights = np.ones((2, len(times)))
                reverse = flow.factory(
                    'gradient', ['x0', 'p', 'u', 'adj:xf'], ['adj:u', 'adj:p']
                )(x0=[0.3, 0], p=-0.7, u=amplitudes, adj_xf=weights)
                self.assert_close(
                    reverse['adj_u'],
                    np.einsum('it,itp->p', weights, reference['u']).reshape(1, 3),
                )
                self.assert_close(reverse['adj_p'], [[np.sum(reference['p'])]])

    def test_dense_observations(self):
        """Exercise accumulated maps beyond the native unrolling block size."""
        times, boundaries, u = (
            np.linspace(0, 1.2, 51),
            [0, 0.2, 0.7, 1.2],
            np.array([[0.1, 0.4, -0.2]]),
        )
        flow = self.compile(controlled_model(), times, boundaries)
        reference = affine_reference(times, boundaries, u[0], -0.7)
        value = flow(x0=[0.3, 0], p=-0.7, u=u)['xf']
        self.assert_close(value, reference['xf'])
        seed = np.array([[0.1, -0.2, 0.3]])
        tangent = flow.factory(
            'direct', ['x0', 'p', 'u', 'fwd:p', 'fwd:u'], ['fwd:xf']
        )(x0=[0.3, 0], p=-0.7, u=u, fwd_p=0.2, fwd_u=seed)['fwd_xf']
        self.assert_close(
            tangent,
            0.2 * reference['p'] + np.einsum('itp,p->it', reference['u'], seed[0]),
        )
        weights = np.ones((2, len(times))) / len(times)
        gradient = flow.factory(
            'adjoint', ['x0', 'p', 'u', 'adj:xf'], ['adj:p', 'adj:u']
        )(x0=[0.3, 0], p=-0.7, u=u, adj_xf=weights)
        self.assert_close(gradient['adj_p'], [[np.sum(weights * reference['p'])]])
        self.assert_close(
            gradient['adj_u'],
            np.einsum('it,itp->p', weights, reference['u']).reshape(1, 3),
        )

    def test_nonlinear_control_amplitudes(self):
        """Control derivatives include nonlinear and stationary amplitude dependence."""
        model = controlled_model(integral=False)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t] ** 2)
        times, boundaries, u = (
            np.array([0, 0.2, 0.5, 0.7, 1.2]),
            [0, 0.2, 0.7, 1.2],
            np.array([[0, 0.4, -0.2]]),
        )
        flow = self.compile(model, times, boundaries, parameters=[])
        lengths = np.maximum(
            0,
            np.minimum(times[:, None], np.array(boundaries[1:]))
            - np.array(boundaries[:-1]),
        )
        self.assert_close(
            flow(x0=0.3, p=[], u=u)['xf'], (0.3 + lengths @ (u[0] ** 2)).reshape(1, -1)
        )
        tangent = flow.factory('direct', ['x0', 'p', 'u', 'fwd:u'], ['fwd:xf'])(
            x0=0.3, p=[], u=u, fwd_u=ca.DM.ones(1, 3)
        )['fwd_xf']
        self.assert_close(tangent, (lengths @ (2 * u[0])).reshape(1, -1))
        weights = np.array([[0.2, -0.1, 0.3, 0.4, 0.7]])
        gradient = flow.factory('adjoint', ['x0', 'p', 'u', 'adj:xf'], ['adj:u'])(
            x0=0.3, p=[], u=u, adj_xf=weights
        )['adj_u']
        self.assert_close(gradient, 2 * u * (weights @ lengths))

    def test_narrow_control_interval(self):
        """Do not round away a finite pulse in adjacent floating-point boundaries."""
        end = np.nextafter(0.2, 1.0)
        model = controlled_model(end=1, integral=False)
        model.k = 0
        times, boundaries = np.array([0, 0.2, end, 0.7, 1]), [0, 0.2, end, 1]
        amplitudes = np.array([[0, 1e16, 0]])
        flow = self.compile(model, times, boundaries, parameters=[])
        width = end - 0.2
        expected = 0.3 + 1e16 * np.maximum(0, np.minimum(times, end) - 0.2)
        self.assert_close(
            flow(x0=0.3, p=[], u=amplitudes)['xf'], expected.reshape(1, -1)
        )
        direct = flow.factory('direct', ['x0', 'p', 'u', 'fwd:u'], ['fwd:xf'])(
            x0=0.3, p=[], u=amplitudes, fwd_u=ca.DM([[0, 1e16, 0]])
        )['fwd_xf']
        self.assert_close(direct, (expected - 0.3).reshape(1, -1))
        gradient = flow.factory('adjoint', ['x0', 'p', 'u', 'adj:xf'], ['adj:u'])(
            x0=0.3, p=[], u=amplitudes, adj_xf=ca.DM([[0, 0, 0, 0, 1]])
        )['adj_u']
        self.assertAlmostEqual(float(gradient[0, 1]) * 1e16, width * 1e16, delta=3e-7)

    def test_control_component_types(self):
        """Time-indexed Vars and mutable/immutable Params retain symbolic amplitudes."""
        for kind in ('var', 'mutable', 'immutable'):
            with self.subTest(kind=kind):
                model = controlled_model(kind=kind, integral=False)
                flow = self.compile(model, [0, 0.5, 1.2], [0, 0.5, 1.2])
                value = flow(x0=0.3, p=-0.7, u=ca.DM([[0.2, 0.4]]))['xf']
                reference = affine_reference(
                    [0, 0.5, 1.2], [0, 0.5, 1.2], [0.2, 0.4], -0.7
                )
                self.assert_close(value, reference['xf'][:1])
                gradient = flow.factory(
                    'gradient', ['x0', 'p', 'u', 'adj:xf'], ['adj:u']
                )(x0=0.3, p=-0.7, u=ca.DM([[0.2, 0.4]]), adj_xf=ca.DM([[0, 0, 1]]))[
                    'adj_u'
                ]
                self.assert_close(gradient, reference['u'][0, -1, :].reshape(1, 2))
                if kind != 'var':
                    self.assertEqual(pyo.value(model.u[0]), 99)

    def test_multiple_controls_directions_and_duality(self):
        """Verify control ordering and block layouts for multiple JVPs and VJPs."""
        model = controlled_model(multiple=True)
        times, boundaries = [0, 0.2, 0.5, 0.7, 1.2], [0, 0.2, 0.7, 1.2]
        amplitudes = np.array([[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]])
        flow = self.compile(model, times, boundaries, controls=[model.v, model.u])
        reference = affine_reference(
            times, boundaries, amplitudes[1] + 2 * amplitudes[0], -0.7
        )
        state_seeds = np.array([[0.2, -0.1], [0.3, 0.5]])
        parameter_seeds = np.array([[0.3, -0.2]])
        control_seeds = np.array(
            [[0.1, 0.2, -0.3, 0.2, -0.1, 0.4], [-0.2, 0.1, 0.3, 0.1, 0.2, -0.1]]
        )
        weights = np.array(
            [
                [0.2, -0.3, 0.1, 0.4, 0.7, 0.3, 0.1, -0.2, 0.4, -0.1],
                [-0.1, 0.4, 0.2, -0.3, 0.1, 0.2, 0.3, 0.1, -0.2, 0.4],
            ]
        )
        value = flow(x0=[0.3, 0], p=-0.7, u=amplitudes)['xf']
        self.assert_close(value, reference['xf'])
        direct = flow.forward(2)(
            x0=[0.3, 0],
            p=-0.7,
            u=amplitudes,
            out_xf=value,
            fwd_x0=state_seeds,
            fwd_p=parameter_seeds,
            fwd_u=control_seeds,
        )['fwd_xf']
        reverse = flow.reverse(2)(
            x0=[0.3, 0], p=-0.7, u=amplitudes, out_xf=value, adj_xf=weights
        )
        for i in range(2):
            control_seed = control_seeds[:, 3 * i : 3 * (i + 1)]
            expected = (
                np.einsum('itp,p->it', reference['x0'], state_seeds[:, i])
                + reference['p'] * parameter_seeds[0, i]
                + np.einsum(
                    'itp,p->it', reference['u'], control_seed[1] + 2 * control_seed[0]
                )
            )
            self.assert_close(direct[:, 5 * i : 5 * (i + 1)], expected)
            weight = weights[:, 5 * i : 5 * (i + 1)]
            gu = np.einsum('it,itp->p', weight, reference['u'])
            self.assert_close(
                reverse['adj_u'][:, 3 * i : 3 * (i + 1)], np.vstack([2 * gu, gu])
            )
            self.assert_close(
                reverse['adj_p'][:, i], [[np.sum(weight * reference['p'])]]
            )
            self.assert_close(
                reverse['adj_x0'][:, i],
                np.einsum('it,itp->p', weight, reference['x0']).reshape(2, 1),
            )
            for j in range(2):
                left = np.sum(weight * np.asarray(direct)[:, 5 * j : 5 * (j + 1)])
                right = (
                    np.sum(state_seeds[:, j] * np.asarray(reverse['adj_x0'])[:, i])
                    + parameter_seeds[0, j] * float(reverse['adj_p'][0, i])
                    + np.sum(
                        control_seeds[:, 3 * j : 3 * (j + 1)]
                        * np.asarray(reverse['adj_u'])[:, 3 * i : 3 * (i + 1)]
                    )
                )
                self.assertAlmostEqual(float(left), float(right), delta=3e-7)

    def test_total_derivatives_through_initial_conditions(self):
        """Include nonlinear ICs, repeated amplitudes, integral and explicit cost terms."""
        model = controlled_model()
        boundaries = [0, 0.2, 0.7, 1.2]
        flow = self.compile(model, [0.2, 0.7, 1.2], boundaries)
        theta = ca.MX.sym('theta', 3)
        amplitudes = ca.horzcat(theta[1], theta[2], theta[1] + theta[2])
        initial = ca.vertcat(theta[0] ** 2 + 0.1 * theta[1], theta[1] ** 2)
        final = flow(x0=initial, p=theta[0], u=amplitudes)['xf'][:, -1]
        cost = (
            2 * final[0]
            + 3 * final[1]
            + theta[0] ** 2
            + 0.2 * ca.dot(ca.DM(np.diff(boundaries)), amplitudes.T**2)
        )
        total = ca.Function('total', [theta], [cost], ['theta'], ['J'])
        point = np.array([-0.7, 0.1, 0.4])
        u = np.array([point[1], point[2], point[1] + point[2]])
        x0 = [point[0] ** 2 + 0.1 * point[1], point[1] ** 2]
        reference = affine_reference([1.2], boundaries, u, point[0], x0)
        weights = np.array([2, 3])
        gx0 = weights @ reference['x0'][:, 0, :]
        gp = weights @ reference['p'][:, 0] + 2 * point[0]
        gu = weights @ reference['u'][:, 0, :] + 0.4 * np.diff(boundaries) * u
        expected = np.array(
            [
                gp + gx0[0] * 2 * point[0],
                gu[0] + gu[2] + 0.1 * gx0[0] + 2 * point[1] * gx0[1],
                gu[1] + gu[2],
            ]
        )
        expected_cost = (
            weights @ reference['xf'][:, 0]
            + point[0] ** 2
            + 0.2 * np.sum(np.diff(boundaries) * u**2)
        )
        value = total(theta=point)['J']
        self.assert_close(value, [[expected_cost]])
        seeds = np.array([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]])
        direct = total.forward(2)(theta=point, out_J=value, fwd_theta=seeds)['fwd_J']
        reverse = total.reverse(2)(theta=point, out_J=value, adj_J=ca.DM([[0.3, 1.1]]))[
            'adj_theta'
        ]
        self.assert_close(direct, (expected @ seeds).reshape(1, 2))
        self.assert_close(reverse, expected.reshape(3, 1) * np.array([[0.3, 1.1]]))
        partial = flow.factory(
            'partial', ['x0', 'p', 'u', 'adj:xf'], ['adj:x0', 'adj:p', 'adj:u']
        )(
            x0=x0,
            p=point[0],
            u=ca.DM(u.reshape(1, 3)),
            adj_xf=ca.DM([[0, 0, 2], [0, 0, 3]]),
        )
        self.assertAlmostEqual(
            float(partial['adj_p'])
            + 2 * point[0]
            + float(partial['adj_x0'][0]) * 2 * point[0],
            expected[0],
            delta=3e-7,
        )
        self.assertGreater(abs(expected[0] - gp), 0.1)
        errors = []
        for step in (1e-2, 1e-3, 1e-4, 1e-5):
            difference = np.array(
                [
                    float(
                        (
                            total(point + step * direction)
                            - total(point - step * direction)
                        )
                        / (2 * step)
                    )
                    for direction in np.eye(3)
                ]
            )
            errors.append(float(np.max(abs(difference - expected))))
        self.assertLess(min(errors), 1e-6)
        self.assertLess(min(errors[1:]), errors[0] / 10)

    def test_shifted_non_autonomous_controls(self):
        """Phase maps use physical time and do not rescale the control amplitudes."""
        model = controlled_model(2, 3.2, integral=False)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.k * t + m.u[t]
        )
        times, boundaries, u = (
            np.array([2, 2.2, 2.5, 2.7, 3.2]),
            [2, 2.2, 2.7, 3.2],
            np.array([[0.1, 0.4, -0.2]]),
        )
        flow = self.compile(model, times, boundaries)
        lengths = np.maximum(
            0,
            np.minimum(times[:, None], np.array(boundaries[1:]))
            - np.array(boundaries[:-1]),
        )
        expected = 0.3 - 0.7 * (times**2 - 4) / 2 + lengths @ u[0]
        self.assert_close(flow(x0=0.3, p=-0.7, u=u)['xf'], expected.reshape(1, -1))
        gradient = flow.factory(
            'gradient', ['x0', 'p', 'u', 'adj:xf'], ['adj:p', 'adj:u']
        )(x0=0.3, p=-0.7, u=u, adj_xf=np.ones((1, len(times))))
        self.assert_close(gradient['adj_p'], [[np.sum((times**2 - 4) / 2)]])
        self.assert_close(gradient['adj_u'], np.sum(lengths, axis=0).reshape(1, 3))
        direct = flow.factory('direct', ['x0', 'p', 'u', 'fwd:p', 'fwd:u'], ['fwd:xf'])(
            x0=0.3, p=-0.7, u=u, fwd_p=0.2, fwd_u=ca.DM([[0.1, -0.2, 0.3]])
        )['fwd_xf']
        self.assert_close(
            direct,
            (0.2 * (times**2 - 4) / 2 + lengths @ np.array([0.1, -0.2, 0.3])).reshape(
                1, -1
            ),
        )

    def test_duration_parameter_on_normalized_time(self):
        """A model duration parameter differentiates a fixed normalized control grid."""
        model = controlled_model(end=1)
        model.horizon = pyo.Param(mutable=True, initialize=1.2)
        model.del_component(model.ode)
        model.del_component(model.integral)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.horizon * (m.k * m.x[t] + m.u[t])
        )
        model.integral = pyo.Constraint(
            model.t, rule=lambda m, t: m.dq[t] == m.horizon * m.x[t]
        )
        times, boundaries, u = (
            [0, 0.2, 0.5, 0.7, 1],
            [0, 0.2, 0.7, 1],
            np.array([[0.1, 0.4, -0.2]]),
        )
        horizon, rate = 1.2, -0.7
        flow = self.compile(
            model, times, boundaries, parameters=[model.k, model.horizon]
        )
        reference = affine_reference(times, boundaries, horizon * u[0], horizon * rate)
        expected = reference['xf'].copy()
        expected[1] *= horizon
        self.assert_close(flow(x0=[0.3, 0], p=[rate, horizon], u=u)['xf'], expected)
        duration_derivative = rate * reference['p'] + np.einsum(
            'itp,p->it', reference['u'], u[0]
        )
        duration_derivative[1] = reference['xf'][1] + horizon * duration_derivative[1]
        direct = flow.factory('direct', ['x0', 'p', 'u', 'fwd:p'], ['fwd:xf'])(
            x0=[0.3, 0], p=[rate, horizon], u=u, fwd_p=[0, 1]
        )['fwd_xf']
        self.assert_close(direct, duration_derivative)
        weights = np.ones((2, len(times)))
        gradient = flow.factory('adjoint', ['x0', 'p', 'u', 'adj:xf'], ['adj:p'])(
            x0=[0.3, 0], p=[rate, horizon], u=u, adj_xf=weights
        )['adj_p']
        rate_derivative = reference['p'] * np.array([[horizon], [horizon**2]])
        self.assert_close(
            gradient,
            [
                [np.sum(weights * rate_derivative)],
                [np.sum(weights * duration_derivative)],
            ],
        )

    def test_controls_without_parameters(self):
        """An empty parameter input preserves direct and adjoint amplitude products."""
        model = controlled_model(integral=False)
        flow = self.compile(model, [0, 0.7, 1.2], [0, 0.7, 1.2], parameters=[])
        u = np.array([[0.1, 0.4]])
        reference = affine_reference([0, 0.7, 1.2], [0, 0.7, 1.2], u[0], -0.7)
        self.assertEqual(flow.size_in('p'), (0, 1))
        self.assert_close(flow(x0=0.3, p=[], u=u)['xf'], reference['xf'][:1])
        reverse = flow.factory(
            'adjoint', ['x0', 'p', 'u', 'adj:xf'], ['adj:p', 'adj:u']
        )(x0=0.3, p=[], u=u, adj_xf=ca.DM([[0, 0, 1]]))
        self.assertEqual(reverse['adj_p'].numel(), 0)
        self.assert_close(reverse['adj_u'], reference['u'][0, -1, :].reshape(1, 2))

    def test_invalid_controls(self):
        """Reject missing, duplicate, foreign, unused and non-time-only selections."""
        model = controlled_model()
        model.unused = pyo.Var(model.t)
        model.multiple = pyo.Var(model.t, [0, 1])
        model.scalar = pyo.Var(initialize=1)
        other = controlled_model()
        sim = Simulator(model, package='casadi')
        for controls in (
            [model.u, model.u],
            [model.unused],
            [other.u],
            [model.multiple],
            [model.scalar],
            [model.x],
            [model.u[0]],
            [None],
        ):
            with (
                self.subTest(types=[type(c).__name__ for c in controls]),
                self.assertRaises(ValueError),
            ):
                sim.compile(
                    parameters=[model.k],
                    times=[1],
                    controls=controls,
                    control_times=[0, 1.2],
                )
        for controls in (set(), frozenset(), {'u': model.u}, None):
            with self.subTest(type=type(controls)), self.assertRaises(TypeError):
                sim.compile(
                    parameters=[model.k],
                    times=[1],
                    controls=controls,
                    control_times=[0, 1.2],
                )
        with self.assertRaisesRegex(DAE_Error, 'all time-varying inputs'):
            sim.compile(parameters=[model.k], times=[1])
        model.v = pyo.Var(model.t)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.u[t] + m.v[t]
        )
        with self.assertRaisesRegex(DAE_Error, 'all time-varying inputs'):
            self.compile(model, [1], [0, 1.2], parameters=[])

    def test_invalid_boundaries(self):
        """Require finite distinct fixed boundaries covering the complete model span."""
        model = controlled_model()
        for boundaries in (
            None,
            [],
            [0],
            0.5,
            [[0, 1.2]],
            [0, 0.2, 0.2, 1.2],
            [0, 0.7, 0.2, 1.2],
            [0, float('nan'), 1.2],
            [0, float('inf'), 1.2],
            [0.1, 1.2],
            [0, 1.3],
        ):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                self.compile(model, [1], boundaries)
        plain = pyo.ConcreteModel()
        plain.t = ContinuousSet(bounds=(0, 1.2))
        plain.x = pyo.Var(plain.t)
        plain.dx = DerivativeVar(plain.x)
        plain.ode = pyo.Constraint(plain.t, rule=lambda m, t: m.dx[t] == 1)
        with self.assertRaisesRegex(ValueError, 'requires selected controls'):
            Simulator(plain, package='casadi').compile(
                parameters=[], times=[1], control_times=[0, 1.2]
            )

    def test_unused_control_symbol(self):
        """Do not expose a misleading zero after a control is simplified away."""
        model = controlled_model()
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.k * m.x[t] + 0 * m.u[t]
        )
        with self.assertRaisesRegex(ValueError, 'unused'):
            self.compile(model, [1], [0, 1.2])

    def test_model_options_and_simulation_preserved(self):
        """Compilation and derivative calls do not alter values or prior simulation."""
        model = controlled_model()
        model.x[0].fix(0.3)
        model.q[0].fix(0)
        model.inputs = pyo.Suffix()
        model.inputs[model.u] = {0: 0.1, 0.2: 0.4, 0.7: -0.2}
        sim = Simulator(model, package='casadi')
        prior_times, prior_profiles = sim.simulate(
            numpoints=9, integrator='cvodes', varying_inputs=model.inputs
        )
        before = [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        order = [id(c) for c in sim.get_variable_order('input')]
        options = {'reltol': 1e-11, 'abstol': 1e-13}
        original_options = deepcopy(options)
        times, boundaries = np.array([0, 0.2, 0.7, 1.2]), np.array([0, 0.2, 0.7, 1.2])
        flow = sim.compile(
            parameters=[model.k],
            controls=[model.u],
            times=times,
            control_times=boundaries,
            integrator_options=options,
        )
        times[:], boundaries[:] = -1, -2
        value = flow(x0=[0.3, 0], p=-0.7, u=ca.DM([[0.1, 0.4, -0.2]]))['xf']
        flow.forward(1)(
            x0=[0.3, 0],
            p=-0.7,
            u=ca.DM([[0.1, 0.4, -0.2]]),
            out_xf=value,
            fwd_x0=ca.DM.zeros(2, 1),
            fwd_p=1,
            fwd_u=ca.DM.zeros(1, 3),
        )
        flow.reverse(1)(
            x0=[0.3, 0],
            p=-0.7,
            u=ca.DM([[0.1, 0.4, -0.2]]),
            out_xf=value,
            adj_xf=ca.DM.ones(2, 4),
        )
        with self.assertRaises(ValueError):
            sim.compile(
                parameters=[model.k],
                controls=[model.u],
                times=[1],
                control_times=[0, 0.2, 0.2, 1.2],
            )
        with self.assertRaises(RuntimeError):
            sim.compile(
                parameters=[model.k],
                controls=[model.u],
                times=[1],
                control_times=[0, 1.2],
                integrator_options={'invalid_option': 1},
            )
        self.assertEqual(options, original_options)
        self.assertEqual(
            before, [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        )
        self.assertEqual(order, [id(c) for c in sim.get_variable_order('input')])
        np.testing.assert_array_equal(sim._tsim, prior_times)
        np.testing.assert_array_equal(sim._simsolution, prior_profiles)


if __name__ == '__main__':
    unittest.main()
