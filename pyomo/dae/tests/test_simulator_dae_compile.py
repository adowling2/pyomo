# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Independent analytic checks of public index-one DAE compilation."""

from copy import deepcopy
import math

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available, check_min_version
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import casadi as ca, casadi_available


def dae_model(kind='linear'):
    """Build uneliminated linear, nonlinear or coupled index-one residuals."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=0.3 if kind == 'nonlinear' else 0.7)
    model.a = pyo.Param(mutable=True, initialize=1.4)
    model.x = pyo.Var(model.t)
    model.q = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.dq = DerivativeVar(model.q)
    model.z = pyo.Var(model.t)
    if kind == 'coupled':
        model.b = pyo.Param(mutable=True, initialize=0.2)
        model.y = pyo.Var(model.t)
    model.ode = pyo.Constraint(
        model.t,
        rule=lambda m, t: m.dx[t] == (m.k if kind == 'nonlinear' else -m.k) * m.z[t],
    )
    model.integral = pyo.Constraint(
        model.t, rule=lambda m, t: m.dq[t] == (m.y[t] if kind == 'coupled' else m.x[t])
    )
    model.alg = pyo.Constraint(
        model.t,
        rule=lambda m, t: (m.z[t] ** 2 if kind == 'nonlinear' else m.z[t])
        == m.a * m.x[t],
    )
    if kind == 'coupled':
        model.other_alg = pyo.Constraint(
            model.t, rule=lambda m, t: m.y[t] == m.z[t] ** 2 + m.b
        )
    return model


def dae_reference(kind, times, point, branch=1):
    """Evaluate closed-form states and hand-derived independent x0/q0/p partials."""
    times = np.asarray(times)
    x0, q0, k, a = point[:4]
    jac = np.zeros((4 if kind == 'coupled' else 3, len(times), len(point)))
    if kind == 'nonlinear':
        r = math.sqrt(a * x0)
        z = branch * r + a * k * times / 2
        x = z * z / a
        q = q0 + x0 * times + branch * r * k * times**2 / 2 + a * k * k * times**3 / 12
        zx, zk, za = (
            branch * a / (2 * r),
            a * times / 2,
            branch * x0 / (2 * r) + k * times / 2,
        )
        jac[0, :, 0], jac[0, :, 2], jac[0, :, 3] = (
            2 * z / a * zx,
            2 * z / a * zk,
            2 * z / a * za - z * z / a**2,
        )
        jac[1, :, 0], jac[1, :, 1] = times + branch * a * k * times**2 / (4 * r), 1
        jac[1, :, 2], jac[1, :, 3] = (
            branch * r * times**2 / 2 + a * k * times**3 / 6,
            branch * x0 * k * times**2 / (4 * r) + k * k * times**3 / 12,
        )
        jac[2, :, 0], jac[2, :, 2], jac[2, :, 3] = zx, zk, za
        return np.vstack([x, q, z]), jac
    rate = -k * a
    e = np.exp(rate * times)
    g = np.expm1(rate * times) / rate
    gp = ((rate * times - 1) * e + 1) / rate**2
    x = x0 * e
    jac[0, :, 0], jac[0, :, 2], jac[0, :, 3] = e, -a * times * x, -k * times * x
    jac[2, :, 0], jac[2, :, 2], jac[2, :, 3] = (
        a * e,
        -a * a * times * x,
        x * (1 - k * a * times),
    )
    if kind == 'linear':
        q = q0 + x0 * g
        jac[1, :, 0], jac[1, :, 1], jac[1, :, 2], jac[1, :, 3] = (
            g,
            1,
            -a * x0 * gp,
            -k * x0 * gp,
        )
        return np.vstack([x, q, a * x]), jac
    b = point[4]
    e2 = np.exp(2 * rate * times)
    g2 = np.expm1(2 * rate * times) / (2 * rate)
    gp2 = ((2 * rate * times - 1) * e2 + 1) / (2 * rate) ** 2
    q = q0 + a * a * x0 * x0 * g2 + b * times
    jac[1, :, 0], jac[1, :, 1] = 2 * a * a * x0 * g2, 1
    jac[1, :, 2], jac[1, :, 3], jac[1, :, 4] = (
        -2 * a**3 * x0 * x0 * gp2,
        2 * a * x0 * x0 * g2 - 2 * k * a * a * x0 * x0 * gp2,
        times,
    )
    jac[3] = 2 * a * x[:, None] * jac[2]
    jac[3, :, 4] += 1
    return np.vstack([x, q, a * x, a * a * x * x + b]), jac


@unittest.skipUnless(
    numpy_available and casadi_available and check_min_version(ca, '3.8.1'),
    'NumPy and CasADi 3.8.1 or later are required',
)
class TestDAECompile(unittest.TestCase):
    """Exercise public DAE maps without an optimizer, ASL or explicit elimination."""

    def compile(self, model, times=None, parameters=None, **kwds):
        """Compile with the independently qualified scalar tolerances."""
        kwds.setdefault('integrator_options', {'reltol': 1e-10, 'abstol': 1e-12})
        return Simulator(model, package='casadi').compile(
            parameters=[model.k, model.a] if parameters is None else parameters,
            times=[0, 0.2, 0.7, 1.2] if times is None else times,
            integrator='idas',
            **kwds,
        )

    def assert_close(self, actual, expected):
        """Require finite values and explicit dimensions before numerical agreement."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)

    def test_analytic_outputs_directions_and_duality(self):
        """Check every x/z observation, mixed weights and multiple directions."""
        for kind in ('linear', 'nonlinear', 'coupled'):
            with self.subTest(kind=kind):
                model = dae_model(kind)
                parameters = [model.k, model.a] + (
                    [model.b] if kind == 'coupled' else []
                )
                point = np.array(
                    [1.3, 0, 0.3 if kind == 'nonlinear' else 0.7, 1.4]
                    + ([0.2] if kind == 'coupled' else [])
                )
                guesses = [0.3, 0.5] if kind == 'coupled' else [0.3]
                flow = self.compile(model, parameters=parameters)
                self.assertEqual(flow.name_in(), ['x0', 'p', 'z0'])
                self.assertEqual(flow.name_out(), ['xf', 'zf'])
                expected, jac = dae_reference(kind, [0, 0.2, 0.7, 1.2], point)
                result = flow(x0=point[:2], p=point[2:], z0=guesses)
                self.assert_close(ca.vertcat(result['xf'], result['zf']), expected)
                n = len(point)
                seeds = np.column_stack(
                    [np.arange(1, n + 1) / 10, np.arange(n, 0, -1) / 10]
                )
                forward = flow.forward(2)(
                    x0=point[:2],
                    p=point[2:],
                    z0=guesses,
                    out_xf=result['xf'],
                    out_zf=result['zf'],
                    fwd_x0=seeds[:2],
                    fwd_p=seeds[2:],
                    fwd_z0=np.ones((len(guesses), 2)),
                )
                direct = np.vstack(
                    [np.asarray(forward['fwd_xf']), np.asarray(forward['fwd_zf'])]
                )
                self.assert_close(
                    direct,
                    np.hstack(
                        [np.einsum('itp,p->it', jac, seeds[:, i]) for i in range(2)]
                    ),
                )
                rng = np.random.default_rng(604)
                weights = rng.normal(size=(len(expected), 8)) / 3
                reverse = flow.reverse(2)(
                    x0=point[:2],
                    p=point[2:],
                    z0=guesses,
                    out_xf=result['xf'],
                    out_zf=result['zf'],
                    adj_xf=weights[:2],
                    adj_zf=weights[2:],
                )
                partial = np.vstack(
                    [np.asarray(reverse['adj_x0']), np.asarray(reverse['adj_p'])]
                )
                self.assert_close(reverse['adj_z0'], np.zeros((len(guesses), 2)))
                for i in range(2):
                    weight = weights[:, 4 * i : 4 * (i + 1)]
                    self.assert_close(
                        partial[:, i : i + 1],
                        np.einsum('it,itp->p', weight, jac).reshape(n, 1),
                    )
                    for j in range(2):
                        self.assertAlmostEqual(
                            float(np.sum(weight * direct[:, 4 * j : 4 * (j + 1)])),
                            float(seeds[:, j] @ partial[:, i]),
                            delta=3e-7,
                        )

    def test_each_terminal_cost(self):
        """Algebraic-only and differential/integral-only adjoints all retain direct terms."""
        model = dae_model('coupled')
        flow = self.compile(model, parameters=[model.k, model.a, model.b])
        point = np.array([1.3, 0, 0.7, 1.4, 0.2])
        expected, jac = dae_reference('coupled', [0, 0.2, 0.7, 1.2], point)
        adjoint = flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:x0', 'adj:p']
        )
        for row in range(4):
            with self.subTest(output=row):
                weights = np.zeros((4, 4))
                weights[row, -1] = 1
                partial = adjoint(
                    x0=point[:2],
                    p=point[2:],
                    z0=[1, 1],
                    adj_xf=weights[:2],
                    adj_zf=weights[2:],
                )
                self.assert_close(
                    ca.vertcat(partial['adj_x0'], partial['adj_p']),
                    jac[row, -1, :].reshape(5, 1),
                )

    def test_total_initial_condition_and_cost_derivatives(self):
        """Differentiate nonlinear ICs, integral/algebraic costs and explicit parameters."""
        for kind in ('linear', 'nonlinear', 'coupled'):
            with self.subTest(kind=kind):
                model = dae_model(kind)
                parameters = [model.k, model.a] + (
                    [model.b] if kind == 'coupled' else []
                )
                flow = self.compile(model, parameters=parameters)
                n = len(parameters) + 1
                theta = ca.MX.sym('theta', n)
                initial = ca.vertcat(theta[0] ** 2, theta[1] ** 2)
                guess = [0.3, 0.5] if kind == 'coupled' else [0.3]
                result = flow(x0=initial, p=ca.vertcat(theta[0], theta[2:]), z0=guess)
                states = ca.vertcat(result['xf'], result['zf'])
                weights = np.arange(2, 2 + states.size1())
                cost = (
                    ca.dot(ca.DM(weights), states[:, -1])
                    + theta[0] ** 2
                    + 0.2 * theta[2] ** 2
                )
                total = ca.Function('total', [theta], [cost])
                point = np.array(
                    [0.3 if kind == 'nonlinear' else 0.7, 0.2, 1.4]
                    + ([0.2] if kind == 'coupled' else [])
                )
                values, jac = dae_reference(
                    kind, [1.2], [point[0] ** 2, point[1] ** 2, point[0], *point[2:]]
                )
                partial = weights @ jac[:, 0, :]
                expected = np.array(
                    [
                        2 * point[0] * partial[0] + partial[2] + 2 * point[0],
                        2 * point[1] * partial[1],
                        partial[3] + 0.4 * point[2],
                        *partial[4:],
                    ]
                )
                value = total(point)
                self.assert_close(
                    value,
                    [[weights @ values[:, 0] + point[0] ** 2 + 0.2 * point[2] ** 2]],
                )
                self.assert_close(
                    total.forward(n)(point, value, np.eye(n)), expected.reshape(1, n)
                )
                self.assert_close(
                    total.reverse(1)(point, value, 1), expected.reshape(n, 1)
                )
                errors = []
                for step in (1e-2, 1e-3, 1e-4, 1e-5):
                    fd = np.array(
                        [
                            float(
                                (total(point + step * d) - total(point - step * d))
                                / (2 * step)
                            )
                            for d in np.eye(n)
                        ]
                    )
                    errors.append(
                        float(np.max(abs(fd - expected) / np.maximum(1, abs(expected))))
                    )
                self.assertLess(min(errors), 1e-6)
                self.assertLess(min(errors[1:]), errors[0] / 10)

    def test_observation_grids_and_initial_only(self):
        """Initial-only, sparse and dense columns retain the consistency chain rule."""
        model = dae_model('nonlinear')
        for times in ([0], [0.1], [1.2], [0, 0.2, 0.7], np.linspace(0, 1.2, 21)):
            with self.subTest(times=times):
                flow = self.compile(model, times=times)
                expected, jac = dae_reference('nonlinear', times, [1.3, 0, 0.3, 1.4])
                result = flow(x0=[1.3, 0], p=[0.3, 1.4], z0=[1])
                self.assert_close(ca.vertcat(result['xf'], result['zf']), expected)
                forward = flow.factory(
                    'direct', ['x0', 'p', 'z0', 'fwd:x0', 'fwd:p'], ['fwd:xf', 'fwd:zf']
                )(
                    x0=[1.3, 0],
                    p=[0.3, 1.4],
                    z0=[1],
                    fwd_x0=[0.1, 0.2],
                    fwd_p=[0.3, 0.4],
                )
                self.assert_close(
                    ca.vertcat(forward['fwd_xf'], forward['fwd_zf']),
                    np.einsum('itp,p->it', jac, [0.1, 0.2, 0.3, 0.4]),
                )
                reverse = flow.factory(
                    'adjoint',
                    ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'],
                    ['adj:x0', 'adj:p'],
                )(
                    x0=[1.3, 0],
                    p=[0.3, 1.4],
                    z0=[1],
                    adj_xf=np.ones((2, len(times))),
                    adj_zf=np.ones((1, len(times))),
                )
                self.assert_close(
                    ca.vertcat(reverse['adj_x0'], reverse['adj_p']),
                    np.sum(jac, axis=(0, 1)).reshape(4, 1),
                )

    def test_guesses_and_both_branches(self):
        """Guess changes within a branch preserve x0, values and physical derivatives."""
        flow = self.compile(dae_model('nonlinear'))
        for guess in (0.2, 3, -0.2, -3):
            with self.subTest(guess=guess):
                branch = 1 if guess > 0 else -1
                expected, jac = dae_reference(
                    'nonlinear', [0, 0.2, 0.7, 1.2], [1.3, 0, 0.3, 1.4], branch
                )
                value = flow(x0=[1.3, 0], p=[0.3, 1.4], z0=guess)
                self.assert_close(ca.vertcat(value['xf'], value['zf']), expected)
                self.assertEqual(float(value['xf'][0, 0]), 1.3)
                adj = flow.factory(
                    'adjoint',
                    ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'],
                    ['adj:x0', 'adj:p', 'adj:z0'],
                )(
                    x0=[1.3, 0],
                    p=[0.3, 1.4],
                    z0=guess,
                    adj_xf=np.ones((2, 4)),
                    adj_zf=np.ones((1, 4)),
                )
                self.assert_close(
                    ca.vertcat(adj['adj_x0'], adj['adj_p']),
                    np.sum(jac, axis=(0, 1)).reshape(4, 1),
                )
                self.assert_close(adj['adj_z0'], [[0]])

    def test_empty_parameters_and_snapshots(self):
        """Unselected scalar values are captured, and the empty parameter input stays valid."""
        model = dae_model()
        flow = self.compile(model, parameters=[])
        model.k.set_value(0.2)
        model.a.set_value(2)
        expected, jac = dae_reference('linear', [0, 0.2, 0.7, 1.2], [1.3, 0, 0.7, 1.4])
        result = flow(x0=[1.3, 0], p=[], z0=[1])
        self.assertEqual(flow.size_in('p'), (0, 1))
        self.assert_close(ca.vertcat(result['xf'], result['zf']), expected)
        partial = flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:x0', 'adj:p']
        )(x0=[1.3, 0], p=[], z0=1, adj_xf=np.ones((2, 4)), adj_zf=np.ones((1, 4)))
        self.assertEqual(partial['adj_p'].numel(), 0)
        self.assert_close(
            partial['adj_x0'], np.sum(jac[:, :, :2], axis=(0, 1)).reshape(2, 1)
        )

    def test_parameter_only_in_algebraic_nonlinear_expressions(self):
        """Selected mutable leaves in g remain symbolic, including stationary points."""
        for stationary in (False, True):
            with self.subTest(stationary=stationary):
                model = dae_model()
                model.del_component(model.alg)
                model.alg = pyo.Constraint(
                    model.t,
                    rule=lambda m, t: m.z[t]
                    == (1 + m.a**2 if stationary else pyo.exp(m.a)) * m.x[t],
                )
                a = 0 if stationary else math.log(1.4)
                gain = 1 + a * a if stationary else math.exp(a)
                factor = 2 * a if stationary else gain
                flow = self.compile(model)
                expected, jac = dae_reference(
                    'linear', [0, 0.2, 0.7, 1.2], [1.3, 0, 0.7, gain]
                )
                value = flow(x0=[1.3, 0], p=[0.7, a], z0=1)
                self.assert_close(ca.vertcat(value['xf'], value['zf']), expected)
                direct = flow.factory(
                    'direct', ['x0', 'p', 'z0', 'fwd:p'], ['fwd:xf', 'fwd:zf']
                )(x0=[1.3, 0], p=[0.7, a], z0=1, fwd_p=[0, 1])
                self.assert_close(
                    ca.vertcat(direct['fwd_xf'], direct['fwd_zf']),
                    jac[:, :, 3] * factor,
                )
                reverse = flow.factory(
                    'adjoint', ['x0', 'p', 'z0', 'adj:zf'], ['adj:p']
                )(x0=[1.3, 0], p=[0.7, a], z0=1, adj_zf=ca.DM([[0, 0, 0, 1]]))
                self.assert_close(
                    reverse['adj_p'], [[jac[2, -1, 2]], [jac[2, -1, 3] * factor]]
                )

    def test_uninitialized_selected_parameters(self):
        """Compilation reads neither selected scalar values nor independent state ICs."""
        model = dae_model()
        model.k.set_value(None)
        model.a.set_value(None)
        flow = self.compile(model)
        result = flow(x0=[1.3, 0], p=[0.7, 1.4], z0=1)
        expected, _ = dae_reference('linear', [0, 0.2, 0.7, 1.2], [1.3, 0, 0.7, 1.4])
        self.assert_close(ca.vertcat(result['xf'], result['zf']), expected)
        self.assertIsNone(model.k.value)
        self.assertIsNone(model.x[0].value)
        self.assertIsNone(model.z[0].value)

    def test_indexed_algebraic_and_parameter_order(self):
        """Selected indexed scalar entries and time-indexed algebraics retain ordering."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1.2))
        model.k = pyo.Param(mutable=True, initialize=0.7)
        model.a = pyo.Param(['a', 'b'], mutable=True, initialize={'a': 0.8, 'b': 1.4})
        model.x, model.q = pyo.Var(model.t), pyo.Var(model.t)
        model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)
        model.z = pyo.Var(model.t, ['a', 'b'])
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == -m.k * m.z[t, 'b']
        )
        model.integral = pyo.Constraint(
            model.t, rule=lambda m, t: m.dq[t] == m.z[t, 'a']
        )
        model.alg = pyo.Constraint(
            model.t, ['a', 'b'], rule=lambda m, t, i: m.z[t, i] == m.a[i] * m.x[t]
        )
        sim = Simulator(model, package='casadi')
        order = sim.get_variable_order('time-varying')
        self.assertEqual([str(v) for v in order], ['z[{t},b]', 'z[{t},a]'])
        flow = sim.compile(
            parameters=[model.a['a'], model.k, model.a['b']],
            times=[0, 1.2],
            integrator='idas',
            integrator_options={'reltol': 1e-10, 'abstol': 1e-12},
        )
        times = np.array([0, 1.2])
        aa, k, ab, x0 = 0.8, 0.7, 1.4, 1.3
        rate = -k * ab
        e = np.exp(rate * times)
        g = np.expm1(rate * times) / rate
        gp = ((rate * times - 1) * e + 1) / rate**2
        x = x0 * e
        q = 0.2 + aa * x0 * g
        result = flow(x0=[x0, 0.2], p=[aa, k, ab], z0=[1, 1])
        self.assert_close(result['xf'], np.vstack([x, q]))
        self.assert_close(result['zf'], np.vstack([ab * x, aa * x]))
        c = 2 + 4 * ab + 5 * aa
        expected = np.array(
            [
                5 * x[-1] + 3 * x0 * g[-1],
                c * (-ab * times[-1] * x[-1]) - 3 * ab * aa * x0 * gp[-1],
                4 * x[-1] + c * (-k * times[-1] * x[-1]) - 3 * k * aa * x0 * gp[-1],
            ]
        )
        partial = flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:p', 'adj:x0']
        )(
            x0=[x0, 0.2],
            p=[aa, k, ab],
            z0=[1, 1],
            adj_xf=ca.DM([[0, 2], [0, 3]]),
            adj_zf=ca.DM([[0, 4], [0, 5]]),
        )
        self.assert_close(partial['adj_p'], expected.reshape(3, 1))
        self.assert_close(partial['adj_x0'], [[c * e[-1] + 3 * aa * g[-1]], [3]])

    def test_shifted_non_autonomous_equations(self):
        """Physical time appears in both differential and algebraic phase equations."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(2, 3.2))
        model.k = pyo.Param(mutable=True, initialize=-0.3)
        model.a = pyo.Param(mutable=True, initialize=1.4)
        model.x, model.q, model.z = pyo.Var(model.t), pyo.Var(model.t), pyo.Var(model.t)
        model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.k * t)
        model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
        model.alg = pyo.Constraint(
            model.t, rule=lambda m, t: m.z[t] == m.a * m.x[t] + t
        )
        times = np.array([2, 2.2, 2.7, 3.2])
        delta = times - 2
        dx = (times**2 - 4) / 2
        dq = (times**3 - 8) / 6 - 2 * delta
        x = 1.3 - 0.3 * dx
        q = 0.1 + 1.3 * delta - 0.3 * dq
        z = 1.4 * x + times
        flow = self.compile(model, times=times)
        result = flow(x0=[1.3, 0.1], p=[-0.3, 1.4], z0=1)
        self.assert_close(result['xf'], np.vstack([x, q]))
        self.assert_close(result['zf'], z.reshape(1, 4))
        direct = flow.factory(
            'direct', ['x0', 'p', 'z0', 'fwd:p'], ['fwd:xf', 'fwd:zf']
        )(x0=[1.3, 0.1], p=[-0.3, 1.4], z0=1, fwd_p=[0.2, 0.3])
        self.assert_close(direct['fwd_xf'], np.vstack([0.2 * dx, 0.2 * dq]))
        self.assert_close(direct['fwd_zf'], (1.4 * 0.2 * dx + 0.3 * x).reshape(1, 4))
        adjoint = flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:x0', 'adj:p']
        )(
            x0=[1.3, 0.1],
            p=[-0.3, 1.4],
            z0=1,
            adj_xf=np.ones((2, 4)),
            adj_zf=np.ones((1, 4)),
        )
        self.assert_close(adjoint['adj_p'], [[np.sum(2.4 * dx + dq)], [np.sum(x)]])
        self.assert_close(adjoint['adj_x0'], [[np.sum(2.4 + delta)], [4]])

    def test_tolerance_refinement(self):
        """Independent algebraic-terminal direct/adjoint errors converge with tolerance."""
        model = dae_model('nonlinear')
        expected, jac = dae_reference(
            'nonlinear', [0, 0.2, 0.7, 1.2], [1.3, 0, 0.3, 1.4]
        )
        errors = []
        for tolerance in (1e-5, 1e-8, 1e-11):
            flow = self.compile(
                model,
                integrator_options={'reltol': tolerance, 'abstol': tolerance / 100},
            )
            values = flow(x0=[1.3, 0], p=[0.3, 1.4], z0=0.3)
            direct = flow.factory('direct', ['x0', 'p', 'z0', 'fwd:p'], ['fwd:zf'])(
                x0=[1.3, 0], p=[0.3, 1.4], z0=0.3, fwd_p=[1, 0]
            )
            adjoint = flow.factory('adjoint', ['x0', 'p', 'z0', 'adj:zf'], ['adj:p'])(
                x0=[1.3, 0], p=[0.3, 1.4], z0=0.3, adj_zf=ca.DM([[0, 0, 0, 1]])
            )
            errors.append(
                max(
                    float(np.max(abs(np.asarray(values['zf']) - expected[2:]))),
                    float(np.max(abs(np.asarray(direct['fwd_zf']) - jac[2:, :, 2]))),
                    float(
                        np.max(
                            abs(np.asarray(adjoint['adj_p']).ravel() - jac[2, -1, 2:])
                        )
                    ),
                )
            )
        self.assertLess(errors[-1], 3e-7)
        self.assertLess(errors[-1], errors[0] / 10)

    def test_time_only_in_algebraic_residual(self):
        """The internal clock also handles time dependence entering only through g."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(2, 3.2))
        model.k = pyo.Param(mutable=True, initialize=0.3)
        model.a = pyo.Param(mutable=True, initialize=1.4)
        model.x, model.q, model.z = pyo.Var(model.t), pyo.Var(model.t), pyo.Var(model.t)
        model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.k * m.z[t])
        model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
        model.alg = pyo.Constraint(
            model.t, rule=lambda m, t: m.z[t] == m.a * m.x[t] + t
        )
        times = np.array([2, 2.2, 2.7, 3.2])
        delta = times - 2
        rate, k, a = -0.42, 0.3, 1.4
        e = np.exp(rate * delta)
        g = np.expm1(rate * delta) / rate
        gp = ((rate * delta - 1) * e + 1) / rate**2
        h = (g - delta) / rate
        hp = (gp - h) / rate
        integral = (h - delta**2 / 2) / rate
        integral_p = (hp - integral) / rate
        x = 1.3 * e - k * (2 * g + h)
        q = 0.1 + 1.3 * g - k * (2 * h + integral)
        xr = 1.3 * delta * e - k * (2 * gp + hp)
        qr = 1.3 * gp - k * (2 * hp + integral_p)
        xk = -a * xr - (2 * g + h)
        xa = -k * xr
        qk = -a * qr - (2 * h + integral)
        qa = -k * qr
        zk = a * xk
        za = x + a * xa
        flow = self.compile(model, times=times)
        result = flow(x0=[1.3, 0.1], p=[k, a], z0=1)
        self.assert_close(result['xf'], np.vstack([x, q]))
        self.assert_close(result['zf'], (a * x + times).reshape(1, 4))
        tangent = flow.factory(
            'direct', ['x0', 'p', 'z0', 'fwd:p'], ['fwd:xf', 'fwd:zf']
        )(x0=[1.3, 0.1], p=[k, a], z0=1, fwd_p=[0.2, 0.3])
        self.assert_close(
            tangent['fwd_xf'], np.vstack([0.2 * xk + 0.3 * xa, 0.2 * qk + 0.3 * qa])
        )
        self.assert_close(tangent['fwd_zf'], (0.2 * zk + 0.3 * za).reshape(1, 4))
        reverse = flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:p']
        )(x0=[1.3, 0.1], p=[k, a], z0=1, adj_xf=np.ones((2, 4)), adj_zf=np.ones((1, 4)))
        self.assert_close(
            reverse['adj_p'], [[np.sum(xk + qk + zk)], [np.sum(xa + qa + za)]]
        )

    def test_dense_algebraic_jacobian(self):
        """A dense algebraic block keeps scalar-parameter direct/adjoint dependence."""
        n = 10
        matrix = 2 * np.eye(n) + 0.02 * np.ones((n, n))
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1.2))
        model.k = pyo.Param(mutable=True, initialize=0.7)
        model.a = pyo.Param(mutable=True, initialize=1.4)
        model.x, model.q = pyo.Var(model.t), pyo.Var(model.t)
        model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)
        model.z = pyo.Var(model.t, range(n))
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == -m.k * m.z[t, 0]
        )
        model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
        model.alg = pyo.Constraint(
            model.t,
            range(n),
            rule=lambda m, t, i: sum(
                (matrix[i, j] + (m.a if i == j else 0)) * m.z[t, j] for j in range(n)
            )
            == m.x[t],
        )
        times = np.array([0, 0.2, 0.7, 1.2])
        k, a, x0 = 0.7, 1.4, 1.3
        system = matrix + a * np.eye(n)
        c = np.linalg.solve(system, np.ones(n))
        dc = -np.linalg.solve(system, c)
        rate = -k * c[0]
        e = np.exp(rate * times)
        g = np.expm1(rate * times) / rate
        gp = ((rate * times - 1) * e + 1) / rate**2
        x = x0 * e
        q = x0 * g
        xk = -c[0] * times * x
        xa = -k * dc[0] * times * x
        qk = -c[0] * x0 * gp
        qa = -k * dc[0] * x0 * gp
        zk = c[:, None] * xk
        za = dc[:, None] * x + c[:, None] * xa
        flow = self.compile(model, times=times)
        values = flow(x0=[x0, 0], p=[k, a], z0=np.zeros(n))
        self.assert_close(values['xf'], np.vstack([x, q]))
        self.assert_close(values['zf'], c[:, None] * x)
        tangent = flow.factory(
            'direct', ['x0', 'p', 'z0', 'fwd:p'], ['fwd:xf', 'fwd:zf']
        )(x0=[x0, 0], p=[k, a], z0=np.zeros(n), fwd_p=[0, 1])
        self.assert_close(tangent['fwd_xf'], np.vstack([xa, qa]))
        self.assert_close(tangent['fwd_zf'], za)
        weights = np.linspace(-0.2, 0.5, n * len(times)).reshape(n, len(times))
        adjoint = flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:p']
        )(
            x0=[x0, 0],
            p=[k, a],
            z0=np.zeros(n),
            adj_xf=np.ones((2, len(times))),
            adj_zf=weights,
        )
        self.assert_close(
            adjoint['adj_p'],
            [
                [np.sum(xk + qk) + np.sum(weights * zk)],
                [np.sum(xa + qa) + np.sum(weights * za)],
            ],
        )

    def test_invalid_and_singular_systems(self):
        """Fail clearly on missing closure, high-index structure and invalid roots."""
        model = dae_model()
        model.extra = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] == 2 * m.x[t])
        with self.assertRaisesRegex(DAE_Error, 'square'):
            self.compile(model)
        model = dae_model()
        model.del_component(model.alg)
        model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.x[t] == 0)
        with self.assertRaisesRegex(RuntimeError, 'rank-deficient'):
            self.compile(model, parameters=[model.k])
        model = dae_model('nonlinear')
        flow = self.compile(model)
        for initial, parameters, guess in (
            ([0, 0], [0.3, 1.4], 0),
            ([1.3, 0], [0.3, 0], 0),
            ([-1, 0], [0.3, 1.4], 1),
            ([1.3, 0], [0.3, -1], 1),
            ([1.3, 0], [0.3, 1.4], 0),
            ([1.3, 0], [0.3, 1.4], float('nan')),
        ):
            with (
                self.subTest(initial=initial, parameters=parameters, guess=guess),
                self.assertRaises(RuntimeError),
            ):
                flow(x0=initial, p=parameters, z0=guess)
        initial_only = self.compile(model, times=[0])
        with self.assertRaisesRegex(RuntimeError, 'locally regular'):
            initial_only(x0=[0, 0], p=[0.3, 1.4], z0=0)
        model = dae_model()
        model.del_component(model.ode)
        model.del_component(model.integral)
        with self.assertRaisesRegex(DAE_Error, 'at least one differential equation'):
            self.compile(model)

    def test_invalid_options_and_default_absolute_tolerance(self):
        """Require consistent forward/backward initialization and algebraic error control."""
        model = dae_model()
        for options in (
            {'calc_ic': False},
            {'calc_icB': False},
            {'suppress_algebraic': True},
            {'quad_err_con': False},
            {'abstolv': [1e-12] * 3},
            {'abstol': 0},
            {'augmented_options': {}},
            {'nfwd': 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.compile(model, integrator_options=options)
        result = self.compile(model, times=[0], integrator_options={})(
            x0=[1.3, 0], p=[0.7, 1.4], z0=1
        )
        self.assert_close(result['zf'], [[1.82]])
        model = dae_model('coupled')
        model.unused = pyo.Param(mutable=True, initialize=0)
        with self.assertRaisesRegex(ValueError, 'unused'):
            self.compile(model, parameters=[model.unused])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.compile(model, parameters=[model.a, model.a])
        model = dae_model()
        model.u = pyo.Var(model.t)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == -m.k * m.z[t] + m.u[t]
        )
        with self.assertRaisesRegex(DAE_Error, 'DAE controls'):
            self.compile(model, controls=[model.u], control_times=[0, 1.2])

    def test_model_options_and_previous_simulation_preserved(self):
        """Successful and failed graph/evaluation calls preserve all caller-owned state."""
        model = dae_model('nonlinear')
        model.x[0].fix(1.3)
        model.q[0].fix(0)
        model.z[0].set_value(1.4)
        simulator = Simulator(model, package='casadi')
        old_times, old_profile = simulator.simulate(numpoints=5, integrator='idas')
        old_inputs, old_algebraics = simulator._siminputvars, simulator._simalgvars
        values = [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        activation = [c.active for c in model.component_objects(pyo.Constraint)]
        options = {'reltol': 1e-10, 'abstol': 1e-12, 'linear_solver_options': {}}
        original = deepcopy(options)
        times = np.array([0, 0.2, 0.7, 1.2])
        flow = simulator.compile(
            parameters=[model.k, model.a],
            times=times,
            integrator='idas',
            integrator_options=options,
        )
        times[:] = -1
        point = {'x0': [1.3, 0], 'p': [0.3, 1.4], 'z0': 1}
        result = flow(**point)
        with self.assertRaises(RuntimeError):
            flow(**dict(point, z0=0))
        repeat = flow(**point)
        self.assert_close(repeat['xf'], result['xf'])
        self.assert_close(repeat['zf'], result['zf'])
        flow.factory('direct', ['x0', 'p', 'z0', 'fwd:p'], ['fwd:xf', 'fwd:zf'])(
            **point, fwd_p=[1, 0]
        )
        flow.factory(
            'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'], ['adj:x0', 'adj:p']
        )(**point, adj_xf=np.ones((2, 4)), adj_zf=np.ones((1, 4)))
        with self.assertRaises(RuntimeError):
            simulator.compile(
                parameters=[model.k, model.a],
                times=[1],
                integrator='idas',
                integrator_options={'invalid_option': 1},
            )
        self.assertEqual(options, original)
        self.assertEqual(
            values, [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        )
        self.assertEqual(
            activation, [c.active for c in model.component_objects(pyo.Constraint)]
        )
        self.assertIs(simulator._siminputvars, old_inputs)
        self.assertIs(simulator._simalgvars, old_algebraics)
        np.testing.assert_array_equal(simulator._tsim, old_times)
        np.testing.assert_array_equal(simulator._simsolution, old_profile)


if __name__ == '__main__':
    unittest.main()
