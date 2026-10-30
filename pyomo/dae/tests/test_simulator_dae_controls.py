# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Exact phase and switch checks of compiled index-one DAE controls."""

from copy import deepcopy
import math

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available, check_min_version
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import casadi as ca, casadi_available

BOUNDARIES = (0, 0.2, 0.7, 1.2)


def controlled_dae(kind='affine', control_kind='var'):
    """Build feedback controls in f and/or g without eliminating an algebraic state."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=0.3 if kind == 'nonlinear' else -0.7)
    model.a = pyo.Param(mutable=True, initialize=1.4)
    model.x, model.q, model.z = pyo.Var(model.t), pyo.Var(model.t), pyo.Var(model.t)
    model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)

    def control():
        return (
            pyo.Var(model.t)
            if control_kind == 'var'
            else pyo.Param(model.t, default=99, mutable=control_kind == 'mutable')
        )

    model.u = control()
    if kind == 'affine':
        model.v = control()
    model.ode = pyo.Constraint(
        model.t,
        rule=lambda m, t: m.dx[t] == m.k * m.z[t] + (m.v[t] if kind == 'affine' else 0),
    )
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    model.alg = pyo.Constraint(
        model.t,
        rule=lambda m, t: (m.z[t] ** 2 if kind == 'nonlinear' else m.z[t])
        == m.a * m.x[t] + m.u[t],
    )
    return model


def control_reference(kind, times, bounds, point, branch=1):
    """Propagate exact phase formulas and independent x0/q0/k/a/amplitude partials."""
    times, bounds, point = np.asarray(times), np.asarray(bounds), np.asarray(point)
    phases = len(bounds) - 1
    controls = point[4:].reshape(-1, phases)
    n = len(point)
    basis = np.eye(n)
    k, a = point[2:4]
    values = []
    jac = []
    for time in times:
        x, q = point[:2]
        dx, dq = basis[0].copy(), basis[1].copy()
        for j, (start, end) in enumerate(zip(bounds, bounds[1:])):
            if time <= start:
                break
            d = min(time, end) - start
            u = controls[0, j]
            du = basis[4 + j]
            if kind == 'nonlinear':
                z = branch * math.sqrt(a * x + u)
                dz = (a * dx + x * basis[3] + du) / (2 * z)
                q, dq = (
                    q + x * d + k * z * d * d / 2 + a * k * k * d**3 / 12,
                    dq
                    + d * dx
                    + (z * d * d / 2 + a * k * d**3 / 6) * basis[2]
                    + k * d * d / 2 * dz
                    + k * k * d**3 / 12 * basis[3],
                )
                x, dx = (
                    x + k * z * d + a * k * k * d * d / 4,
                    dx
                    + (z * d + a * k * d * d / 2) * basis[2]
                    + k * d * dz
                    + k * k * d * d / 4 * basis[3],
                )
            else:
                v = controls[1, j]
                dv = basis[4 + phases + j]
                rate = k * a
                dr = a * basis[2] + k * basis[3]
                forcing = k * u + v
                df = u * basis[2] + k * du + dv
                e = math.exp(rate * d)
                if rate:
                    g = math.expm1(rate * d) / rate
                    gp = ((rate * d - 1) * e + 1) / rate**2
                    h = (g - d) / rate
                    hp = (gp - h) / rate
                else:
                    g, gp, h, hp = d, d * d / 2, d * d / 2, d**3 / 6
                q, dq = (
                    q + g * x + forcing * h,
                    dq + g * dx + (gp * x + forcing * hp) * dr + h * df,
                )
                x, dx = (
                    e * x + forcing * g,
                    e * dx + (d * e * x + forcing * gp) * dr + g * df,
                )
        right = min(np.searchsorted(bounds, time, side='right') - 1, phases - 1)
        u = controls[0, right]
        du = basis[4 + right]
        if kind == 'nonlinear':
            z = branch * math.sqrt(a * x + u)
            dz = (a * dx + x * basis[3] + du) / (2 * z)
        else:
            z = a * x + u
            dz = a * dx + x * basis[3] + du
        values.append([x, q, z])
        jac.append([dx, dq, dz])
    return np.asarray(values).T, np.asarray(jac).transpose(1, 0, 2)


@unittest.skipUnless(
    numpy_available and casadi_available and check_min_version(ca, '3.8.1'),
    'NumPy and CasADi 3.8.1 or later are required',
)
class TestDAEControls(unittest.TestCase):
    """Check public amplitude and total sensitivities across algebraic jumps."""

    def compile(
        self,
        model,
        times=None,
        bounds=BOUNDARIES,
        controls=None,
        parameters=None,
        **kwds,
    ):
        """Use explicit scalar tolerances and the public control-selection API."""
        kwds.setdefault('integrator_options', {'reltol': 1e-10, 'abstol': 1e-12})
        if controls is None:
            controls = [model.u] + ([model.v] if hasattr(model, 'v') else [])
        return Simulator(model, package='casadi').compile(
            parameters=[model.k, model.a] if parameters is None else parameters,
            times=[0, 0.2, 0.5, 0.7, 1.2] if times is None else times,
            controls=controls,
            control_times=bounds,
            integrator='idas',
            **kwds,
        )

    def assert_close(self, actual, expected):
        """Check dimensions, finiteness and analytic agreement."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)

    def assert_products(self, flow, arguments, expected, jac):
        """Compare two independent native products with exact input partials and duality."""
        result = flow(**arguments)
        self.assert_close(ca.vertcat(result['xf'], result['zf']), expected)
        nx, np_ = flow.numel_in('x0'), flow.numel_in('p')
        rows, phases = flow.size_in('u')
        nt = expected.shape[1]
        n = nx + np_ + rows * phases
        self.assertEqual(jac.shape, (len(expected), nt, n))
        rng = np.random.default_rng(606)
        directions = rng.normal(size=(n, 2)) / 3
        seeds_u = np.hstack(
            [directions[nx + np_ :, i].reshape(rows, phases) for i in range(2)]
        )
        forward = flow.forward(2)(
            **arguments,
            out_xf=result['xf'],
            out_zf=result['zf'],
            fwd_x0=directions[:nx],
            fwd_p=directions[nx : nx + np_],
            fwd_z0=np.ones((flow.numel_in('z0'), 2)),
            fwd_u=seeds_u,
        )
        product = np.vstack(
            [np.asarray(forward['fwd_xf']), np.asarray(forward['fwd_zf'])]
        )
        self.assert_close(
            product,
            np.hstack(
                [np.einsum('itp,p->it', jac, directions[:, i]) for i in range(2)]
            ),
        )
        weights = np.hstack(
            [rng.normal(size=expected.shape) / 3, rng.normal(size=expected.shape) / 3]
        )
        reverse = flow.reverse(2)(
            **arguments,
            out_xf=result['xf'],
            out_zf=result['zf'],
            adj_xf=weights[:nx],
            adj_zf=weights[nx:],
        )
        gu = np.column_stack(
            [
                np.asarray(reverse['adj_u'])[:, phases * i : phases * (i + 1)].ravel()
                for i in range(2)
            ]
        )
        partial = np.vstack(
            [np.asarray(reverse['adj_x0']), np.asarray(reverse['adj_p']), gu]
        )
        self.assert_close(reverse['adj_z0'], np.zeros((flow.numel_in('z0'), 2)))
        for i in range(2):
            weight = weights[:, nt * i : nt * (i + 1)]
            self.assert_close(
                partial[:, i : i + 1], np.einsum('it,itp->p', weight, jac).reshape(n, 1)
            )
            for j in range(2):
                self.assertAlmostEqual(
                    float(np.sum(weight * product[:, nt * j : nt * (j + 1)])),
                    float(directions[:, j] @ partial[:, i]),
                    delta=3e-7,
                )

    def test_affine_feedback_and_integral(self):
        """Controls in f and g retain all parameter, IC and amplitude derivatives."""
        times = [0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.2]
        u = np.array([[0.1, 0.4, -0.2], [0.2, -0.1, 0.3]])
        for rate in (-0.7, 0):
            with self.subTest(rate=rate):
                model = controlled_dae()
                # Native IDAS needs explicit step control at zero feedback;
                # see examples/dae/sensitivity/dae_zero_rate_reproducer.py.
                options = {'reltol': 1e-10, 'abstol': 1e-12}
                if rate == 0:
                    options.update(
                        reltol=1e-11,
                        abstol=1e-13,
                        max_step_size=0.001,
                        max_num_steps=50000,
                    )
                flow = self.compile(model, times=times, integrator_options=options)
                self.assertEqual(flow.name_in(), ['x0', 'p', 'z0', 'u'])
                self.assertEqual(flow.name_out(), ['xf', 'zf'])
                self.assertEqual(flow.size_in('u'), (2, 3))
                expected, jac = control_reference(
                    'affine', times, BOUNDARIES, [1.3, 0, rate, 1.4, *u.ravel()]
                )
                self.assert_products(
                    flow,
                    {'x0': [1.3, 0], 'p': [rate, 1.4], 'z0': 0.3, 'u': u},
                    expected,
                    jac,
                )

    def test_right_side_and_terminal_algebraic_derivatives(self):
        """A new amplitude affects z at a switch even before it affects the dynamics."""
        model = controlled_dae()
        u = ca.DM([[0.1, 0.4, -0.2], [0, 0, 0]])
        for times, expected_u in (
            ([0], [1, 0, 0]),
            ([0.2], [0, 1, 0]),
            ([0.7], [0, 0, 1]),
            ([1.2], [0, 0, 1]),
        ):
            with self.subTest(times=times):
                flow = self.compile(model, times=times)
                reference, jac = control_reference(
                    'affine',
                    times,
                    BOUNDARIES,
                    [1.3, 0, 0, 1.4, *np.asarray(u).ravel()],
                )
                result = flow(x0=[1.3, 0], p=[0, 1.4], z0=1, u=u)
                self.assert_close(ca.vertcat(result['xf'], result['zf']), reference)
                adjoint = flow.factory(
                    'adjoint', ['x0', 'p', 'z0', 'u', 'adj:zf'], ['adj:u']
                )(x0=[1.3, 0], p=[0, 1.4], z0=1, u=u, adj_zf=1)['adj_u']
                self.assert_close(adjoint[0, :], np.asarray(expected_u).reshape(1, 3))
                direct = flow.factory(
                    'direct', ['x0', 'p', 'z0', 'u', 'fwd:u'], ['fwd:xf', 'fwd:zf']
                )(
                    x0=[1.3, 0],
                    p=[0, 1.4],
                    z0=1,
                    u=u,
                    fwd_u=ca.DM([[0, 1, 0], [0, 0, 0]]),
                )
                self.assert_close(direct['fwd_xf'], np.zeros((2, 1)))
                self.assert_close(direct['fwd_zf'], [[expected_u[1]]])
        flow = self.compile(model, times=[0.2, 0.7, 1.2])
        adjoint = flow.factory('adjoint', ['x0', 'p', 'z0', 'u', 'adj:zf'], ['adj:u'])(
            x0=[1.3, 0], p=[0, 1.4], z0=1, u=u, adj_zf=ca.DM([[1, 2, 3]])
        )['adj_u']
        self.assert_close(adjoint[0, :], [[0, 1, 5]])

    def test_sparse_dense_and_adjacent_observations(self):
        """Keep all exact output columns and propagate through unobserved control switches."""
        model = controlled_dae()
        u = np.array([[0.1, 0.4, -0.2], [0.2, -0.1, 0.3]])
        for times in (
            [0],
            [0.1],
            [1.2],
            [0.2, 0.7],
            np.linspace(0, 1.2, 51),
            [np.nextafter(0.2, 0), 0.2, np.nextafter(0.2, 1), 1.2],
        ):
            with self.subTest(times=times):
                flow = self.compile(model, times=times)
                expected, jac = control_reference(
                    'affine', times, BOUNDARIES, [1.3, 0, -0.7, 1.4, *u.ravel()]
                )
                self.assert_products(
                    flow,
                    {'x0': [1.3, 0], 'p': [-0.7, 1.4], 'z0': 0.3, 'u': u},
                    expected,
                    jac,
                )

    def test_nonlinear_branches_and_guesses(self):
        """Nonlinear consistent jumps retain the fixed branch and all control derivatives."""
        model = controlled_dae('nonlinear')
        times = [0, 0.2, 0.5, 0.7, 1.2]
        u = np.array([[0.1, 0.4, -0.2]])
        flow = self.compile(model, times=times)
        for guess in (0.2, 3, -0.2, -3):
            with self.subTest(guess=guess):
                expected, jac = control_reference(
                    'nonlinear',
                    times,
                    BOUNDARIES,
                    [1.3, 0, 0.3, 1.4, *u.ravel()],
                    1 if guess > 0 else -1,
                )
                self.assert_products(
                    flow,
                    {'x0': [1.3, 0], 'p': [0.3, 1.4], 'z0': guess, 'u': u},
                    expected,
                    jac,
                )

    def test_control_order_types_and_empty_parameters(self):
        """Selected Var/Param components remain symbolic; row ordering and empty p are stable."""
        times = [0, 0.2, 0.7, 1.2]
        u = np.array([[0.1, 0.4, -0.2], [0.2, -0.1, 0.3]])
        expected, jac = control_reference(
            'affine', times, BOUNDARIES, [1.3, 0, -0.7, 1.4, *u.ravel()]
        )
        for kind in ('var', 'mutable', 'immutable'):
            with self.subTest(kind=kind):
                model = controlled_dae(control_kind=kind)
                flow = self.compile(model, times=times, controls=[model.v, model.u])
                self.assert_products(
                    flow,
                    {'x0': [1.3, 0], 'p': [-0.7, 1.4], 'z0': 0.3, 'u': u[::-1]},
                    expected,
                    jac[:, :, [0, 1, 2, 3, 7, 8, 9, 4, 5, 6]],
                )
        model = controlled_dae()
        flow = self.compile(model, times=times, parameters=[])
        model.k.set_value(-0.2)
        model.a.set_value(2)
        self.assertEqual(flow.size_in('p'), (0, 1))
        self.assert_products(
            flow,
            {'x0': [1.3, 0], 'p': [], 'z0': 0.3, 'u': u},
            expected,
            jac[:, :, [0, 1, 4, 5, 6, 7, 8, 9]],
        )

    def test_total_initial_condition_and_shared_controls(self):
        """Compose nonlinear ICs, repeated amplitudes, an integral cost and explicit terms."""
        times = [0, 0.2, 0.7, 1.2]
        bounds = np.asarray(BOUNDARIES)
        for kind in ('affine', 'nonlinear'):
            with self.subTest(kind=kind):
                model = controlled_dae(kind)
                flow = self.compile(
                    model,
                    times=times,
                    integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
                )
                theta = ca.MX.sym('theta', 4)
                controls = ca.horzcat(theta[2], theta[3], theta[2] + theta[3])
                if kind == 'affine':
                    controls = ca.vertcat(
                        controls, ca.horzcat(theta[3], theta[2], theta[2] - theta[3])
                    )
                initial = ca.vertcat(
                    1 + theta[0] ** 2 + 0.1 * theta[2], 0.2 * theta[3] ** 2
                )
                out = flow(x0=initial, p=theta[:2], z0=0.3, u=controls)
                final = ca.vertcat(out['xf'][:, -1], out['zf'][:, -1])
                cost = (
                    ca.dot(ca.DM([2, 3, 4]), final)
                    + theta[0] ** 2
                    + 0.1 * theta[1] ** 2
                    + 0.2
                    * ca.dot(
                        ca.vec(
                            ca.repmat(ca.DM(np.diff(bounds)).T, controls.size1(), 1)
                        ),
                        ca.vec(controls**2),
                    )
                )
                total = ca.Function('total', [theta], [cost])
                point = np.array([-0.7 if kind == 'affine' else 0.3, 1.4, 0.1, 0.2])
                u = np.array(
                    [[point[2], point[3], point[2] + point[3]]]
                    + (
                        [[point[3], point[2], point[2] - point[3]]]
                        if kind == 'affine'
                        else []
                    )
                )
                physical = [
                    1 + point[0] ** 2 + 0.1 * point[2],
                    0.2 * point[3] ** 2,
                    *point[:2],
                    *u.ravel(),
                ]
                expected, jac = control_reference(kind, [1.2], bounds, physical)
                chain = np.zeros((len(physical), 4))
                chain[0] = [2 * point[0], 0, 0.1, 0]
                chain[1] = [0, 0, 0, 0.4 * point[3]]
                chain[2, 0] = 1
                chain[3, 1] = 1
                chain[4:7] = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]]
                if kind == 'affine':
                    chain[7:] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, -1]]
                partial = np.array([2, 3, 4]) @ jac[:, 0, :]
                gradient = (
                    partial @ chain
                    + np.array([2 * point[0], 0.2 * point[1], 0, 0])
                    + (0.4 * np.tile(np.diff(bounds), len(u)) * u.ravel()) @ chain[4:]
                )
                value = total(point)
                expected_cost = (
                    np.array([2, 3, 4]) @ expected[:, 0]
                    + point[0] ** 2
                    + 0.1 * point[1] ** 2
                    + 0.2 * np.sum(np.diff(bounds) * u**2)
                )
                self.assert_close(value, [[expected_cost]])
                self.assert_close(
                    total.forward(4)(point, value, np.eye(4)), gradient.reshape(1, 4)
                )
                self.assert_close(
                    total.reverse(1)(point, value, 1), gradient.reshape(4, 1)
                )
                errors = []
                for step in (1e-2, 1e-3, 1e-4, 1e-5):
                    difference = np.array(
                        [
                            float(
                                (total(point + step * d) - total(point - step * d))
                                / (2 * step)
                            )
                            for d in np.eye(4)
                        ]
                    )
                    errors.append(
                        float(
                            np.max(
                                abs(difference - gradient)
                                / np.maximum(1, abs(gradient))
                            )
                        )
                    )
                self.assertLess(min(errors), 1e-6)
                self.assertLess(min(errors[1:]), errors[0] / 10)

    def test_narrow_control_interval(self):
        """An interval between adjacent floats retains its finite-area input pulse."""
        model = controlled_dae()
        model.del_component(model.alg)
        model.del_component(model.u)
        model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] == m.a * m.x[t])
        bounds = [0, 0.2, np.nextafter(0.2, 1), 1.2]
        times = [0, 0.2, bounds[2], 0.7, 1.2]
        v = np.array([[0, 1e16, 0]])
        expected, jac = control_reference(
            'affine', times, bounds, [0.3, 0, 0, 1, 0, 0, 0, *v.ravel()]
        )
        flow = self.compile(model, times=times, bounds=bounds, controls=[model.v])
        result = flow(x0=[0.3, 0], p=[0, 1], z0=1, u=v)
        self.assert_close(ca.vertcat(result['xf'], result['zf']), expected)
        tangent = flow.factory(
            'direct', ['x0', 'p', 'z0', 'u', 'fwd:u'], ['fwd:xf', 'fwd:zf']
        )(x0=[0.3, 0], p=[0, 1], z0=1, u=v, fwd_u=ca.DM([[0, 1e16, 0]]))
        self.assert_close(
            ca.vertcat(tangent['fwd_xf'], tangent['fwd_zf']), jac[:, :, 8] * 1e16
        )
        adjoint = flow.factory('adjoint', ['x0', 'p', 'z0', 'u', 'adj:zf'], ['adj:u'])(
            x0=[0.3, 0], p=[0, 1], z0=1, u=v, adj_zf=ca.DM([[0, 0, 0, 0, 1]])
        )['adj_u']
        self.assertAlmostEqual(
            float(adjoint[0, 1]) * 1e16, float(jac[2, -1, 8]) * 1e16, delta=3e-7
        )

    def test_shifted_physical_time_and_controls(self):
        """The hidden clock and right-held amplitudes preserve time in f and g."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(2, 3.2))
        model.k = pyo.Param(mutable=True, initialize=-0.3)
        model.a = pyo.Param(mutable=True, initialize=1.4)
        model.x, model.q, model.z = pyo.Var(model.t), pyo.Var(model.t), pyo.Var(model.t)
        model.dx, model.dq = DerivativeVar(model.x), DerivativeVar(model.q)
        model.u, model.v = pyo.Var(model.t), pyo.Var(model.t)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.k * t + m.v[t]
        )
        model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
        model.alg = pyo.Constraint(
            model.t, rule=lambda m, t: m.z[t] == m.a * m.x[t] + m.u[t] + t
        )
        bounds = np.array([2, 2.2, 2.7, 3.2])
        times = np.array([2, 2.2, 2.5, 2.7, 3.2])
        delta = times - 2
        u = np.array([[0.1, 0.4, -0.2], [0.2, -0.1, 0.3]])
        lengths = np.maximum(0, np.minimum(times[:, None], bounds[1:]) - bounds[:-1])
        integrals = lengths * (times[:, None] - bounds[:-1]) - lengths**2 / 2
        tx = (times**2 - 4) / 2
        tq = (times**3 - 8) / 6 - 2 * delta
        x = 1.3 - 0.3 * tx + lengths @ u[1]
        q = 0.1 + 1.3 * delta - 0.3 * tq + integrals @ u[1]
        right = np.minimum(np.searchsorted(bounds, times, side='right') - 1, 2)
        expected = np.vstack([x, q, 1.4 * x + u[0, right] + times])
        jac = np.zeros((3, len(times), 10))
        jac[0, :, 0] = 1
        jac[1, :, 0] = delta
        jac[2, :, 0] = 1.4
        jac[1, :, 1] = 1
        jac[0, :, 2] = tx
        jac[1, :, 2] = tq
        jac[2, :, 2] = 1.4 * tx
        jac[2, :, 3] = x
        for j in range(3):
            jac[2, :, 4 + j] = right == j
        jac[0, :, 7:] = lengths
        jac[1, :, 7:] = integrals
        jac[2, :, 7:] = 1.4 * lengths
        flow = self.compile(model, times=times, bounds=bounds)
        self.assert_products(
            flow, {'x0': [1.3, 0.1], 'p': [-0.3, 1.4], 'z0': 1, 'u': u}, expected, jac
        )

    def test_failures_and_previous_simulation_preserved(self):
        """Failed switch consistency leaves the model/options/cache reusable and unchanged."""
        model = controlled_dae('nonlinear')
        model.x[0].fix(1.3)
        model.q[0].fix(0)
        model.z[0].set_value(1.4)
        model.inputs = pyo.Suffix()
        model.inputs[model.u] = {0: 0.1, 0.2: 0.4, 0.7: -0.2}
        sim = Simulator(model, package='casadi')
        old_times, old_values = sim.simulate(
            numpoints=9, integrator='idas', varying_inputs=model.inputs
        )
        old_inputs, old_algebraics = sim._siminputvars, sim._simalgvars
        values = [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        activation = [c.active for c in model.component_objects(pyo.Constraint)]
        options = {'reltol': 1e-10, 'abstol': 1e-12, 'linear_solver_options': {}}
        original = deepcopy(options)
        bounds = np.array(BOUNDARIES)
        times = np.array([0, 0.2, 0.7, 1.2])
        flow = sim.compile(
            parameters=[model.k, model.a],
            times=times,
            controls=[model.u],
            control_times=bounds,
            integrator='idas',
            integrator_options=options,
        )
        bounds[:] = -1
        times[:] = -2
        point = {
            'x0': [1.3, 0],
            'p': [0.3, 1.4],
            'z0': 0.3,
            'u': ca.DM([[0.1, 0.4, -0.2]]),
        }
        value = flow(**point)
        with self.assertRaises(RuntimeError):
            flow(**dict(point, u=ca.DM([[0.1, -10, -0.2]])))
        with self.assertRaises(RuntimeError):
            sim.compile(
                parameters=[model.k, model.a],
                times=[1.2],
                controls=[model.u],
                control_times=BOUNDARIES,
                integrator='idas',
                integrator_options={'invalid_option': 1},
            )
        repeat = flow(**point)
        self.assert_close(repeat['xf'], value['xf'])
        self.assert_close(repeat['zf'], value['zf'])
        flow.factory('direct', ['x0', 'p', 'z0', 'u', 'fwd:u'], ['fwd:xf', 'fwd:zf'])(
            **point, fwd_u=ca.DM([[0.1, 0.2, -0.3]])
        )
        flow.factory('adjoint', ['x0', 'p', 'z0', 'u', 'adj:xf', 'adj:zf'], ['adj:u'])(
            **point, adj_xf=np.ones((2, 4)), adj_zf=np.ones((1, 4))
        )
        self.assertEqual(options, original)
        self.assertEqual(
            values, [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        )
        self.assertEqual(
            activation, [c.active for c in model.component_objects(pyo.Constraint)]
        )
        self.assertIs(sim._siminputvars, old_inputs)
        self.assertIs(sim._simalgvars, old_algebraics)
        np.testing.assert_array_equal(sim._tsim, old_times)
        np.testing.assert_array_equal(sim._simsolution, old_values)

    def test_invalid_controls_boundaries_and_unused_symbols(self):
        """Require complete algebraic closure and structurally used independent controls."""
        model = controlled_dae()
        for controls in (
            [model.u],
            [model.u, model.u],
            [model.z],
            [controlled_dae().u, model.v],
        ):
            with (
                self.subTest(controls=[str(v) for v in controls]),
                self.assertRaises((ValueError, DAE_Error)),
            ):
                self.compile(model, controls=controls)
        for bounds in (None, [0, 0.2, 0.2, 1.2], [0, 0.7, 0.2, 1.2], [0, 1.3]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                self.compile(model, bounds=bounds)
        model = controlled_dae()
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.k * m.z[t] + 0 * m.v[t]
        )
        with self.assertRaisesRegex(ValueError, 'unused'):
            self.compile(model)


if __name__ == '__main__':
    unittest.main()
