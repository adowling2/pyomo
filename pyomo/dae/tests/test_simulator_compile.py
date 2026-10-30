# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Analytic checks of compiled ODE maps and direct/adjoint sensitivities."""

from copy import deepcopy
import math

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
    check_min_version,
)
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import casadi as ca, casadi_available


def exponential_model(start=0, end=1.2):
    """Build exponential growth with no assigned state initial conditions."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(start, end))
    model.k = pyo.Param(mutable=True, initialize=-0.7)
    model.x = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.k * m.x[t])
    return model


@unittest.skipUnless(
    numpy_available and casadi_available and check_min_version(ca, '3.8.1'),
    'NumPy and CasADi 3.8.1 or later are required',
)
class TestSimulatorCompile(unittest.TestCase):
    """Exercise the public map without requiring a solver or ASL."""

    def compile(self, model, parameters=None, times=None, **kwds):
        """Construct a map with tight scalar tolerances for the analytic checks."""
        kwds.setdefault('integrator_options', {'reltol': 1e-11, 'abstol': 1e-13})
        return Simulator(model, package='casadi').compile(
            parameters=[model.k] if parameters is None else parameters,
            times=[0, 0.2, 0.7, 1.2] if times is None else times,
            **kwds,
        )

    def assert_close(self, actual, expected, atol=3e-7):
        """Check finite derivatives and explicit array shapes against an oracle."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=atol)

    def test_exponential_forward_and_adjoint(self):
        """Check both independent inputs, multiple observations, and named factories."""
        model = exponential_model()
        flow = self.compile(model)
        self.assertEqual(flow.name_in(), ['x0', 'p'])
        self.assertEqual(flow.name_out(), ['xf'])
        self.assertEqual(flow.size_out('xf'), (1, 4))
        tangent = flow.factory('tangent', ['x0', 'p', 'fwd:x0', 'fwd:p'], ['fwd:xf'])
        adjoint = flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p'])
        times = np.array([0, 0.2, 0.7, 1.2])
        weights = np.array([[0.7, -0.3, 0.5, 1.2]])
        for rate, initial in ((-0.7, 1.3), (-0.2, 0.8), (-0.7, 1.3)):
            with self.subTest(rate=rate, initial=initial):
                e = np.exp(rate * times)
                x = initial * e
                value = flow(x0=[initial], p=[rate])['xf']
                direct = tangent(x0=initial, p=rate, fwd_x0=0.3, fwd_p=-0.2)['fwd_xf']
                reverse = adjoint(x0=initial, p=rate, adj_xf=weights)
                self.assert_close(value, x.reshape(1, -1))
                self.assert_close(direct, (0.3 * e - 0.2 * times * x).reshape(1, -1))
                self.assert_close(reverse['adj_x0'], [[(weights @ e).item()]])
                self.assert_close(reverse['adj_p'], [[(weights @ (times * x)).item()]])
        self.assertIsNone(model.x[0].value)

    def test_multiple_directions_and_duality(self):
        """Check seed block layouts for multiple states, parameters, and directions."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1.2))
        model.rate = pyo.Param([0, 1], mutable=True, initialize={0: -0.7, 1: -1.2})
        model.offset = pyo.Param(mutable=True, initialize=0.3)
        model.x = pyo.Var(model.t, [0, 1])
        model.dx = DerivativeVar(model.x, wrt=model.t)
        model.ode = pyo.Constraint(
            model.t,
            [0, 1],
            rule=lambda m, t, i: m.dx[t, i]
            == m.rate[i] * m.x[t, i] + (m.offset if i == 0 else 0),
        )
        times = np.array([0.2, 0.7, 1.2])
        flow = self.compile(
            model, parameters=[model.rate[0], model.rate[1], model.offset], times=times
        )
        point, initial = [-0.7, -1.2, 0.3], [1.3, -0.2]
        e0, e1 = np.exp(point[0] * times), np.exp(point[1] * times)
        integral = np.expm1(point[0] * times) / point[0]
        expected = np.vstack([initial[0] * e0 + point[2] * integral, initial[1] * e1])
        parameter_jacobian = np.zeros((2, 3, 3))
        parameter_jacobian[0, :, 0] = (
            initial[0] * times * e0
            + point[2] * ((point[0] * times - 1) * e0 + 1) / point[0] ** 2
        )
        parameter_jacobian[1, :, 1] = initial[1] * times * e1
        parameter_jacobian[0, :, 2] = integral
        state_jacobian = np.zeros((2, 3, 2))
        state_jacobian[0, :, 0], state_jacobian[1, :, 1] = e0, e1
        state_seeds = np.array([[0.3, 0.1], [-0.2, 0.4]])
        parameter_seeds = np.array([[0.1, 0.2], [0.5, -0.1], [-0.2, 0.3]])
        output_seeds = np.array(
            [[0.2, -0.4, 0.7, 0.3, 0.1, -0.2], [-0.1, 0.5, 0.2, 0.6, -0.3, 0.4]]
        )
        value = flow(x0=initial, p=point)['xf']
        direct = flow.forward(2)(
            x0=initial, p=point, out_xf=value, fwd_x0=state_seeds, fwd_p=parameter_seeds
        )['fwd_xf']
        reverse = flow.reverse(2)(
            x0=initial, p=point, out_xf=value, adj_xf=output_seeds
        )
        expected_direct = [
            np.einsum('itp,p->it', parameter_jacobian, parameter_seeds[:, i])
            + np.einsum('itp,p->it', state_jacobian, state_seeds[:, i])
            for i in range(2)
        ]
        self.assert_close(value, expected)
        self.assert_close(direct, np.hstack(expected_direct))
        for i in range(2):
            weights = output_seeds[:, 3 * i : 3 * (i + 1)]
            self.assert_close(
                reverse['adj_p'][:, i],
                np.einsum('it,itp->p', weights, parameter_jacobian).reshape(3, 1),
            )
            self.assert_close(
                reverse['adj_x0'][:, i],
                np.einsum('it,itp->p', weights, state_jacobian).reshape(2, 1),
            )
            for j in range(2):
                left = float(
                    np.sum(weights * np.asarray(direct)[:, 3 * j : 3 * (j + 1)])
                )
                right = float(
                    state_seeds[:, j] @ np.asarray(reverse['adj_x0'])[:, i]
                    + parameter_seeds[:, j] @ np.asarray(reverse['adj_p'])[:, i]
                )
                self.assertAlmostEqual(left, right, delta=3e-7)

    def test_nonlinear_ode(self):
        """Check nonlinear state dynamics against rational decay and its derivatives."""
        model = exponential_model()
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == -m.k * m.x[t] ** 2
        )
        flow = self.compile(model)
        times, initial, parameter = np.array([0, 0.2, 0.7, 1.2]), 1.3, 0.7
        denominator = 1 + parameter * initial * times
        self.assert_close(
            flow(x0=initial, p=parameter)['xf'], (initial / denominator).reshape(1, -1)
        )
        direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
            x0=initial, p=parameter, fwd_p=1
        )['fwd_xf']
        self.assert_close(
            direct, (-(initial**2) * times / denominator**2).reshape(1, -1)
        )
        gradient = flow.factory('gradient', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p'])(
            x0=initial, p=parameter, adj_xf=ca.DM([[0.2, 0.3, -0.1, 0.7]])
        )
        weights = np.array([0.2, 0.3, -0.1, 0.7])
        self.assert_close(gradient['adj_x0'], [[float(sum(weights / denominator**2))]])
        self.assert_close(
            gradient['adj_p'],
            [[-float(sum(weights * initial**2 * times / denominator**2))]],
        )

    def test_coupled_ode(self):
        """Use a closed matrix exponential for a coupled two-state system."""
        model = exponential_model()
        model.y = pyo.Var(model.t)
        model.dy = DerivativeVar(model.y)
        model.coupling = pyo.Param(mutable=True, initialize=0.2)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.k * m.x[t] + m.coupling * m.y[t]
        )
        model.other = pyo.Constraint(
            model.t, rule=lambda m, t: m.dy[t] == m.coupling * m.x[t] + m.k * m.y[t]
        )
        times = np.array([0, 0.2, 0.7, 1.2])
        flow = self.compile(model, parameters=[model.k, model.coupling])
        e, c, s = np.exp(-0.7 * times), np.cosh(0.2 * times), np.sinh(0.2 * times)
        expected = np.vstack([e * (c + 0.3 * s), e * (s + 0.3 * c)])
        value = flow(x0=[1, 0.3], p=[-0.7, 0.2])['xf']
        direct = flow.forward(2)(
            x0=[1, 0.3],
            p=[-0.7, 0.2],
            out_xf=value,
            fwd_x0=np.zeros((2, 2)),
            fwd_p=np.eye(2),
        )['fwd_xf']
        self.assert_close(value, expected)
        self.assert_close(direct, np.hstack([times * expected, times * expected[::-1]]))

    def test_stiff_ode(self):
        """Fast/slow states and small fast-mode sensitivities retain accuracy."""
        model = exponential_model(end=1)
        model.fast = pyo.Param(mutable=True, initialize=1000)
        model.y = pyo.Var(model.t)
        model.dy = DerivativeVar(model.y)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        model.other = pyo.Constraint(
            model.t, rule=lambda m, t: m.dy[t] == -m.fast * m.y[t]
        )
        times = np.array([0, 1e-4, 0.01, 0.1, 1])
        for rate, method in ((100, None), (1000, 'adams')):
            with self.subTest(rate=rate, method=method):
                options = {'reltol': 1e-11, 'abstol': 1e-13}
                if method:
                    options['linear_multistep_method'] = method
                flow = self.compile(
                    model,
                    parameters=[model.fast],
                    times=times,
                    integrator_options=options,
                )
                expected = np.vstack([np.exp(-times), np.exp(-rate * times)])
                value = flow(x0=[1, 1], p=[rate])['xf']
                direct = flow.factory('tangent', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
                    x0=[1, 1], p=rate, fwd_p=1
                )['fwd_xf']
                self.assert_close(value, expected, atol=3e-9)
                self.assert_close(
                    direct,
                    np.vstack([np.zeros(len(times)), -times * expected[1]]),
                    atol=3e-10,
                )
                weights = np.zeros((2, len(times)))
                weights[1] = [0, 0.2, 0.3, 0.4, 0.5]
                reverse = flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:p'])(
                    x0=[1, 1], p=rate, adj_xf=weights
                )['adj_p']
                self.assert_close(
                    reverse,
                    [[-float(np.sum(weights[1] * times * expected[1]))]],
                    atol=3e-10,
                )

    def test_tolerance_refinement(self):
        """Analytic adjoint accuracy improves as the requested tolerances tighten."""
        model = exponential_model()
        expected = 1.2 * 1.3 * math.exp(-0.7 * 1.2)
        errors = []
        for tolerance in (1e-3, 1e-7, 1e-11):
            flow = self.compile(
                model,
                times=[1.2],
                integrator_options={'reltol': tolerance, 'abstol': tolerance / 100},
            )
            gradient = flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:p'])(
                x0=1.3, p=-0.7, adj_xf=1
            )['adj_p']
            errors.append(abs(float(gradient) - expected))
        self.assertLess(errors[-1], 3e-7)
        self.assertLess(errors[-1], errors[0] / 10)

    def test_integral_terminal_objective_and_total_derivative(self):
        """Retain integral, explicit terminal, and parameter-dependent IC terms."""
        model = exponential_model()
        model.q = pyo.Var(model.t)
        model.dq = DerivativeVar(model.q)
        model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
        flow = self.compile(model, times=[1.2])
        parameter = ca.MX.sym('parameter')
        initial = ca.MX.sym('initial', 2)
        terminal = flow(x0=initial, p=parameter)['xf']
        cost = 2 * terminal[0] + 3 * terminal[1] + parameter**2
        objective = ca.Function(
            'objective', [initial, parameter], [cost], ['x0', 'p'], ['J']
        )
        point, a, T = -0.7, 1.3, 1.2
        e = math.exp(point * T)
        integral = math.expm1(point * T) / point
        derivative = a * ((point * T - 1) * e + 1) / point**2
        adjoint = objective.factory(
            'gradient', ['x0', 'p', 'adj:J'], ['adj:x0', 'adj:p']
        )(x0=[a, 0], p=point, adj_J=1)
        self.assert_close(flow(x0=[a, 0], p=point)['xf'], [[a * e], [a * integral]])
        self.assert_close(adjoint['adj_x0'], [[2 * e + 3 * integral], [3]])
        self.assert_close(
            adjoint['adj_p'], [[2 * T * a * e + 3 * derivative + 2 * point]]
        )
        total_cost = objective(x0=ca.vertcat(parameter**2, 0), p=parameter)['J']
        total = ca.Function('total', [parameter], [total_cost])
        value = total(point)
        direct = total.forward(1)(point, value, 1)
        reverse = total.reverse(1)(point, value, 1)
        expected = (
            2 * T * point**2 * e
            + 3 * point**2 * ((point * T - 1) * e + 1) / point**2
            + 2 * point
            + (2 * e + 3 * integral) * 2 * point
        )
        self.assert_close(direct, [[expected]])
        self.assert_close(reverse, [[expected]])
        errors = []
        for step in (1e-2, 1e-3, 1e-4, 1e-5):
            difference = float((total(point + step) - total(point - step)) / (2 * step))
            errors.append(abs(difference - expected))
        self.assertLess(min(errors), 3e-7)
        self.assertLess(min(errors[1:]), errors[0] / 10)

    def test_explicit_sensitivity_equations(self):
        """Compare AD products to independently written direct and adjoint ODEs."""
        model = exponential_model()
        flow = self.compile(model, times=[1.2])
        direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
            x0=1.3, p=-0.7, fwd_p=1
        )['fwd_xf']
        adjoint = flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p'])(
            x0=1.3, p=-0.7, adj_xf=1
        )
        oracle = exponential_model()
        oracle.sensitivity = pyo.Var(oracle.t)
        oracle.ds = DerivativeVar(oracle.sensitivity)
        oracle.adjoint = pyo.Var(oracle.t)
        oracle.da = DerivativeVar(oracle.adjoint)
        oracle.gradient = pyo.Var(oracle.t)
        oracle.dg = DerivativeVar(oracle.gradient)
        oracle.direct_equation = pyo.Constraint(
            oracle.t, rule=lambda m, t: m.ds[t] == m.k * m.sensitivity[t] + m.x[t]
        )
        # The last two equations use reversed time sigma=T-t.
        oracle.adjoint_equation = pyo.Constraint(
            oracle.t, rule=lambda m, t: m.da[t] == m.k * m.adjoint[t]
        )
        oracle.gradient_equation = pyo.Constraint(
            oracle.t,
            rule=lambda m, t: m.dg[t] == m.adjoint[t] * 1.3 * pyo.exp(m.k * (1.2 - t)),
        )
        sim = Simulator(oracle, package='casadi')
        _, result = sim.simulate(
            numpoints=11,
            integrator='cvodes',
            initcon=[1.3, 0, 1, 0],
            integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
        )
        self.assertAlmostEqual(float(direct), result[-1, 1], delta=3e-7)
        self.assertAlmostEqual(float(adjoint['adj_x0']), result[-1, 2], delta=3e-7)
        self.assertAlmostEqual(float(adjoint['adj_p']), result[-1, 3], delta=3e-7)

    def test_parameter_order_and_snapshots(self):
        """Selected subsets remain symbolic while unselected scalar data are frozen."""
        model = exponential_model()
        model.b = pyo.Param(mutable=True, initialize=0.2)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == (m.k + 2 * m.b) * m.x[t]
        )
        sim = Simulator(model, package='casadi')
        parameters = [model.b, model.k]
        times = [1.2]
        full = sim.compile(
            parameters=parameters,
            times=times,
            integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
        )
        subset = sim.compile(
            parameters=[model.b],
            times=times,
            integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
        )
        parameters.reverse()
        times[0] = 0.5
        model.k = -0.9
        model.b = 99
        self.assert_close(full(x0=1, p=[0.1, -0.7])['xf'], [[math.exp(-0.5 * 1.2)]])
        self.assert_close(subset(x0=1, p=0.1)['xf'], [[math.exp(-0.5 * 1.2)]])
        recompiled = self.compile(model, parameters=[model.b], times=[1.2])
        self.assert_close(recompiled(x0=1, p=0.1)['xf'], [[math.exp(-0.7 * 1.2)]])

    def test_stationary_and_nonlinear_parameters(self):
        """Nonlinear parameter leaves stay symbolic, including at zero derivatives."""
        for function, derivative, point in (
            (lambda p: p**2, lambda p: 2 * p, 0),
            (pyo.sin, math.cos, 0.2),
            (pyo.exp, math.exp, 0.2),
            (pyo.log, lambda p: 1 / p, 1.3),
        ):
            with self.subTest(function=function, point=point):
                model = exponential_model()
                model.k = point
                model.rate = pyo.Expression(expr=function(model.k))
                model.del_component(model.ode)
                model.ode = pyo.Constraint(
                    model.t, rule=lambda m, t: m.dx[t] == m.rate * m.x[t]
                )
                flow = self.compile(model, times=[1.2])
                rate = float(pyo.value(function(point)))
                expected = math.exp(rate * 1.2)
                direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
                    x0=1, p=point, fwd_p=1
                )['fwd_xf']
                self.assert_close(flow(x0=1, p=point)['xf'], [[expected]])
                self.assert_close(direct, [[1.2 * expected * derivative(point)]])

    def test_uninitialized_selected_parameter(self):
        """Compilation does not require numerical parameter or initial-state values."""
        model = exponential_model()
        model.k.set_value(None)
        flow = self.compile(model, times=[1.2])
        self.assert_close(flow(x0=1.3, p=-0.7)['xf'], [[1.3 * math.exp(-0.7 * 1.2)]])

    def test_empty_parameters(self):
        """A map with no selected parameters still supports initial-state adjoints."""
        model = exponential_model()
        model.del_component(model.ode)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == 2)
        flow = self.compile(model, parameters=[])
        self.assertEqual(flow.size_in('p'), (0, 1))
        self.assert_close(flow(x0=1, p=[])['xf'], [[1, 1.4, 2.4, 3.4]])
        gradient = flow.factory('gradient', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p'])(
            x0=1, p=[], adj_xf=ca.DM([[1, 2, 3, 4]])
        )
        self.assert_close(gradient['adj_x0'], [[10]])
        self.assertEqual(gradient['adj_p'].numel(), 0)

    def test_observation_grids_and_physical_time(self):
        """Initial-only and shifted grids retain physical-time parameter dependence."""
        model = exponential_model(2, 3)
        model.del_component(model.ode)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.k * t)
        for times in ([2], [2.5], [2, 2.2, 3]):
            with self.subTest(times=times):
                flow = self.compile(model, times=times)
                delta = (np.array(times) ** 2 - 4) / 2
                self.assert_close(
                    flow(x0=1, p=0.3)['xf'], (1 + 0.3 * delta).reshape(1, -1)
                )
                direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
                    x0=1, p=0.3, fwd_p=1
                )['fwd_xf']
                gradient = flow.factory(
                    'adjoint', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p']
                )(x0=1, p=0.3, adj_xf=np.ones((1, len(times))))
                self.assert_close(direct, delta.reshape(1, -1))
                self.assert_close(gradient['adj_p'], [[sum(delta)]])
                self.assert_close(gradient['adj_x0'], [[len(times)]])

    def test_invalid_parameters(self):
        """Reject duplicates, unused, immutable, foreign, time-indexed and Var selections."""
        model = exponential_model()
        model.constant = pyo.Param(initialize=1)
        model.unused = pyo.Param(mutable=True, initialize=1)
        model.indexed = pyo.Param([1, 2], mutable=True, initialize=1)
        model.control = pyo.Param(model.t, mutable=True, initialize=1)
        other = exponential_model()
        sim = Simulator(model, package='casadi')
        selections = (
            [model.k, model.k],
            [model.constant],
            [model.unused],
            [other.k],
            [model.x[0]],
            [model.indexed],
            [model.control[0]],
            [None],
            [1],
        )
        for parameters in selections:
            with (
                self.subTest(parameters=[type(p).__name__ for p in parameters]),
                self.assertRaises(ValueError),
            ):
                sim.compile(parameters=parameters, times=[1])
        for parameters in ({'k': model.k}, frozenset(), None):
            with self.subTest(type=type(parameters)), self.assertRaises(TypeError):
                sim.compile(parameters=parameters, times=[1])

    def test_invalid_times(self):
        """Reject non-vector, nonfinite, unordered, duplicate, and out-of-domain times."""
        model = exponential_model()
        sim = Simulator(model, package='casadi')
        for times in (
            [],
            0.5,
            [[0.5]],
            [float('nan')],
            [float('inf')],
            [-0.1],
            [1.3],
            [0.2, 0.2],
            [0.7, 0.2],
            ['invalid'],
        ):
            with self.subTest(times=times), self.assertRaises(ValueError):
                sim.compile(parameters=[model.k], times=times)

    def test_invalid_options(self):
        """Refuse vector weights and overrides of the primal map's time/seed layout."""
        model = exponential_model()
        sim = Simulator(model, package='casadi')
        for option in (
            'abstolv',
            'augmented_options',
            'nfwd',
            'nadj',
            't0',
            'tf',
            'grid',
            'output_t0',
        ):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, 'does not support'),
            ):
                sim.compile(
                    parameters=[model.k], times=[1], integrator_options={option: 1}
                )
        for name in ('abstol', 'reltol'):
            for tolerance in (0, -1, float('inf'), float('nan'), [1e-12]):
                with (
                    self.subTest(name=name, tolerance=tolerance),
                    self.assertRaises(ValueError),
                ):
                    sim.compile(
                        parameters=[model.k],
                        times=[1],
                        integrator_options={name: tolerance},
                    )
        with self.assertRaises(TypeError):
            sim.compile(parameters=[model.k], times=[1], integrator_options=[])
        with self.assertRaisesRegex(ValueError, 'quad_err_con=True'):
            sim.compile(
                parameters=[model.k],
                times=[1],
                integrator_options={'quad_err_con': False},
            )

    def test_direction_batch_option(self):
        """Use supported native batching, rejecting the defective older configuration."""
        model = exponential_model()
        options = {'reltol': 1e-11, 'abstol': 1e-13, 'max_num_dir': 2}
        if not check_min_version(ca, '3.8.2'):
            with self.assertRaisesRegex(ValueError, 'max_num_dir=1'):
                self.compile(model, times=[1.2], integrator_options=options)
        else:
            flow = self.compile(model, times=[1.2], integrator_options=options)
            value = flow(x0=1.3, p=-0.7)['xf']
            gradient = flow.reverse(2)(
                x0=1.3, p=-0.7, out_xf=value, adj_xf=ca.DM([[1, 2]])
            )['adj_p']
            expected = 1.2 * 1.3 * math.exp(-0.7 * 1.2)
            self.assert_close(gradient, [[expected, 2 * expected]])

    def test_unsupported_models(self):
        """Require explicit IDAS selection and complete time-varying input selection."""
        for controlled in (False, True):
            with self.subTest(controlled=controlled):
                model = exponential_model()
                model.z = pyo.Var(model.t)
                model.del_component(model.ode)
                model.ode = pyo.Constraint(
                    model.t, rule=lambda m, t: m.dx[t] == m.k * m.z[t]
                )
                if not controlled:
                    model.alg = pyo.Constraint(
                        model.t, rule=lambda m, t: m.z[t] == m.x[t]
                    )
                sim = Simulator(model, package='casadi')
                with self.assertRaisesRegex(
                    DAE_Error, "integrator='idas'|time-varying inputs"
                ):
                    sim.compile(parameters=[model.k], times=[1])
        model = exponential_model()
        sim = Simulator(model, package='casadi')
        with self.assertRaisesRegex(DAE_Error, 'cvodes'):
            sim.compile(parameters=[model.k], times=[1], integrator='idas')
        model.del_component(model.ode)
        model.inequality = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] >= m.x[t])
        with self.assertRaisesRegex(DAE_Error, 'at least one differential equation'):
            Simulator(model, package='casadi').compile(parameters=[model.k], times=[1])

    @unittest.skipUnless(
        scipy_available, 'SciPy is required to construct a SciPy simulator'
    )
    def test_scipy_mode_rejected(self):
        """The new method has no accidental dependency on a SciPy-compiled RHS."""
        model = exponential_model()
        with self.assertRaisesRegex(DAE_Error, "package='casadi'"):
            Simulator(model, package='scipy').compile(parameters=[model.k], times=[1])

    def test_state_and_options_preservation(self):
        """Compilation and derivative evaluation preserve successful simulation state."""
        model = exponential_model()
        model.x[0].fix(1.3)
        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        model.scaling_factor[model.x[0]] = 10
        sim = Simulator(model, package='casadi')
        previous_times, previous_profiles = sim.simulate(
            numpoints=5, integrator='cvodes'
        )
        before = [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        options = {'reltol': 1e-11, 'abstol': 1e-13}
        original_options = deepcopy(options)
        flow = sim.compile(
            parameters=[model.k], times=[0.2, 1.2], integrator_options=options
        )
        flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
            x0=1.3, p=-0.7, fwd_p=1
        )
        flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:p'])(
            x0=1.3, p=-0.7, adj_xf=ca.DM([[0, 1]])
        )
        invalid = {'augmented_options': {'abstolv': [1e-12]}}
        with self.assertRaises(ValueError):
            sim.compile(parameters=[model.k], times=[1], integrator_options=invalid)
        unknown = {'invalid_option': 1}
        with self.assertRaises(RuntimeError):
            sim.compile(parameters=[model.k], times=[1], integrator_options=unknown)
        self.assertEqual(options, original_options)
        self.assertEqual(invalid, {'augmented_options': {'abstolv': [1e-12]}})
        self.assertEqual(unknown, {'invalid_option': 1})
        self.assertEqual(
            before, [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        )
        self.assertEqual(model.k.value, -0.7)
        self.assertTrue(model.ode.active)
        self.assertEqual(model.scaling_factor[model.x[0]], 10)
        np.testing.assert_array_equal(sim._tsim, previous_times)
        np.testing.assert_array_equal(sim._simsolution, previous_profiles)
        self.assertEqual(sim.get_variable_order('algebraic'), [])
        self.assertEqual(sim.get_variable_order('input'), {})


if __name__ == '__main__':
    unittest.main()
