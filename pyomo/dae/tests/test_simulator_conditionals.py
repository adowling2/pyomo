# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Generic declarative conditional values and branch-interior sensitivities."""

import math

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available, check_min_version
import pyomo.environ as pyo
import pyomo.core.expr as EXPR
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.simulator import (
    casadi as ca,
    casadi_available,
    substitute_pyomo2casadi,
    convert_pyomo2casadi,
)


def conditional_decay():
    """Build a scalar ODE with two invariant-sign solution branches."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 0.8))
    model.x = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.k = pyo.Param(mutable=True, initialize=0.6)
    model.ode = pyo.Constraint(
        model.t,
        rule=lambda m, t: m.dx[t]
        == pyo.Expr_if(IF=m.x[t] >= 0, THEN=-m.k * m.x[t], ELSE=-2 * m.k * m.x[t]),
    )
    return model


@unittest.skipUnless(
    numpy_available and casadi_available and check_min_version(ca, '3.8.1'),
    'NumPy and CasADi 3.8.1 or later are required',
)
class TestSimulatorConditionals(unittest.TestCase):
    """Check conversion and public maps against selected-branch mathematics."""

    def assert_close(self, actual, expected):
        """Require matching dimensions, finite values and analytic agreement."""
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)

    def test_nested_expression_values_and_local_derivatives(self):
        """Nested predicates retain symbolic state and scalar-parameter dependencies."""
        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.k = pyo.Param(mutable=True, initialize=2)
        expression = pyo.Expr_if(
            IF=model.x < -1,
            THEN=model.x**3 + model.k,
            ELSE=pyo.Expr_if(
                IF=model.x <= 1,
                THEN=model.k * model.x**2,
                ELSE=pyo.log(model.x) + model.k,
            ),
        )
        x, k = ca.SX.sym('x'), ca.SX.sym('k')
        converted = convert_pyomo2casadi(
            substitute_pyomo2casadi(expression, {}),
            substitutions={id(model.x): x, id(model.k): k},
        )
        dx = ca.jacobian(converted, x)
        function = ca.Function(
            'conditional',
            [x, k],
            [converted, dx, ca.jacobian(converted, k), ca.jacobian(dx, x)],
        )
        for point, expected in (
            (-2, (-6, 12, 1, -12)),
            (0.4, (0.32, 1.6, 0.16, 4)),
            (3, (math.log(3) + 2, 1 / 3, 1, -1 / 9)),
        ):
            with self.subTest(point=point):
                result = function(point, 2)
                self.assert_close(np.array([float(v) for v in result]), expected)
        # Equality selects the declared branch; no derivative claim at a jump.
        for point in (-1, 1):
            self.assertAlmostEqual(float(function(point, 2)[0]), 2)

    def test_parameter_only_expression(self):
        """Non-variable conditional nodes remain symbolic when a Param is selected."""
        model = pyo.ConcreteModel()
        model.k = pyo.Param(mutable=True, initialize=-7)
        expression = pyo.Expr_if(
            IF=model.k >= 0, THEN=model.k**2 + 1, ELSE=-3 * model.k
        )
        numeric_condition = pyo.Expr_if(
            IF=model.k, THEN=model.k**2 + 1, ELSE=-3 * model.k
        )
        self.assertIs(type(numeric_condition), EXPR.NPV_Expr_ifExpression)
        k = ca.SX.sym('k')
        for expression, cases in (
            (expression, ((1.5, (3.25, 3)), (-0.5, (1.5, -3)))),
            (numeric_condition, ((1.5, (3.25, 3)), (-0.5, (1.25, -1)))),
        ):
            with self.subTest(node_type=type(expression).__name__):
                converted = convert_pyomo2casadi(
                    substitute_pyomo2casadi(expression, {}),
                    substitutions={id(model.k): k},
                )
                function = ca.Function(
                    'parameter_conditional', [k], [converted, ca.jacobian(converted, k)]
                )
                for point, expected in cases:
                    with self.subTest(point=point):
                        self.assert_close(
                            np.array([float(v) for v in function(point)]), expected
                        )
        self.assertEqual(float(function(0)[0]), 0)
        self.assertEqual(model.k.value, -7)

    def test_inactive_expression_domains(self):
        """Selected interiors of these log/sqrt branches have finite local derivatives."""
        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        x = ca.SX.sym('x')
        for operator in (pyo.log, pyo.sqrt):
            expression = pyo.Expr_if(
                IF=model.x > 1, THEN=operator(model.x - 1), ELSE=operator(1 - model.x)
            )
            converted = convert_pyomo2casadi(
                substitute_pyomo2casadi(expression, {}), substitutions={id(model.x): x}
            )
            dx = ca.jacobian(converted, x)
            function = ca.Function(
                'domain_conditional', [x], [converted, dx, ca.jacobian(dx, x)]
            )
            for point in (-1, 3):
                with self.subTest(operator=operator.__name__, point=point):
                    sign = 1 if point > 1 else -1
                    expected = (
                        (math.log(2), sign / 2, -1 / 4)
                        if operator is pyo.log
                        else (
                            math.sqrt(2),
                            sign / (2 * math.sqrt(2)),
                            -1 / (4 * 2**1.5),
                        )
                    )
                    self.assert_close(
                        np.array([float(v) for v in function(point)]), expected
                    )

    def test_constant_boolean_predicates(self):
        """Native boolean leaves select variable and parameter-only branches."""
        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.k = pyo.Param(mutable=True, initialize=-7)
        symbol = ca.SX.sym('value')
        for condition in (True, False, np.bool_(True), np.bool_(False)):
            for leaf in (model.x, model.k):
                with self.subTest(condition=condition, leaf=leaf.name):
                    expression = pyo.Expr_if(IF=condition, THEN=leaf**2, ELSE=-leaf)
                    converted = convert_pyomo2casadi(
                        substitute_pyomo2casadi(expression, {}),
                        substitutions={id(leaf): symbol},
                    )
                    function = ca.Function(
                        'boolean_conditional',
                        [symbol],
                        [converted, ca.jacobian(converted, symbol)],
                    )
                    expected = (4, 4) if condition else (-2, -1)
                    self.assert_close(
                        np.array([float(v) for v in function(2)]), expected
                    )

    def test_public_ode_direct_and_adjoint(self):
        """Both invariant branches match exact x0/parameter products at several times."""
        model = conditional_decay()
        times = np.array([0, 0.25, 0.8])
        flow = Simulator(model, package='casadi').compile(
            parameters=[model.k],
            times=times,
            integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
        )
        direct = flow.factory('direct', ['x0', 'p', 'fwd:x0', 'fwd:p'], ['fwd:xf'])
        adjoint = flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p'])
        weights = np.array([[0.3, -0.4, 0.7]])
        for initial in (-1.5, 1.5):
            with self.subTest(initial=initial):
                factor = 1 if initial > 0 else 2
                e = np.exp(-factor * 0.6 * times)
                expected = initial * e
                dk = -factor * times * expected
                self.assert_close(
                    flow(x0=initial, p=0.6)['xf'], expected.reshape(1, -1)
                )
                tangent = direct(
                    x0=initial, p=0.6, fwd_x0=ca.DM([[1, 0]]), fwd_p=ca.DM([[0, 1]])
                )
                self.assert_close(tangent['fwd_xf'], np.hstack([e, dk]).reshape(1, -1))
                gradient = adjoint(x0=initial, p=0.6, adj_xf=weights)
                self.assert_close(gradient['adj_x0'], [[(weights @ e).item()]])
                self.assert_close(gradient['adj_p'], [[(weights @ dk).item()]])

    def test_public_ode_parameter_predicate(self):
        """A selected parameter predicate chooses its branch at evaluation."""
        model = conditional_decay()
        model.del_component(model.ode)
        rate = pyo.Expr_if(IF=model.k >= 0, THEN=model.k + 1, ELSE=model.k**2 + 1)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -rate * m.x[t])
        times = np.array([0, 0.25, 0.8])
        flow = Simulator(model, package='casadi').compile(
            parameters=[model.k],
            times=times,
            integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
        )
        direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])
        adjoint = flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:p'])
        for point in (-0.5, 0.75):
            with self.subTest(point=point):
                rate, dr = (point + 1, 1) if point >= 0 else (point**2 + 1, 2 * point)
                expected = 1.3 * np.exp(-rate * times)
                derivative = -dr * times * expected
                self.assert_close(flow(x0=1.3, p=point)['xf'], expected.reshape(1, -1))
                self.assert_close(
                    direct(x0=1.3, p=point, fwd_p=1)['fwd_xf'],
                    derivative.reshape(1, -1),
                )
                self.assert_close(
                    adjoint(x0=1.3, p=point, adj_xf=np.ones((1, 3)))['adj_p'],
                    [[derivative.sum()]],
                )

    def test_public_dae_controls_and_algebraic_products(self):
        """Conditional residuals retain IC/parameter/amplitude partials and switch values."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 0.6))
        model.x, model.z, model.u = pyo.Var(model.t), pyo.Var(model.t), pyo.Var(model.t)
        model.dx = DerivativeVar(model.x)
        model.k, model.a = pyo.Param(mutable=True, initialize=0.4), pyo.Param(
            mutable=True, initialize=1.6
        )
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.k * m.x[t])
        model.alg = pyo.Constraint(
            model.t,
            rule=lambda m, t: m.z[t]
            == pyo.Expr_if(
                IF=m.x[t] >= 0,
                THEN=m.a * m.x[t] + m.u[t],
                ELSE=2 * m.a * m.x[t] + m.u[t],
            ),
        )
        times = np.array([0, 0.3, 0.6])
        flow = Simulator(model, package='casadi').compile(
            parameters=[model.k, model.a],
            times=times,
            integrator='idas',
            controls=[model.u],
            control_times=times,
            integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
        )
        e = np.exp(-0.4 * times)
        wx, wz = np.array([[0.2, -0.4, 0.7]]), np.array([[-0.3, 0.8, 0.5]])
        direct = flow.factory(
            'direct',
            ['x0', 'p', 'z0', 'u', 'fwd:x0', 'fwd:p', 'fwd:z0', 'fwd:u'],
            ['fwd:xf', 'fwd:zf'],
        )
        adjoint = flow.factory(
            'adjoint',
            ['x0', 'p', 'z0', 'u', 'adj:xf', 'adj:zf'],
            ['adj:x0', 'adj:p', 'adj:z0', 'adj:u'],
        )
        for initial in (-1.2, 1.2):
            with self.subTest(initial=initial):
                factor = 1 if initial > 0 else 2
                x = initial * e
                point = {
                    'x0': initial,
                    'p': [0.4, 1.6],
                    'z0': factor * 1.6 * initial + 0.2,
                    'u': ca.DM([[0.2, -0.1]]),
                }
                result = flow(**point)
                self.assert_close(result['xf'], x.reshape(1, -1))
                self.assert_close(
                    result['zf'],
                    (factor * 1.6 * x + np.array([0.2, -0.1, -0.1])).reshape(1, -1),
                )
                tangent = direct(
                    **point,
                    fwd_x0=0.25,
                    fwd_p=[-0.2, 0.3],
                    fwd_z0=1,
                    fwd_u=ca.DM([[0.4, -0.5]]),
                )
                dx = 0.25 * e + 0.2 * times * x
                dz = factor * 1.6 * dx + factor * 0.3 * x + np.array([0.4, -0.5, -0.5])
                self.assert_close(tangent['fwd_xf'], dx.reshape(1, -1))
                self.assert_close(tangent['fwd_zf'], dz.reshape(1, -1))
                gradient = adjoint(**point, adj_xf=wx, adj_zf=wz)
                w = wx + factor * 1.6 * wz
                self.assert_close(gradient['adj_x0'], [[(w @ e).item()]])
                self.assert_close(
                    gradient['adj_p'],
                    [[(w @ (-times * x)).item()], [(wz @ (factor * x)).item()]],
                )
                self.assert_close(gradient['adj_u'], [[wz[0, 0], wz[0, 1] + wz[0, 2]]])
                self.assert_close(gradient['adj_z0'], [[0]])

    def test_legacy_simulate(self):
        """Existing CasADi simulation supports the same declarative conditional ODE."""
        model = conditional_decay()
        sim = Simulator(model, package='casadi')
        for initial in (-1.5, 1.5):
            with self.subTest(initial=initial):
                times, profiles = sim.simulate(
                    numpoints=5,
                    initcon=[initial],
                    integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
                )
                factor = 1 if initial > 0 else 2
                self.assert_close(
                    profiles, (initial * np.exp(-factor * 0.6 * times)).reshape(-1, 1)
                )


if __name__ == '__main__':
    unittest.main()
