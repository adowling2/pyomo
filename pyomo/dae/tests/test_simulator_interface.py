# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Analytic and state-preservation tests of the public Simulator interface."""

from copy import deepcopy
import math
from pathlib import Path

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
from pyomo.common.fileutils import import_file
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import casadi_available


def create_model(start=0, end=1):
    """Create a scalar state with a mutable decay coefficient."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(start, end))
    model.x = pyo.Var(model.t, initialize=1)
    model.dx = DerivativeVar(model.x)
    model.k = pyo.Param(initialize=2, mutable=True)
    return model


def piecewise_solution(times, profile, initial=1, rate=0):
    """Evaluate the exact flow of x_dot=rate*x+u for a fixed input profile."""
    switches = sorted(profile)
    result = []
    for time in times:
        state = initial
        for index, start in enumerate(switches):
            if time <= start:
                break
            end = min(time, switches[index + 1]) if index + 1 < len(switches) else time
            duration = end - start
            if rate:
                state = (
                    math.exp(rate * duration) * state
                    + profile[start] * math.expm1(rate * duration) / rate
                )
            else:
                state += profile[start] * duration
        result.append(state)
    return np.asarray(result)


class SimulatorChecks:
    """Exercise the same public contracts with both integration backends."""

    def options(self):
        """Return backend-specific tolerances for the analytic examples."""
        if self.package == 'scipy':
            return {'rtol': 1e-10, 'atol': 1e-12}
        return {'reltol': 1e-10, 'abstol': 1e-12}

    def simulate(self, model, **kwds):
        """Construct a Simulator and evaluate a short trajectory."""
        simulator = Simulator(model, package=self.package)
        kwds.setdefault('numpoints', 11)
        kwds.setdefault('integrator_options', self.options())
        return simulator, simulator.simulate(**kwds)

    def assert_profile(self, times, profiles, expected):
        """Check trajectory dimensions, finiteness, and analytic accuracy."""
        expected = np.asarray(expected)
        if expected.ndim == 1:
            expected = expected.reshape(-1, 1)
        self.assertEqual(profiles.shape, (len(times), expected.shape[1]))
        self.assertTrue(np.isfinite(profiles).all())
        np.testing.assert_allclose(profiles, expected, rtol=2e-7, atol=2e-7)

    def test_equivalent_equation_forms(self):
        """Equation rearrangement must preserve every term and derivative sign."""
        forms = [
            (lambda m, t: m.dx[t] == -m.k * m.x[t], lambda t: np.exp(-2 * t)),
            (lambda m, t: -m.k * m.x[t] == m.dx[t], lambda t: np.exp(-2 * t)),
            (lambda m, t: m.dx[t] + m.k * m.x[t] == 0, lambda t: np.exp(-2 * t)),
            (lambda m, t: m.k * m.x[t] + m.dx[t] == 0, lambda t: np.exp(-2 * t)),
            (lambda m, t: m.x[t] * m.x[t] + m.dx[t] == 0, lambda t: 1 / (1 + t)),
            (lambda m, t: m.dx[t] + m.x[t] * m.x[t] == 0, lambda t: 1 / (1 + t)),
            (lambda m, t: 2 * m.x[t] + 3 * m.dx[t] + m.x[t] == 0, lambda t: np.exp(-t)),
            (lambda m, t: 0 == m.k * m.x[t] + m.dx[t], lambda t: np.exp(-2 * t)),
            (lambda m, t: -m.dx[t] == m.k * m.x[t], lambda t: np.exp(-2 * t)),
            (
                lambda m, t: m.x[t] * m.dx[t] == -m.k * m.x[t] ** 2,
                lambda t: np.exp(-2 * t),
            ),
            (lambda m, t: 1 / (m.dx[t] / m.x[t]) == 0.5, lambda t: np.exp(2 * t)),
            (lambda m, t: m.x[t] / (2 / m.dx[t]) == 2 * m.x[t], lambda t: 1 + 4 * t),
        ]
        for index, (rule, analytic) in enumerate(forms):
            with self.subTest(form=index):
                model = create_model()
                model.ode = pyo.Constraint(model.t, rule=rule)
                _, (times, profiles) = self.simulate(model)
                self.assert_profile(times, profiles, analytic(times))

    def test_constant_rhs(self):
        """Zero and constant derivatives do not require a callable RHS leaf."""
        for slope in (0, 2, -0.25):
            with self.subTest(slope=slope):
                model = create_model()
                model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == slope)
                _, (times, profiles) = self.simulate(model)
                self.assert_profile(times, profiles, 1 + slope * times)

    def test_coupled_linear_ode(self):
        """Check coupled modes against their exact exponential solution."""
        model = create_model()
        model.y = pyo.Var(model.t, initialize=0)
        model.dy = DerivativeVar(model.y)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == -2 * m.x[t] + m.y[t]
        )
        model.other = pyo.Constraint(
            model.t, rule=lambda m, t: m.dy[t] == m.x[t] - 2 * m.y[t]
        )
        sim, (times, profiles) = self.simulate(model)
        self.assertEqual(len(sim.get_variable_order()), 2)
        self.assert_profile(
            times,
            profiles,
            np.column_stack(
                [
                    (np.exp(-times) + np.exp(-3 * times)) / 2,
                    (np.exp(-times) - np.exp(-3 * times)) / 2,
                ]
            ),
        )

    def test_stiff_ode(self):
        """Check resolved fast and slow modes without a nonlinear optimizer."""
        model = create_model()
        model.y = pyo.Var(model.t, initialize=1)
        model.dy = DerivativeVar(model.y)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        model.other = pyo.Constraint(
            model.t, rule=lambda m, t: m.dy[t] == -1000 * m.y[t]
        )
        _, (times, profiles) = self.simulate(model)
        self.assert_profile(
            times, profiles, np.column_stack([np.exp(-times), np.exp(-1000 * times)])
        )

    def test_integrators(self):
        """All advertised ODE integrators respect fixed control changes."""
        if self.package == 'scipy':
            integrators = ('lsoda', 'vode', 'zvode', 'dopri5', 'dop853', 'odeint')
        else:
            integrators = ('cvodes', 'idas', 'collocation', 'rk')
        for integrator in integrators:
            with self.subTest(integrator=integrator):
                model = create_model()
                model.u = pyo.Var(model.t)
                model.ode = pyo.Constraint(
                    model.t, rule=lambda m, t: m.dx[t] == -0.7 * m.x[t] + m.u[t]
                )
                model.inputs = pyo.Suffix()
                profile = {0: 1, 0.3: 3, 0.65: -0.5}
                model.inputs[model.u] = profile
                options = (
                    {'number_of_finite_elements': 100}
                    if integrator in ('collocation', 'rk')
                    else self.options()
                )
                _, (times, profiles) = self.simulate(
                    model,
                    integrator=integrator,
                    integrator_options=options,
                    varying_inputs=model.inputs,
                )
                self.assert_profile(
                    times, profiles, piecewise_solution(times, profile, rate=-0.7)
                )

    def test_explicit_time(self):
        """The independent variable denotes physical time, including shifted grids."""
        for start in (0, 2):
            with self.subTest(start=start):
                model = create_model(start, start + 1)
                model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == t)
                _, (times, profiles) = self.simulate(model)
                self.assert_profile(times, profiles, 1 + (times**2 - start**2) / 2)

    def test_indexed_states(self):
        """Multi-indexed state profiles follow the reported state order."""
        model = pyo.ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1))
        model.species = pyo.Set(initialize=[1, 2], ordered=True)
        model.x = pyo.Var(model.species, model.t, initialize=1)
        model.dx = DerivativeVar(model.x, wrt=model.t)
        model.ode = pyo.Constraint(
            model.species, model.t, rule=lambda m, i, t: m.dx[i, t] == -i * m.x[i, t]
        )
        sim, (times, profiles) = self.simulate(model)
        self.assertEqual(
            [str(v) for v in sim.get_variable_order()], ['x[1,{t}]', 'x[2,{t}]']
        )
        self.assert_profile(
            times, profiles, np.column_stack([np.exp(-times), np.exp(-2 * times)])
        )

    def test_output_grid(self):
        """Point counts and step grids include the full simulation interval."""
        model = create_model(2, 3)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        sim = Simulator(model, package=self.package)
        grids = [{'numpoints': n} for n in (2, 9, np.int64(7))]
        grids += [{'tstep': dt} for dt in (0.25, 0.3, 1, np.float64(0.4))]
        for grid in grids:
            with self.subTest(grid=grid):
                times, profiles = sim.simulate(
                    **grid, integrator_options=self.options()
                )
                self.assertEqual(times[0], 2)
                self.assertEqual(times[-1], 3)
                self.assertTrue((np.diff(times) > 0).all())
                if 'numpoints' in grid:
                    self.assertEqual(len(times), grid['numpoints'])
                self.assert_profile(times, profiles, np.exp(-(times - 2)))
                sim.initialize_model()
                self.assertAlmostEqual(model.x[3].value, math.exp(-1), delta=2e-7)

    def test_invalid_grid(self):
        """Reject empty, degenerate, nonfinite, and negative grids before integration."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        sim = Simulator(model, package=self.package)
        grids = [{'numpoints': n} for n in (0, 1, -1, 2.5, True, float('nan'))]
        grids += [{'tstep': dt} for dt in (0, -1, float('nan'), float('inf'), 2, True)]
        grids.append({'numpoints': 10, 'tstep': 0.1})
        for grid in grids:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                sim.simulate(**grid)

    def test_switches_adjacent_to_endpoints(self):
        """Exact interval endpoints and close switches retain finite pulse areas."""
        for start, end, switch, amplitude in (
            (0, 1, np.nextafter(1.0, 0.0), 1e16),
            (1, 2, np.nextafter(1.0, 2.0), 1e16),
        ):
            with self.subTest(start=start):
                model = create_model(start, end)
                model.u = pyo.Var(model.t)
                model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
                model.inputs = pyo.Suffix()
                profile = (
                    {start: 0, switch: amplitude}
                    if start == 0
                    else {start: amplitude, switch: 0}
                )
                model.inputs[model.u] = profile
                _, (times, profiles) = self.simulate(
                    model, numpoints=2, varying_inputs=model.inputs
                )
                np.testing.assert_array_equal(times, [start, switch, end])
                self.assert_profile(times, profiles, piecewise_solution(times, profile))

    def test_distinct_adjacent_switches(self):
        """Grid snapping cannot merge distinct user-specified switching times."""
        model = create_model()
        model.u = pyo.Var(model.t)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
        model.inputs = pyo.Suffix()
        switch = np.nextafter(0.5, 1.0)
        profile = {0: 0, 0.5: 1e16, switch: 0}
        model.inputs[model.u] = profile
        _, (times, profiles) = self.simulate(
            model, numpoints=3, varying_inputs=model.inputs
        )
        np.testing.assert_array_equal(times, [0, 0.5, switch, 1])
        self.assert_profile(times, profiles, piecewise_solution(times, profile))

    def test_initial_conditions(self):
        """Explicit initial conditions override values without modifying the model."""
        model = create_model()
        model.x[0].value = None
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        _, (times, profiles) = self.simulate(model, initcon=[1.3])
        self.assert_profile(times, profiles, 1.3 * np.exp(-times))
        self.assertIsNone(model.x[0].value)

    def test_invalid_initial_conditions(self):
        """Reject incorrect dimensions and nonfinite independent initial states."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        sim = Simulator(model, package=self.package)
        for initial in ([], [1, 2], [[1]], [float('nan')], [float('inf')], [1 + 1j]):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                sim.simulate(initcon=initial)

    def test_invalid_integrator(self):
        """An unknown integrator gives a Pyomo error."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        with self.assertRaisesRegex(DAE_Error, 'Unrecognized'):
            self.simulate(model, integrator='missing')

    def test_repeated_simulation(self):
        """Repeated calls read updated mutable values and supplied initial states."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.k * m.x[t])
        sim = Simulator(model, package=self.package)
        for rate, initial in ((2, 1), (0.5, 1.3), (2, 1)):
            with self.subTest(rate=rate, initial=initial):
                model.k = rate
                times, profiles = sim.simulate(
                    numpoints=11, initcon=[initial], integrator_options=self.options()
                )
                self.assert_profile(times, profiles, initial * np.exp(-rate * times))
                self.assertEqual(model.x[0].value, 1)
                self.assertEqual(model.x[1].value, 1)

    def test_changed_input_profiles(self):
        """Reused simulators read changed amplitudes and switch grids."""
        model = create_model()
        model.u = pyo.Var(model.t)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
        model.inputs = pyo.Suffix()
        sim = Simulator(model, package=self.package)
        for profile in ({0: 1, 0.5: 3}, {0: 2, 0.4: -1}, {0: 1, 0.5: 3}):
            with self.subTest(profile=profile):
                model.inputs[model.u] = profile
                times, profiles = sim.simulate(
                    numpoints=13,
                    varying_inputs=model.inputs,
                    integrator_options=self.options(),
                )
                self.assert_profile(times, profiles, piecewise_solution(times, profile))

    def test_options_preserved(self):
        """Caller option dictionaries survive successful and failed calls."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        sim = Simulator(model, package=self.package)
        options = self.options()
        original = deepcopy(options)
        sim.simulate(numpoints=5, integrator_options=options)
        self.assertEqual(options, original)
        invalid = {'invalid_option': {'nested': [1]}}
        original = deepcopy(invalid)
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            sim.simulate(numpoints=5, integrator_options=invalid)
        self.assertEqual(invalid, original)

    def test_empty_input_suffix(self):
        """An empty suffix is a valid no-input simulation."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        model.inputs = pyo.Suffix()
        _, (times, profiles) = self.simulate(model, varying_inputs=model.inputs)
        self.assert_profile(times, profiles, np.exp(-times))

    def test_piecewise_inputs(self):
        """Control switches take effect even for a constant derivative on each phase."""
        for rate in (0, -0.7):
            for start in (0, 2):
                with self.subTest(rate=rate, start=start):
                    model = create_model(start, start + 1)
                    model.u = pyo.Var(model.t)
                    model.ode = pyo.Constraint(
                        model.t, rule=lambda m, t: m.dx[t] == rate * m.x[t] + m.u[t]
                    )
                    model.inputs = pyo.Suffix()
                    profile = {
                        start: 1,
                        start + 0.3: 3,
                        start + 0.65: -0.5,
                        start + 1: 100,
                    }
                    model.inputs[model.u] = profile
                    before = deepcopy(profile)
                    sim, (times, profiles) = self.simulate(
                        model, varying_inputs=model.inputs
                    )
                    self.assert_profile(
                        times, profiles, piecewise_solution(times, profile, rate=rate)
                    )
                    self.assertEqual(profile, before)
                    self.assertEqual(list(sim.get_variable_order('input')), [model.u])

    def test_asynchronous_inputs(self):
        """Different input grids share the union of their switch times."""
        model = create_model()
        model.u = pyo.Var(model.t)
        model.v = pyo.Param(model.t, default=0, mutable=True)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.u[t] + 2 * m.v[t]
        )
        model.inputs = pyo.Suffix()
        u, v = {0: 1, 0.3: 3, 0.7: -1}, {0: 2, 0.4: -1}
        model.inputs[model.u], model.inputs[model.v] = u, v
        _, (times, profiles) = self.simulate(model, varying_inputs=model.inputs)
        expected = piecewise_solution(times, u) + 2 * piecewise_solution(
            times, v, initial=0
        )
        self.assert_profile(times, profiles, expected)
        self.assertTrue({0.3, 0.4, 0.7}.issubset(set(times)))

    def test_time_with_inputs(self):
        """Segment normalization preserves physical time in nonautonomous equations."""
        model = create_model(2, 3)
        model.u = pyo.Var(model.t)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == t + m.u[t])
        model.inputs = pyo.Suffix()
        profile = {2: 1, 2.4: 3}
        model.inputs[model.u] = profile
        _, (times, profiles) = self.simulate(model, varying_inputs=model.inputs)
        self.assert_profile(
            times, profiles, piecewise_solution(times, profile) + (times**2 - 4) / 2
        )

    def test_invalid_input_profiles(self):
        """Missing initial values and nonfinite or out-of-range inputs fail clearly."""
        model = create_model(2, 3)
        model.u = pyo.Var(model.t)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
        model.inputs = pyo.Suffix()
        sim = Simulator(model, package=self.package)
        profiles = (
            {},
            {2.1: 1},
            {1.9: 1, 2: 2},
            {2: 1, 3.1: 2},
            {2: float('nan')},
            {2: 1, float('nan'): 2},
            {2: [1]},
            [],
        )
        for profile in profiles:
            with (
                self.subTest(profile=profile),
                self.assertRaises((ValueError, TypeError)),
            ):
                model.inputs[model.u] = profile
                sim.simulate(varying_inputs=model.inputs)

    def test_model_state_preserved(self):
        """Simulation does not initialize the model or alter fixity and scaling."""
        model = create_model()
        model.x[0].fix(1)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        model.scaling_factor[model.x[0]] = 10
        before = [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)]
        self.simulate(model, initcon=[1.3])
        self.assertEqual(
            [(v.value, v.fixed) for v in model.component_data_objects(pyo.Var)], before
        )
        self.assertTrue(model.ode.active)
        self.assertEqual(model.scaling_factor[model.x[0]], 10)

    def test_initialize_discretized_model(self):
        """Profiles initialize all discretization points, including the endpoint."""
        model = create_model()
        model.x[0].fix()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == 2)
        sim, _ = self.simulate(model)
        pyo.TransformationFactory('dae.collocation').apply_to(model, nfe=4, ncp=3)
        sim.initialize_model()
        for time in model.t:
            self.assertAlmostEqual(model.x[time].value, 1 + 2 * time, delta=2e-7)
        self.assertTrue(model.x[0].fixed)

    def test_failure_preserves_last_simulation(self):
        """Validation and solver failures leave the last successful profile usable."""
        for failure in ('validation', 'solver'):
            with self.subTest(failure=failure):
                model = create_model()
                model.u = pyo.Var(model.t)
                model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
                model.inputs = pyo.Suffix()
                model.inputs[model.u] = {0: 1, 0.5: 3}
                sim, (times, profiles) = self.simulate(
                    model, varying_inputs=model.inputs
                )
                input_order = [id(v) for v in sim.get_variable_order('input')]
                kwds = (
                    {'initcon': []}
                    if failure == 'validation'
                    else {'integrator_options': {'invalid_option': 1}}
                )
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    sim.simulate(varying_inputs=model.inputs, **kwds)
                self.assertEqual(
                    [id(v) for v in sim.get_variable_order('input')], input_order
                )
                sim.initialize_model()
                self.assertAlmostEqual(model.x[1].value, 3, delta=2e-7)
                times, profiles = sim.simulate(
                    numpoints=11,
                    varying_inputs=model.inputs,
                    integrator_options=self.options(),
                )
                self.assert_profile(
                    times, profiles, piecewise_solution(times, model.inputs[model.u])
                )


@unittest.skipUnless(
    numpy_available and scipy_available, 'NumPy and SciPy are required'
)
class TestScipyInterface(SimulatorChecks, unittest.TestCase):
    """Run shared public interface checks through SciPy."""

    package = 'scipy'

    def test_missing_input_profiles_after_simulation(self):
        """Omitted inputs fail consistently without reusing a previous amplitude."""
        model = create_model()
        model.u = pyo.Var(model.t, initialize=99)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
        model.inputs, model.empty = pyo.Suffix(), pyo.Suffix()
        model.inputs[model.u] = {0: 1, 0.5: 3}
        simulator, (times, profiles) = self.simulate(model, varying_inputs=model.inputs)
        prior_inputs, prior_algebraics = (
            simulator._siminputvars,
            simulator._simalgvars,
        )
        for reused in (False, True):
            for inputs in (None, model.empty):
                with self.subTest(reused=reused, omitted=inputs is None):
                    candidate = (
                        simulator if reused else Simulator(model, package='scipy')
                    )
                    with self.assertRaisesRegex(DAE_Error, 'provide values for all'):
                        candidate.simulate(varying_inputs=inputs)
                    if reused:
                        self.assertIs(candidate._siminputvars, prior_inputs)
                        self.assertIs(candidate._simalgvars, prior_algebraics)
                        np.testing.assert_array_equal(candidate._tsim, times)
                        np.testing.assert_array_equal(candidate._simsolution, profiles)
                        candidate.initialize_model()
                        self.assertAlmostEqual(model.x[1].value, 3, delta=2e-7)
        repeated_times, repeated_profile = simulator.simulate(
            numpoints=11, varying_inputs=model.inputs, integrator_options=self.options()
        )
        self.assert_profile(
            repeated_times,
            repeated_profile,
            piecewise_solution(repeated_times, model.inputs[model.u]),
        )
        self.assertEqual(model.u[0].value, 99)
        self.assertEqual(model.u[1].value, 99)

    def test_complex_initial_conditions(self):
        """zvode preserves complex ICs supplied as lists or NumPy arrays."""
        model = create_model()
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.x[t])
        for initial in ([1 + 1j], np.array([1 + 1j])):
            with self.subTest(initial_type=type(initial).__name__):
                _, (times, profiles) = self.simulate(
                    model, integrator='zvode', initcon=initial
                )
                self.assert_profile(times, profiles, (1 + 1j) * np.exp(-times))
                self.assertEqual(model.x[0].value, 1)
        for initial in ([complex(float('nan'), 1)], [complex(1, float('inf'))]):
            with self.assertRaises(ValueError):
                self.simulate(model, integrator='zvode', initcon=initial)

    def test_complex_input_profiles(self):
        """zvode retains finite complex amplitudes through constant and switched inputs."""
        model = create_model(2, 3)
        model.u = pyo.Var(model.t)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == m.u[t])
        model.inputs = pyo.Suffix()
        for profile in ({2: 1j}, {2: 1j, 2.5: 2j}):
            with self.subTest(profile=profile):
                model.inputs[model.u] = profile
                before = deepcopy(profile)
                _, (times, profiles) = self.simulate(
                    model, integrator='zvode', varying_inputs=model.inputs
                )
                self.assert_profile(times, profiles, piecewise_solution(times, profile))
                self.assertEqual(profile, before)
        model.inputs[model.u] = {2: complex(1, float('inf'))}
        with self.assertRaisesRegex(ValueError, 'finite numbers'):
            self.simulate(model, integrator='zvode', varying_inputs=model.inputs)

    def test_piecewise_pendulum_reference(self):
        """Check the existing pendulum example against an independent segmented solve."""
        path = (
            Path(__file__).resolve().parents[3]
            / 'examples'
            / 'dae'
            / 'simulator_ode_multindex_example.py'
        )
        example = import_file(str(path))
        model = example.create_model()
        simulator = Simulator(model, package='scipy')
        times, profiles = simulator.simulate(
            numpoints=100, varying_inputs=model.var_input
        )
        reference = np.empty_like(profiles)
        initial = [0, 3.14 - 0.1]
        for start, end, b, c in (
            (0, 7, 0.25, 5),
            (7, 15, 0.25, 50),
            (15, 20, 0.025, 50),
        ):
            solution = scipy.integrate.solve_ivp(
                lambda t, y: [-b * y[0] - c * np.sin(y[1]), y[0]],
                (start, end),
                initial,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13,
                dense_output=True,
            )
            self.assertTrue(solution.success)
            mask = (times >= start) & (times <= end)
            reference[mask] = solution.sol(times[mask]).T
            initial = solution.y[:, -1]
        np.testing.assert_allclose(profiles, reference, rtol=0, atol=1e-3)


@unittest.skipUnless(
    numpy_available and casadi_available, 'NumPy and CasADi are required'
)
class TestCasadiInterface(SimulatorChecks, unittest.TestCase):
    """Run shared public interface checks and index-one DAEs through CasADi."""

    package = 'casadi'

    def test_linear_dae(self):
        """Differential and consistent algebraic profiles match exact elimination."""
        model = create_model()
        model.z = pyo.Var(model.t, initialize=1.4)
        model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -0.7 * m.z[t])
        model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] == 1.4 * m.x[t])
        sim, (times, profiles) = self.simulate(model)
        x = np.exp(-0.98 * times)
        self.assert_profile(times, profiles, np.column_stack([x, 1.4 * x]))
        self.assertEqual(len(sim.get_variable_order('algebraic')), 1)

    def test_dae_with_inputs(self):
        """Input discontinuities preserve a coupled algebraic state constraint."""
        model = create_model(2, 3)
        model.z = pyo.Var(model.t, initialize=1.4)
        model.u = pyo.Var(model.t)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == -0.7 * m.z[t] + m.u[t]
        )
        model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] == 1.4 * m.x[t])
        model.inputs = pyo.Suffix()
        profile = {2: 1, 2.3: 3, 2.65: -0.5}
        model.inputs[model.u] = profile
        _, (times, profiles) = self.simulate(model, varying_inputs=model.inputs)
        x = piecewise_solution(times, profile, rate=-0.98)
        self.assert_profile(times, profiles, np.column_stack([x, 1.4 * x]))

    def test_nonlinear_algebraic_initial_guess(self):
        """Algebraic guesses select the consistent branch without changing x0."""
        for controlled in (False, True):
            for branch in (-1, 1):
                with self.subTest(controlled=controlled, branch=branch):
                    model = create_model()
                    model.x[0].fix(1)
                    model.z = pyo.Var(model.t, initialize=1.1 * branch)
                    model.alg = pyo.Constraint(
                        model.t, rule=lambda m, t: m.z[t] ** 2 == m.x[t]
                    )
                    kwds = {}
                    if controlled:
                        model.u = pyo.Var(model.t)
                        model.ode = pyo.Constraint(
                            model.t, rule=lambda m, t: m.dx[t] == m.u[t] * m.z[t]
                        )
                        model.inputs = pyo.Suffix()
                        model.inputs[model.u] = {0: 0.2, 0.5: 0.4}
                        kwds['varying_inputs'] = model.inputs
                    else:
                        model.ode = pyo.Constraint(
                            model.t, rule=lambda m, t: m.dx[t] == m.z[t]
                        )
                    _, (times, profiles) = self.simulate(model, **kwds)
                    integral = (
                        piecewise_solution(times, model.inputs[model.u], initial=0)
                        if controlled
                        else times
                    )
                    z = branch + integral / 2
                    self.assert_profile(times, profiles, np.column_stack([z**2, z]))
                    self.assertEqual(model.x[0].value, 1)
                    self.assertEqual(model.z[0].value, 1.1 * branch)
                    self.assertTrue(model.x[0].fixed)

    def test_algebraic_outputs_at_switches(self):
        """Algebraic outputs use new amplitudes at internal and final switches."""
        model = create_model(2, 3)
        model.u = pyo.Var(model.t)
        model.v = pyo.Var(model.t)
        model.z = pyo.Var(model.t)
        model.ode = pyo.Constraint(
            model.t, rule=lambda m, t: m.dx[t] == m.u[t] + 2 * m.v[t]
        )
        model.alg = pyo.Constraint(
            model.t, rule=lambda m, t: m.z[t] == m.x[t] + m.u[t] + m.v[t]
        )
        model.inputs = pyo.Suffix()
        u, v = {2: 1, 2.3: 3, 3: 100}, {2: 2, 2.4: -1}
        model.inputs[model.u], model.inputs[model.v] = u, v
        _, (times, profiles) = self.simulate(model, varying_inputs=model.inputs)
        x = piecewise_solution(times, u) + 2 * piecewise_solution(times, v, initial=0)
        z = x + np.array(
            [
                u[max(s for s in u if s <= t)] + v[max(s for s in v if s <= t)]
                for t in times
            ]
        )
        self.assert_profile(times, profiles, np.column_stack([x, z]))

    def test_nonfinite_algebraic_guess(self):
        """Reject nonfinite guesses before invoking the native solver."""
        for controlled in (False, True):
            with self.subTest(controlled=controlled):
                model = create_model()
                model.z = pyo.Var(model.t)
                model.z[0].set_value(float('nan'), skip_validation=True)
                model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] == m.x[t])
                kwds = {}
                if controlled:
                    model.u = pyo.Var(model.t)
                    model.ode = pyo.Constraint(
                        model.t, rule=lambda m, t: m.dx[t] == m.z[t] + m.u[t]
                    )
                    model.inputs = pyo.Suffix()
                    model.inputs[model.u] = {0: 1}
                    kwds['varying_inputs'] = model.inputs
                else:
                    model.ode = pyo.Constraint(
                        model.t, rule=lambda m, t: m.dx[t] == m.z[t]
                    )
                with self.assertRaisesRegex(ValueError, 'Algebraic initial guesses'):
                    self.simulate(model, **kwds)

    def test_time_dependent_dae(self):
        """Algebraic time dependence uses the physical independent variable."""
        for controlled in (False, True):
            with self.subTest(controlled=controlled):
                model = create_model(2, 3)
                model.z = pyo.Var(model.t)
                model.alg = pyo.Constraint(model.t, rule=lambda m, t: m.z[t] == t)
                model.u = pyo.Var(model.t)
                model.ode = pyo.Constraint(
                    model.t,
                    rule=lambda m, t: m.dx[t] == m.z[t] + (m.u[t] if controlled else 0),
                )
                kwds = {}
                if controlled:
                    model.inputs = pyo.Suffix()
                    model.inputs[model.u] = {2: 1, 2.4: 3}
                    kwds['varying_inputs'] = model.inputs
                _, (times, profiles) = self.simulate(model, **kwds)
                x = 1 + (times**2 - 4) / 2
                if controlled:
                    x += piecewise_solution(times, model.inputs[model.u], initial=0)
                self.assert_profile(times, profiles, np.column_stack([x, times]))


if __name__ == '__main__':
    unittest.main()
