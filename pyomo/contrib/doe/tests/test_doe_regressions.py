# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import json
import logging
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition


class LinearExperiment:
    """Two identifiable parameters, including a negative nominal value."""

    def __init__(self, bounded=False):
        self.bounded = bounded
        self.model = None

    def get_labeled_model(self):
        if self.model is not None:
            return self.model
        m = pyo.ConcreteModel()
        m.design = pyo.Var(initialize=2, bounds=(1, 3))
        m.theta = pyo.Var([0, 1], initialize={0: 3, 1: -4})
        m.theta.fix()
        if self.bounded:
            m.theta[0].setlb(3)
            m.theta[0].setub(6)
            m.theta[1].setlb(-8)
            m.theta[1].setub(-4)
        m.y = pyo.Var(initialize=0)
        m.z = pyo.Var(initialize=0)
        m.response_y = pyo.Constraint(expr=m.y == m.design * m.theta[0] + m.theta[1])
        m.response_z = pyo.Constraint(expr=m.z == m.theta[0] - 2 * m.theta[1])
        # Must be evaluated before the perturbed parameter is reset.
        m.expression_output = pyo.Expression(expr=m.theta[0] + 3 * m.theta[1])
        m.sigma = pyo.Param(initialize=1.5, mutable=True)
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.design] = None
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((v, pyo.value(v)) for v in m.theta.values())
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for output, sigma in ((m.y, 0.5), (m.z, 2.0), (m.expression_output, 1.5)):
            m.experiment_outputs[output] = None
            m.measurement_error[output] = sigma
        self.model = m
        return m


class ExactLinearSolver:
    """Record each fixed-design solve and evaluate the linear response exactly."""

    def __init__(self, fail_on=None, raise_error=False):
        self.calls = []
        self.bounds = []
        self.fail_on = fail_on
        self.raise_error = raise_error

    def solve(self, model, **kwds):
        assert model.design.fixed
        assert all(v.fixed for v in model.theta.values())
        theta = [pyo.value(v) for v in model.theta.values()]
        self.calls.append(theta)
        self.bounds.append([v.bounds for v in model.theta.values()])
        model.y.set_value(pyo.value(model.design) * theta[0] + theta[1])
        model.z.set_value(theta[0] - 2 * theta[1])
        result = SolverResults()
        result.solver.status = SolverStatus.ok
        result.solver.termination_condition = TerminationCondition.optimal
        if len(self.calls) == self.fail_on:
            if self.raise_error:
                raise ValueError("test solver failure")
            result.solver.termination_condition = TerminationCondition.infeasible
        return result


def make_doe(formula="central", bounded=False, scaled=False, solver=None):
    return DesignOfExperiments(
        experiment=LinearExperiment(bounded),
        solver=solver if solver is not None else ExactLinearSolver(),
        fd_formula=formula,
        step=0.01,
        scale_nominal_param_value=scaled,
        objective_option="zero",
    )


@unittest.skipIf(not (numpy_available and scipy_available), "Requires numpy and scipy")
class TestDoEFiniteDifferenceAndReporting(unittest.TestCase):
    def test_sequential_formulas(self):
        expected_calls = {
            "central": [[3, -4], [3.03, -4], [2.97, -4], [3, -4.04], [3, -3.96]],
            "forward": [[3, -4], [3.03, -4], [3, -4.04]],
            "backward": [[3, -4], [2.97, -4], [3, -3.96]],
        }
        for formula in expected_calls:
            for scaled in (False, True):
                with self.subTest(formula=formula, scaled=scaled):
                    doe = make_doe(formula, scaled=scaled)
                    fim = doe.compute_FIM()
                    jac = np.array([[2.0, 1.0], [1.0, -2.0], [1.0, 3.0]])
                    if scaled:
                        jac *= [3, -4]
                    np.testing.assert_allclose(
                        doe.solver.calls, expected_calls[formula]
                    )
                    np.testing.assert_allclose(doe.seq_jac, jac)
                    np.testing.assert_allclose(
                        fim, jac.T @ np.diag([4, 0.25, 1 / 1.5**2]) @ jac
                    )
                    np.testing.assert_allclose(
                        [pyo.value(v) for v in doe.compute_FIM_model.theta.values()],
                        [3, -4],
                    )

    def test_bounds_sequential_and_simultaneous(self):
        for formula in ("central", "forward", "backward"):
            for simultaneous in (False, True):
                with self.subTest(formula=formula, simultaneous=simultaneous):
                    doe = make_doe(formula, bounded=True)
                    output = StringIO()
                    with LoggingIntercept(output, "pyomo", logging.WARNING):
                        if simultaneous:
                            doe.create_doe_model()
                        else:
                            fim = doe.compute_FIM()
                    # Forward perturbations move both signed parameters inward.
                    warning_count = 0 if formula == "forward" else 2
                    self.assertEqual(
                        output.getvalue().count("Widening the bounds"), warning_count
                    )
                    self.assertNotIn("W1002", output.getvalue())
                    for values, bounds in zip(doe.solver.calls, doe.solver.bounds):
                        for value, (lb, ub) in zip(values, bounds):
                            self.assertGreaterEqual(value, lb)
                            self.assertLessEqual(value, ub)
                    if simultaneous:
                        for block in doe.model.scenario_blocks.values():
                            for param in block.theta.values():
                                self.assertGreaterEqual(pyo.value(param), param.lb)
                                self.assertLessEqual(pyo.value(param), param.ub)
                    else:
                        self.assertEqual(doe.compute_FIM_model.theta[0].bounds, (3, 6))
                        self.assertEqual(
                            doe.compute_FIM_model.theta[1].bounds, (-8, -4)
                        )
                        np.testing.assert_allclose(fim, make_doe(formula).compute_FIM())
                    # The experiment's source model is not mutated.
                    source = doe.experiment.get_labeled_model()
                    self.assertEqual(source.theta[0].bounds, (3, 6))

    def test_one_sided_and_interior_bounds(self):
        bound_cases = (
            ((None, 3), (-4, None)),
            ((3, None), (None, -4)),
            ((None, None), (None, None)),
            ((0, 6), (-8, 0)),
            ((3, 3), (-4, -4)),
        )
        for formula in ("central", "forward", "backward"):
            for bounds in bound_cases:
                for simultaneous in (False, True):
                    with self.subTest(
                        formula=formula, bounds=bounds, simultaneous=simultaneous
                    ):
                        doe = make_doe(formula)
                        source = doe.experiment.get_labeled_model()
                        for param, (lb, ub) in zip(source.theta.values(), bounds):
                            param.setlb(lb)
                            param.setub(ub)
                        output = StringIO()
                        with LoggingIntercept(output, "pyomo", logging.WARNING):
                            if simultaneous:
                                doe.create_doe_model()
                            else:
                                doe.compute_FIM()
                        self.assertNotIn("W1002", output.getvalue())
                        for values, solve_bounds in zip(
                            doe.solver.calls, doe.solver.bounds
                        ):
                            for value, (lb, ub) in zip(values, solve_bounds):
                                if lb is not None:
                                    self.assertGreaterEqual(value, lb)
                                if ub is not None:
                                    self.assertLessEqual(value, ub)
                        self.assertEqual(
                            tuple(v.bounds for v in source.theta.values()), bounds
                        )
                        if not simultaneous:
                            self.assertEqual(
                                tuple(
                                    v.bounds
                                    for v in doe.compute_FIM_model.theta.values()
                                ),
                                bounds,
                            )

    def test_sequential_preserves_bound_expressions(self):
        doe = make_doe()
        model = doe.experiment.get_labeled_model()
        model.lower = pyo.Param(initialize=3, mutable=True)
        model.theta[0].setlb(model.lower)
        doe.compute_FIM(model=model)
        self.assertIs(model.theta[0].lower, model.lower)
        model.lower.set_value(2)
        self.assertEqual(model.theta[0].lb, 2)

    def test_sequential_failure_restores_parameters_and_bounds(self):
        for formula in ("central", "forward", "backward"):
            for fail_on in (1, 2):
                for raise_error in (False, True):
                    with self.subTest(
                        formula=formula, fail_on=fail_on, raise_error=raise_error
                    ):
                        solver = ExactLinearSolver(fail_on, raise_error)
                        doe = make_doe(formula, bounded=True, solver=solver)
                        with self.assertRaisesRegex(RuntimeError, "nominal|scenario"):
                            doe.compute_FIM()
                        model = doe.compute_FIM_model
                        np.testing.assert_allclose(
                            [pyo.value(v) for v in model.theta.values()], [3, -4]
                        )
                        self.assertEqual(model.theta[0].bounds, (3, 6))
                        self.assertEqual(model.theta[1].bounds, (-8, -4))
                        self.assertTrue(all(v.fixed for v in model.theta.values()))

    def test_measurement_errors_flat_and_scenario_models(self):
        doe = make_doe()
        model = doe.experiment.get_labeled_model()
        model.measurement_error[model.expression_output] = model.sigma
        self.assertEqual(doe.get_measurement_error_values(model), [0.5, 2.0, 1.5])
        doe.model.scenario_blocks = pyo.Block([0])
        doe.model.scenario_blocks[0].transfer_attributes_from(model.clone())
        self.assertEqual(doe.get_measurement_error_values(), [0.5, 2.0, 1.5])

    @unittest.skipIf(not pyo.SolverFactory("ipopt").available(), "Requires ipopt")
    def test_results_file_path_and_string(self):
        with TemporaryDirectory() as directory:
            for path_type in (Path, str):
                with self.subTest(path_type=path_type):
                    filename = path_type(Path(directory) / "results.json")
                    doe = make_doe(solver=pyo.SolverFactory("ipopt"))
                    doe.run_doe(results_file=filename)
                    with open(filename) as stream:
                        results = json.load(stream)
                    self.assertEqual(results["Measurement Error"], [0.5, 2.0, 1.5])
                    self.assertEqual(results["FIM"], doe.results["FIM"])
                    self.assertEqual(results["Termination Condition"], "optimal")
