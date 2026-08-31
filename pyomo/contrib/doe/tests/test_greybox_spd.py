# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import logging

from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

import pyomo.environ as pyo

from pyomo.contrib.doe import FIMExternalGreyBox
from pyomo.contrib.doe.examples.grey_box_spd_example import run_indefinite_fim_benchmark


cyipopt_available = pyo.SolverFactory("cyipopt").available()


class _SmallDoEObject:
    """Minimal two-parameter DoE structure for matrix formulation tests."""

    def __init__(self, fim_initial=None):
        self.logger = logging.getLogger(__name__)
        self.model = pyo.ConcreteModel()
        self.model.parameter_names = pyo.Set(initialize=["p1", "p2"], ordered=True)
        self.model.output_names = pyo.Set(initialize=["y[1]", "y[2]"], ordered=True)
        self.model.scenario_blocks = pyo.Block([0])
        scenario = self.model.scenario_blocks[0]
        scenario.y = pyo.Var([1, 2])
        scenario.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        scenario.measurement_error[scenario.y[1]] = 1.0
        scenario.measurement_error[scenario.y[2]] = 2.0

        self.fim_initial = (
            np.asarray(fim_initial, dtype=float)
            if fim_initial is not None
            else np.asarray([[1.0, 2.0], [2.0, 1.0]])
        )
        self.jac_initial = np.asarray([[1.0, 0.5], [0.2, 1.0]])
        self.prior_FIM = 0.1 * np.eye(2)


class _ThreeParameterDoEObject:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = pyo.ConcreteModel()
        self.model.parameter_names = pyo.Set(
            initialize=["p1", "p2", "p3"], ordered=True
        )
        self.model.output_names = pyo.Set(initialize=[], ordered=True)
        self.fim_initial = np.asarray(
            [[1.0, 2.0, 0.0], [2.0, 1.0, 0.2], [0.0, 0.2, 0.5]]
        )
        self.jac_initial = np.empty((0, 3))
        self.prior_FIM = np.zeros((3, 3))


def _value_at(grey_box, inputs):
    grey_box.set_input_values(inputs)
    return grey_box.evaluate_outputs()[0]


def _finite_difference_gradient(grey_box, inputs, step=1e-6):
    gradient = np.zeros(len(inputs))
    for index in range(len(inputs)):
        plus = inputs.copy()
        minus = inputs.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (_value_at(grey_box, plus) - _value_at(grey_box, minus)) / (
            2.0 * step
        )
    grey_box.set_input_values(inputs)
    return gradient


def _finite_difference_hessian(grey_box, inputs, step=2e-4):
    size = len(inputs)
    hessian = np.zeros((size, size))
    center = _value_at(grey_box, inputs)
    for row in range(size):
        plus = inputs.copy()
        minus = inputs.copy()
        plus[row] += step
        minus[row] -= step
        hessian[row, row] = (
            _value_at(grey_box, plus) - 2.0 * center + _value_at(grey_box, minus)
        ) / step**2
        for col in range(row):
            plus_plus = inputs.copy()
            plus_minus = inputs.copy()
            minus_plus = inputs.copy()
            minus_minus = inputs.copy()
            plus_plus[row] += step
            plus_plus[col] += step
            plus_minus[row] += step
            plus_minus[col] -= step
            minus_plus[row] -= step
            minus_plus[col] += step
            minus_minus[row] -= step
            minus_minus[col] -= step
            value = (
                _value_at(grey_box, plus_plus)
                - _value_at(grey_box, plus_minus)
                - _value_at(grey_box, minus_plus)
                + _value_at(grey_box, minus_minus)
            ) / (4.0 * step**2)
            hessian[row, col] = value
            hessian[col, row] = value
    grey_box.set_input_values(inputs)
    return hessian


class TestGreyBoxSPDFormulations(unittest.TestCase):
    def _assert_derivatives(self, formulation, objective):
        grey_box = FIMExternalGreyBox(
            _SmallDoEObject(),
            objective_option=objective,
            fim_formulation=formulation,
            eigenvalue_floor=0.1,
            softplus_beta=10.0,
            softmin_temperature=0.2,
        )
        inputs = grey_box._input_values.copy()
        analytic_gradient = grey_box.evaluate_jacobian_outputs().toarray()[0]
        numerical_gradient = _finite_difference_gradient(grey_box, inputs)
        self.assertTrue(
            np.allclose(analytic_gradient, numerical_gradient, rtol=1e-5, atol=1e-6)
        )

        analytic_hessian = grey_box.evaluate_hessian_outputs().toarray()
        analytic_hessian += analytic_hessian.T - np.diag(np.diag(analytic_hessian))
        numerical_hessian = _finite_difference_hessian(grey_box, inputs)
        self.assertTrue(
            np.allclose(analytic_hessian, numerical_hessian, rtol=2e-4, atol=5e-6)
        )

    def test_sensitivity_formulation_derivatives(self):
        for objective in (
            "determinant",
            "trace",
            "pseudo_trace",
            "minimum_eigenvalue",
            "condition_number",
        ):
            with self.subTest(objective=objective):
                self._assert_derivatives("sensitivity", objective)

    def test_softplus_formulation_derivatives(self):
        for formulation in ("softplus_exact", "softplus_smooth"):
            for objective in (
                "determinant",
                "trace",
                "pseudo_trace",
                "minimum_eigenvalue",
                "condition_number",
            ):
                with self.subTest(formulation=formulation, objective=objective):
                    self._assert_derivatives(formulation, objective)

    def test_indefinite_fim_benchmark(self):
        doe_object = _SmallDoEObject()
        benchmark = FIMExternalGreyBox(
            doe_object, objective_option="determinant", fim_formulation="fim"
        )
        shifted = FIMExternalGreyBox(
            doe_object,
            objective_option="determinant",
            fim_formulation="softplus_exact",
            eigenvalue_floor=0.1,
            softplus_beta=10.0,
        )

        # The existing formulation returns log(abs(det(FIM))) because slogdet's
        # sign is discarded, even though this FIM is indefinite.
        self.assertLess(np.min(np.linalg.eigvalsh(benchmark._get_FIM())), 0.0)
        self.assertAlmostEqual(benchmark.evaluate_outputs()[0], np.log(3.0))

        # The shifted formulation presents a positive-definite matrix to the
        # same objective and derivative implementation.
        self.assertGreater(np.min(np.linalg.eigvalsh(shifted._get_FIM())), 0.1)
        self.assertTrue(np.isfinite(shifted.evaluate_outputs()[0]))

    def test_sensitivity_formulation_reconstructs_information_matrix(self):
        doe_object = _SmallDoEObject()
        grey_box = FIMExternalGreyBox(
            doe_object, objective_option="determinant", fim_formulation="sensitivity"
        )
        weights = np.diag([1.0, 0.25])
        expected = (
            doe_object.jac_initial.T @ weights @ doe_object.jac_initial
            + doe_object.prior_FIM
        )
        self.assertTrue(np.allclose(grey_box._get_FIM(), expected))

    def test_smooth_shift_accepts_repeated_minimum_eigenvalue(self):
        grey_box = FIMExternalGreyBox(
            _SmallDoEObject(fim_initial=-np.eye(2)),
            objective_option="determinant",
            fim_formulation="softplus_smooth",
            eigenvalue_floor=0.1,
            softplus_beta=10.0,
            softmin_temperature=0.2,
        )
        self.assertGreater(np.min(np.linalg.eigvalsh(grey_box._get_FIM())), 0.1)
        self.assertTrue(np.all(np.isfinite(grey_box.evaluate_jacobian_outputs().data)))
        self.assertTrue(np.all(np.isfinite(grey_box.evaluate_hessian_outputs().data)))

        exact_grey_box = FIMExternalGreyBox(
            _SmallDoEObject(fim_initial=-np.eye(2)),
            objective_option="determinant",
            fim_formulation="softplus_exact",
        )
        with self.assertRaisesRegex(ValueError, "repeated minimum eigenvalue"):
            exact_grey_box.evaluate_jacobian_outputs()

    def test_three_parameter_smooth_shift_derivatives(self):
        grey_box = FIMExternalGreyBox(
            _ThreeParameterDoEObject(),
            objective_option="determinant",
            fim_formulation="softplus_smooth",
            eigenvalue_floor=0.1,
            softplus_beta=10.0,
            softmin_temperature=0.2,
        )
        inputs = grey_box._input_values.copy()
        analytic_gradient = grey_box.evaluate_jacobian_outputs().toarray()[0]
        self.assertTrue(
            np.allclose(
                analytic_gradient,
                _finite_difference_gradient(grey_box, inputs),
                rtol=1e-5,
                atol=1e-6,
            )
        )
        analytic_hessian = grey_box.evaluate_hessian_outputs().toarray()
        analytic_hessian += analytic_hessian.T - np.diag(np.diag(analytic_hessian))
        self.assertTrue(
            np.allclose(
                analytic_hessian,
                _finite_difference_hessian(grey_box, inputs),
                rtol=3e-4,
                atol=1e-5,
            )
        )

    @unittest.skipIf(not cyipopt_available, "The 'cyipopt' solver is not available")
    def test_tiny_indefinite_fim_optimization_benchmark(self):
        results = run_indefinite_fim_benchmark()
        self.assertGreater(abs(results["fim"]["design"]), 1.9)
        self.assertLess(abs(results["softplus_smooth"]["design"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
