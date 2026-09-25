# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Generic tests for FIM/J GreyBox inputs and native log-E derivatives."""
import logging
from types import SimpleNamespace

from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest('These tests require numpy and scipy')

import pyomo.environ as pyo
from pyomo.contrib.doe import (
    DesignOfExperiments,
    FIMExternalGreyBox,
    GreyBoxFIMFormulation,
)
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.doe.examples.grey_box_e_optimality import (
    LinearResponseExperiment as LinearExperiment,
)


def small_doe():
    m = pyo.ConcreteModel()
    m.parameter_names = pyo.Set(initialize=['p2', 'p1'], ordered=True)
    m.output_names = pyo.Set(initialize=['y[2]', 'y[0]', 'y[1]'], ordered=True)
    m.scenario_blocks = pyo.Block([0])
    b = m.scenario_blocks[0]
    b.y = pyo.Var([0, 1, 2])
    b.measurement_error = pyo.Suffix()
    for name, sigma in zip(m.output_names, [1.0, 2.0, 0.5]):
        b.measurement_error[b.find_component(name)] = sigma
    j = np.array([[1.0, 0.5], [0.2, 1.0], [0.7, -0.3]])
    prior = np.array([[0.8, 0.1], [0.1, 0.5]])
    fim = j.T @ (np.array([1.0, 0.25, 4.0])[:, None] * j) + prior
    return SimpleNamespace(
        model=m,
        logger=logging.getLogger(__name__),
        jac_initial=j,
        prior_FIM=prior,
        fim_initial=fim,
    )


def dense_hessian(grey):
    lower = grey.evaluate_hessian_outputs().toarray()
    return lower + lower.T - np.diag(np.diag(lower))


def finite_differences(value, x):
    gradient = np.empty(len(x))
    hessian = np.empty((len(x), len(x)))
    eye = np.eye(len(x))
    for i, direction in enumerate(eye):
        gradient[i] = (value(x + 1e-6 * direction) - value(x - 1e-6 * direction)) / 2e-6
        for k, other in enumerate(eye):
            h = 1e-4
            hessian[i, k] = (
                value(x + h * direction + h * other)
                - value(x + h * direction - h * other)
                - value(x - h * direction + h * other)
                + value(x - h * direction - h * other)
            ) / (4 * h * h)
    return gradient, hessian


class TestSensitivityGreyBox(unittest.TestCase):
    def test_values_and_exact_derivatives(self):
        doe = small_doe()
        weights = np.array([1.0, 0.25, 4.0])
        for formulation in ('fim', 'sensitivity'):
            for objective in (
                'trace',
                'pseudo_trace',
                'determinant',
                'minimum_eigenvalue',
                'log_minimum_eigenvalue',
                'condition_number',
            ):
                with self.subTest(formulation=formulation, objective=objective):
                    grey = FIMExternalGreyBox(
                        doe,
                        objective,
                        fim_formulation=formulation,
                        eigenvalue_reference=2.5,
                    )
                    x = grey._input_values.copy()

                    def value(inputs):
                        if formulation == 'sensitivity':
                            j = inputs.reshape(3, 2)
                            fim = j.T @ (weights[:, None] * j) + doe.prior_FIM
                        else:
                            fim = np.array(
                                [[inputs[0], inputs[1]], [inputs[1], inputs[2]]]
                            )
                        eig = np.linalg.eigvalsh(fim)
                        return {
                            'trace': lambda: np.trace(np.linalg.inv(fim)),
                            'pseudo_trace': lambda: np.trace(fim),
                            'determinant': lambda: np.log(np.linalg.det(fim)),
                            'minimum_eigenvalue': lambda: eig[0],
                            'log_minimum_eigenvalue': lambda: np.log(eig[0] / 2.5),
                            'condition_number': lambda: np.log(eig[-1] / eig[0]),
                        }[objective]()

                    self.assertAlmostEqual(grey.evaluate_outputs()[0], value(x))
                    gradient, hessian = finite_differences(value, x)
                    np.testing.assert_allclose(
                        grey.evaluate_jacobian_outputs().toarray()[0],
                        gradient,
                        atol=1e-7,
                        rtol=1e-6,
                    )
                    for multiplier in (0.0, -2.5, 1.0, 3.0):
                        grey.set_output_constraint_multipliers([multiplier])
                        np.testing.assert_allclose(
                            dense_hessian(grey),
                            multiplier * hessian,
                            atol=3e-6,
                            rtol=2e-5,
                        )
                    expected_names = (
                        [
                            ('y[2]', 'p2'),
                            ('y[2]', 'p1'),
                            ('y[0]', 'p2'),
                            ('y[0]', 'p1'),
                            ('y[1]', 'p2'),
                            ('y[1]', 'p1'),
                        ]
                        if formulation == 'sensitivity'
                        else [('p2', 'p2'), ('p2', 'p1'), ('p1', 'p1')]
                    )
                    self.assertEqual(grey.input_names(), expected_names)

    def test_zero_jacobian_retains_second_chain_rule_term_and_sparsity(self):
        doe = small_doe()
        grey = FIMExternalGreyBox(
            doe, 'log_minimum_eigenvalue', fim_formulation='sensitivity'
        )
        before = grey.evaluate_hessian_outputs()
        grey.set_input_values(np.zeros(6))
        after = grey.evaluate_hessian_outputs()
        np.testing.assert_array_equal(before.row, after.row)
        np.testing.assert_array_equal(before.col, after.col)
        np.testing.assert_allclose(grey.evaluate_jacobian_outputs().toarray(), 0)
        eigenvalues, eigenvectors = np.linalg.eigh(doe.prior_FIM)
        g = np.outer(eigenvectors[:, 0], eigenvectors[:, 0]) / eigenvalues[0]
        expected = np.kron(np.diag([2.0, 0.5, 8.0]), g)
        np.testing.assert_allclose(dense_hessian(grey), expected, atol=1e-14)

    def test_reference_changes_only_value(self):
        doe = small_doe()
        for formulation in ('fim', 'sensitivity'):
            a = FIMExternalGreyBox(
                doe, 'log_minimum_eigenvalue', fim_formulation=formulation
            )
            b = FIMExternalGreyBox(
                doe,
                'log_minimum_eigenvalue',
                fim_formulation=formulation,
                eigenvalue_reference=10.0,
            )
            self.assertAlmostEqual(
                a.evaluate_outputs()[0] - b.evaluate_outputs()[0], np.log(10.0)
            )
            np.testing.assert_allclose(
                a.evaluate_jacobian_outputs().toarray(),
                b.evaluate_jacobian_outputs().toarray(),
            )
            np.testing.assert_allclose(dense_hessian(a), dense_hessian(b))

    def test_log_domain_and_repeated_eigenvalue(self):
        for diagonal in ([0.0, 1.0], [-1.0, 2.0]):
            doe = small_doe()
            doe.fim_initial = np.diag(diagonal)
            grey = FIMExternalGreyBox(doe, 'log_minimum_eigenvalue')
            for callback in (
                grey.evaluate_outputs,
                grey.evaluate_jacobian_outputs,
                grey.evaluate_hessian_outputs,
            ):
                with self.assertRaisesRegex(ValueError, 'positive-definite'):
                    callback()
        doe = small_doe()
        doe.fim_initial = np.eye(2)
        grey = FIMExternalGreyBox(doe, 'log_minimum_eigenvalue')
        self.assertEqual(grey.evaluate_outputs()[0], 0.0)
        for callback in (grey.evaluate_jacobian_outputs, grey.evaluate_hessian_outputs):
            with self.assertRaisesRegex(ValueError, 'simple minimum eigenvalue'):
                callback()

    def test_singular_prior_needs_information_from_j(self):
        doe = small_doe()
        doe.prior_FIM = np.zeros((2, 2))
        grey = FIMExternalGreyBox(
            doe, 'log_minimum_eigenvalue', fim_formulation='sensitivity'
        )
        self.assertTrue(np.isfinite(grey.evaluate_outputs()[0]))
        grey.set_input_values(np.zeros(6))
        with self.assertRaisesRegex(ValueError, 'positive-definite'):
            grey.evaluate_outputs()

    def test_configuration_errors(self):
        for reference in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(ValueError, 'finite and positive'):
                    FIMExternalGreyBox(small_doe(), eigenvalue_reference=reference)
                with self.assertRaisesRegex(ValueError, 'finite and positive'):
                    DesignOfExperiments(
                        experiment=LinearExperiment(),
                        grey_box_eigenvalue_reference=reference,
                    )
        for options in (
            {'grey_box_fim_formulation': 'sensitivity'},
            {'objective_option': 'log_minimum_eigenvalue'},
        ):
            with self.assertRaisesRegex(ValueError, 'use_grey_box_objective=True'):
                DesignOfExperiments(experiment=LinearExperiment(), **options)
        with self.assertRaises(ValueError):
            FIMExternalGreyBox(small_doe(), fim_formulation='unknown')
        doe = small_doe()
        doe.prior_FIM = np.diag([-1.0, 2.0])
        with self.assertRaisesRegex(ValueError, 'positive-semidefinite'):
            FIMExternalGreyBox(doe, fim_formulation='sensitivity')
        for error in (0.0, -1.0, float('nan'), float('inf')):
            doe = small_doe()
            scenario = doe.model.scenario_blocks[0]
            scenario.measurement_error[scenario.y[2]] = error
            with self.assertRaisesRegex(
                ValueError, 'finite positive measurement errors'
            ):
                FIMExternalGreyBox(doe, fim_formulation='sensitivity')
        doe = small_doe()
        doe.jac_initial = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'jac_initial has shape'):
            FIMExternalGreyBox(doe, fim_formulation='sensitivity')


@unittest.skipUnless(
    pyo.SolverFactory('ipopt').available(False)
    and pyo.SolverFactory('cyipopt').available(False)
    and AmplInterface.available(),
    'Requires IPOPT, CyIpopt, and PyNumero ASL',
)
class TestSensitivityGreyBoxSolve(unittest.TestCase):
    def test_raw_and_log_e_both_input_formulations(self):
        prior = np.array([[1.0, 0.1], [0.1, 2.0]])
        for formulation in GreyBoxFIMFormulation:
            for objective in ('minimum_eigenvalue', 'log_minimum_eigenvalue'):
                for scaled in (False, True):
                    with self.subTest(
                        formulation=formulation, objective=objective, scaled=scaled
                    ):
                        solver = pyo.SolverFactory('ipopt')
                        grey_solver = pyo.SolverFactory('cyipopt')
                        grey_solver.config.options['tol'] = 1e-8
                        doe = DesignOfExperiments(
                            experiment=LinearExperiment(),
                            solver=solver,
                            grey_box_solver=grey_solver,
                            use_grey_box_objective=True,
                            grey_box_fim_formulation=formulation,
                            objective_option=objective,
                            scale_nominal_param_value=scaled,
                            prior_FIM=prior,
                            grey_box_eigenvalue_reference=2.0,
                        )
                        doe.run_doe()
                        self.assertEqual(
                            str(doe.results['Termination Condition']), 'optimal'
                        )
                        self.assertAlmostEqual(
                            doe.get_experiment_input_values()[0], 2.0, places=5
                        )
                        j = np.diag([2.0, 3.0])
                        if scaled:
                            j *= np.array([1.2, 0.8])
                        expected = j.T @ np.diag([1.0, 0.25]) @ j + prior
                        np.testing.assert_allclose(doe.get_FIM(), expected, atol=1e-6)
                        eig = np.linalg.eigvalsh(expected)[0]
                        expected_value = (
                            np.log(eig / 2.0)
                            if objective == 'log_minimum_eigenvalue'
                            else eig
                        )
                        self.assertAlmostEqual(
                            pyo.value(doe.model.objective), expected_value, places=5
                        )
                        self.assertEqual(
                            doe.results['GreyBox FIM Formulation'], formulation.value
                        )
                        if formulation == GreyBoxFIMFormulation.sensitivity:
                            self.assertFalse(doe.model.fim_constraint.active)
                            # Reporting must follow current J rather than stale lifted FIM values.
                            for v in doe.model.sensitivity_jacobian.values():
                                v.set_value(2 * pyo.value(v))
                            np.testing.assert_allclose(
                                doe.get_FIM(), 4 * (expected - prior) + prior, atol=4e-6
                            )
                        else:
                            self.assertTrue(doe.model.fim_constraint.active)
