# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Analytical and solver-backed checks for k_aug Expression measurements."""

import io
import logging
from unittest.mock import patch

from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.common.fileutils import Executable
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.doe import DesignOfExperiments


class ExpressionExperiment:
    """Two algebraic states with mixed variable and Expression measurements."""

    def __init__(self, extended=False):
        """Optionally include nonlinear, constant, and direct-parameter outputs."""
        self.extended = extended

    def get_labeled_model(self):
        """Return a model whose output sensitivities are available analytically."""
        m = pyo.ConcreteModel()
        m.u = pyo.Var(initialize=1.5)
        m.theta = pyo.Var([0, 1], initialize={0: 2.0, 1: 3.0})
        m.theta.fix()
        m.x = pyo.Var([0, 1], initialize={0: 3.0, 1: 6.75})
        m.c1 = pyo.Constraint(expr=m.x[0] == m.theta[0] * m.u)
        m.c2 = pyo.Constraint(expr=m.x[1] == m.theta[1] * m.u**2)
        m.total = pyo.Expression(expr=m.x[0] + m.x[1])
        m.product = pyo.Expression(expr=m.x[0] * m.x[1])
        m.direct = pyo.Expression(expr=m.theta[0] * m.x[0])
        m.constant = pyo.Expression(expr=7.0)
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.u] = None
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((v, pyo.value(v)) for v in m.theta.values())
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        outputs = [m.x[0], m.total]
        if self.extended:
            outputs.extend([m.product, m.direct, m.theta[0], m.u, m.constant])
        for output in outputs:
            m.experiment_outputs[output] = None
            m.measurement_error[output] = 1.0
        return m


def make_doe(extended=False, **kwds):
    """Create an unscaled experiment with an explicitly selected IPOPT solver."""
    return DesignOfExperiments(
        experiment=ExpressionExperiment(extended),
        solver=pyo.SolverFactory("ipopt"),
        objective_option="zero",
        **kwds,
    )


@unittest.skipIf(not (numpy_available and scipy_available), "Requires numpy and scipy")
class TestKaugOutputJacobian(unittest.TestCase):
    """Check chain-rule extraction independently of the external executables."""

    def test_active_inequality_warning_before_solve(self):
        """Warn for all inequality forms, including constraints in nested blocks."""
        for form in ("lower", "upper", "ranged", "nested"):
            with self.subTest(form=form):
                doe = make_doe()
                model = doe.experiment.get_labeled_model()
                if form == "lower":
                    model.limit = pyo.Constraint(expr=model.x[0] >= 0)
                elif form == "upper":
                    model.limit = pyo.Constraint(expr=model.x[0] <= 10)
                elif form == "ranged":
                    model.limit = pyo.Constraint(expr=(0, model.x[0], 10))
                else:
                    model.block = pyo.Block()
                    model.block.limit = pyo.Constraint(
                        [0, 1], rule=lambda b, i: model.x[i] <= 10
                    )
                output = io.StringIO()
                # A failing nominal solve must not hide the diagnostic. Stopping
                # here makes this a solver-independent test of the public path.
                with LoggingIntercept(output, "pyomo", logging.WARNING):
                    with patch.object(
                        doe.solver, "solve", side_effect=RuntimeError("nominal failure")
                    ) as solve:
                        with self.assertRaisesRegex(RuntimeError, "nominal failure"):
                            doe.compute_FIM(model=model, method="kaug")
                solve.assert_called_once()
                warning = output.getvalue()
                self.assertIn("active inequality constraint(s)", warning)
                self.assertIn("method='sequential'", warning)
                self.assertIn("explicit slack variables", warning)
                self.assertEqual(warning.count("Computing the FIM with k_aug"), 1)
                first = "block.limit[0]" if form == "nested" else "limit"
                self.assertIn("first: '%s'" % first, warning)
                self.assertTrue(model.find_component(first).active)

    def test_no_warning_for_equalities_bounds_or_inactive_constraints(self):
        """Only active inequality rows, not variable bounds, trigger the warning."""
        doe = make_doe()
        model = doe.experiment.get_labeled_model()
        model.x[0].setlb(0)
        model.x[0].setub(10)
        model.limit = pyo.Constraint(expr=model.x[0] <= 10)
        model.limit.deactivate()
        model.block = pyo.Block()
        model.block.limit = pyo.Constraint(expr=model.x[1] >= 0)
        model.block.deactivate()
        output = io.StringIO()
        with LoggingIntercept(output, "pyomo", logging.WARNING):
            with patch.object(
                doe.solver, "solve", side_effect=RuntimeError("nominal failure")
            ) as solve:
                with self.assertRaisesRegex(RuntimeError, "nominal failure"):
                    doe.compute_FIM(model=model, method="kaug")
        solve.assert_called_once()
        self.assertEqual(output.getvalue(), "")

    def test_mixed_outputs(self):
        """Include direct parameters, fixed variables, and nonlinear Expressions."""
        doe = make_doe(extended=True)
        model = doe.experiment.get_labeled_model()
        model.u.fix()
        # Reversed column order verifies mapping through the supplied NL names.
        jac = doe._extract_kaug_output_jacobian(
            model, [[0.0, 2.25], [1.5, 0.0]], ["x[1]", "x[0]"]
        )
        expected = [
            [1.5, 0.0],
            [1.5, 2.25],
            [10.125, 6.75],
            [6.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        np.testing.assert_allclose(jac, expected)

    def test_missing_unfixed_dependency_raises(self):
        """Do not silently zero a sensitivity when an NL column is missing."""
        doe = make_doe()
        model = doe.experiment.get_labeled_model()
        with self.assertRaisesRegex(ValueError, r"unfixed variable 'x\[1\]'"):
            doe._extract_kaug_output_jacobian(model, [[1.5, 0.0]], ["x[0]"])

    def test_fixed_dependency_has_zero_sensitivity(self):
        """An ordinary fixed state can be absent from the sensitivity columns."""
        doe = make_doe()
        model = doe.experiment.get_labeled_model()
        model.x[1].fix()
        jac = doe._extract_kaug_output_jacobian(model, [[1.5, 0.0]], ["x[0]"])
        np.testing.assert_allclose(jac, [[1.5, 0.0], [1.5, 0.0]])

    def test_mutable_parameter_dependency(self):
        """Direct mutable parameter dependence contributes an identity derivative."""
        doe = make_doe()
        model = doe.experiment.get_labeled_model()
        model.p = pyo.Param(initialize=5.0, mutable=True)
        model.unknown_parameters.clear()
        model.unknown_parameters[model.p] = 5.0
        model.expression = pyo.Expression(expr=model.p**2)
        model.experiment_outputs.clear()
        model.experiment_outputs[model.expression] = None
        jac = doe._extract_kaug_output_jacobian(model, np.empty((0, 1)), [])
        np.testing.assert_allclose(jac, [[10.0]])

    def test_invalid_sensitivity_dimensions(self):
        """Reject a matrix that does not match the NL columns and parameters."""
        doe = make_doe()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            doe._extract_kaug_output_jacobian(
                doe.experiment.get_labeled_model(), np.zeros((1, 1)), ["x[0]"]
            )

    @unittest.skipUnless(
        all(Executable(name).available() for name in ("ipopt", "k_aug", "dot_sens")),
        "Requires ipopt, k_aug, and dot_sens",
    )
    def test_kaug_matches_sequential_and_analytical_fim(self):
        """Exercise the real k_aug pipeline with scaling, weighting, and prior."""
        for extended in (False, True):
            for scaled in (False, True):
                with self.subTest(extended=extended, scaled=scaled):
                    prior = np.array([[2.0, 0.25], [0.25, 1.0]])
                    doe = make_doe(
                        extended,
                        scale_nominal_param_value=scaled,
                        scale_constant_value=2.0,
                        prior_FIM=prior,
                    )
                    model = doe.experiment.get_labeled_model()
                    for i, output in enumerate(model.measurement_error):
                        model.measurement_error[output] = i + 1.0
                    seq_fim = doe.compute_FIM(model=model.clone(), method="sequential")
                    kaug_fim = doe.compute_FIM(model=model.clone(), method="kaug")
                    jac = np.array(
                        [
                            [1.5, 0.0],
                            [1.5, 2.25],
                            [10.125, 6.75],
                            [6.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ]
                    )
                    if not extended:
                        jac = jac[:2]
                    jac *= 2.0
                    if scaled:
                        jac *= [2.0, 3.0]
                    sigma = np.arange(1, len(jac) + 1)
                    expected = (jac / sigma[:, None]).T @ (jac / sigma[:, None]) + prior
                    np.testing.assert_allclose(doe.kaug_jac, jac, rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(kaug_fim, expected, rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(kaug_fim, seq_fim, rtol=1e-5, atol=1e-6)
