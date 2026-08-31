# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Small benchmark exposing an indefinite lifted-FIM objective.

The one-variable model uses ``FIM(x) = [[1, x], [x, 1]]``. With the current
lifted formulation, D-optimality evaluates ``log(abs(det(FIM)))`` and therefore
prefers ``|x| = 2``, where the matrix is indefinite. The penalized smooth-shift
formulation instead converges near ``x = 0``, where no corrective shift is
needed. This deliberately artificial example isolates the matrix-domain issue
without a dynamic model or finite-difference parameter sensitivities.
"""
import logging

from pyomo.common.dependencies import numpy as np
import pyomo.environ as pyo

from pyomo.contrib.doe import FIMExternalGreyBox
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock


class _TwoParameterDoE:
    def __init__(self, initial_design=1.5):
        self.logger = logging.getLogger(__name__)
        self.model = pyo.ConcreteModel()
        self.model.parameter_names = pyo.Set(initialize=["p1", "p2"], ordered=True)
        self.model.output_names = pyo.Set(initialize=[], ordered=True)
        self.fim_initial = np.asarray([[1.0, initial_design], [initial_design, 1.0]])
        self.jac_initial = np.empty((0, 2))
        self.prior_FIM = np.zeros((2, 2))


def build_indefinite_fim_benchmark(
    fim_formulation="fim", initial_design=1.5, shift_penalty=1.0
):
    """Build the tiny lifted-FIM D-optimality benchmark."""
    doe_object = _TwoParameterDoE(initial_design=initial_design)
    external_model = FIMExternalGreyBox(
        doe_object,
        objective_option="determinant",
        fim_formulation=fim_formulation,
        eigenvalue_floor=0.1,
        softplus_beta=10.0,
        softmin_temperature=0.2,
        shift_penalty=shift_penalty,
    )

    model = pyo.ConcreteModel()
    model.design = pyo.Var(initialize=initial_design, bounds=(-2.0, 2.0))
    model.grey_box = ExternalGreyBoxBlock(external_model=external_model)
    model.fim_11 = pyo.Constraint(expr=model.grey_box.inputs[("p1", "p1")] == 1.0)
    model.fim_12 = pyo.Constraint(
        expr=model.grey_box.inputs[("p1", "p2")] == model.design
    )
    model.fim_22 = pyo.Constraint(expr=model.grey_box.inputs[("p2", "p2")] == 1.0)
    model.objective = pyo.Objective(
        expr=model.grey_box.outputs["log-D-opt"], sense=pyo.maximize
    )
    return model


def run_indefinite_fim_benchmark():
    """Solve and return the current and smooth-shift benchmark designs."""
    results = {}
    for formulation in ("fim", "softplus_smooth"):
        model = build_indefinite_fim_benchmark(fim_formulation=formulation)
        solver = pyo.SolverFactory("cyipopt")
        solver.config.options["tol"] = 1e-8
        solver.config.options["max_iter"] = 100
        solver.solve(model)
        results[formulation] = {
            "design": pyo.value(model.design),
            "objective": pyo.value(model.objective),
        }
    return results


if __name__ == "__main__":
    for name, values in run_indefinite_fim_benchmark().items():
        print(
            "%s: design=%+.6f, objective=%+.6f"
            % (name, values["design"], values["objective"])
        )
