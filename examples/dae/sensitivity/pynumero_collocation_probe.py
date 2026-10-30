# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tiny discrete sensitivity example; no PSA model or integrator is used.

Requires Pyomo, the PyNumero ASL extension, NumPy and SciPy. A selected model
parameter is represented as a free Var to retain its Jacobian column. The
state equations are solved at a specified parameter; this is not an NLP solve.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyomo
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.dae import ContinuousSet, DerivativeVar
from scipy.sparse.linalg import spsolve


def check_mesh(elements):
    """Compare implicit collocation derivatives with analytic and discrete FD values."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 2))
    model.k = pyo.Var(initialize=0.7)
    model.x = pyo.Var(model.t, initialize=1.2)
    model.dx = DerivativeVar(model.x, wrt=model.t)
    model.x[0].fix(1.2)

    def ode(m, t):
        """Define exponential decay away from the fixed initial condition."""
        if t == 0:
            return pyo.Constraint.Skip
        return m.dx[t] == -m.k * m.x[t]

    model.ode = pyo.Constraint(model.t, rule=ode)
    model.obj = pyo.Objective(expr=0.0)
    pyo.TransformationFactory("dae.collocation").apply_to(
        model, wrt=model.t, nfe=elements, ncp=3, scheme="LAGRANGE-RADAU"
    )
    for t in model.t:
        model.x[t].set_value(1.2 * math.exp(-0.7 * t))
        model.dx[t].set_value(-0.7 * 1.2 * math.exp(-0.7 * t))

    nlp = PyomoNLP(model)
    variables = nlp.get_pyomo_variables()
    parameter_column = next(i for i, v in enumerate(variables) if v is model.k)
    state_columns = np.array(
        [i for i in range(len(variables)) if i != parameter_column]
    )
    equality_objects = nlp.get_pyomo_equality_constraints()
    lower = nlp.constraints_lb()[nlp.get_constraint_indices(equality_objects)]
    initial = nlp.get_primals().copy()

    def solve_at(parameter):
        """Solve the linear state equations at a specified parameter value."""
        point = initial.copy()
        point[parameter_column] = parameter
        nlp.set_primals(point)
        # With k fixed, these equations are linear in all state/derivative
        # variables. The second update checks residual numerical roundoff.
        for _ in range(2):
            jacobian = nlp.evaluate_jacobian_eq().tocsr()
            residual = nlp.evaluate_eq_constraints() - lower
            point[state_columns] -= spsolve(
                jacobian[:, state_columns].tocsc(), residual
            )
            nlp.set_primals(point)
        error = float(np.max(abs(nlp.evaluate_eq_constraints() - lower)))
        return point.copy(), error

    point, error = solve_at(0.7)
    nlp.set_primals(point)
    jacobian = nlp.evaluate_jacobian_eq().tocsr()
    sensitivity = -spsolve(
        jacobian[:, state_columns].tocsc(),
        jacobian[:, parameter_column].toarray().ravel(),
    )
    endpoint_column = next(i for i, v in enumerate(variables) if v is model.x[2])
    endpoint_position = int(np.flatnonzero(state_columns == endpoint_column)[0])
    plus, _ = solve_at(0.7 + 1e-5)
    minus, _ = solve_at(0.7 - 1e-5)
    finite_difference = (plus[endpoint_column] - minus[endpoint_column]) / 2e-5
    analytic = -2 * 1.2 * math.exp(-0.7 * 2)
    result = {
        "elements": elements,
        "free_variables_including_parameter": len(variables),
        "equalities": len(equality_objects),
        "endpoint": float(point[endpoint_column]),
        "endpoint_parameter_sensitivity": float(sensitivity[endpoint_position]),
        "analytic_continuous_sensitivity": analytic,
        "absolute_discretization_difference": abs(
            float(sensitivity[endpoint_position]) - analytic
        ),
        "central_difference_discrete_sensitivity": float(finite_difference),
        "AD_implicit_vs_FD_difference": abs(
            float(sensitivity[endpoint_position]) - finite_difference
        ),
        "discrete_residual_inf": error,
    }
    if error >= 1e-10 or result["AD_implicit_vs_FD_difference"] >= 1e-7:
        raise AssertionError(result)
    return result


def main():
    """Run probes and record numerical errors and dependency failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [check_mesh(elements) for elements in (4, 8, 16)]
    if not all(
        a["absolute_discretization_difference"]
        > b["absolute_discretization_difference"]
        for a, b in zip(results, results[1:])
    ):
        raise AssertionError("Expected mesh refinement trend was not observed")
    report = {
        "purpose": "Implicit differentiation of collocation equations, not the adaptive IDAS flow map",
        "model": "x'=-k*x; x(0)=1.2; k=.7; T=2; Radau3",
        "pyomo_version": pyomo.__version__,
        "passed": True,
        "results": results,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
