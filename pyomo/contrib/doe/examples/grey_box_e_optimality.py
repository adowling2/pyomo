# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Compare raw/log E-optimality with FIM and sensitivity GreyBox inputs.

Requires IPOPT, CyIpopt, and PyNumero ASL. Both responses gain information as
u increases, so every formulation has the known optimal design u = 2.
"""
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.doe import DesignOfExperiments


class LinearResponseExperiment:
    def get_labeled_model(self):
        m = pyo.ConcreteModel()
        m.u = pyo.Var(initialize=1.0, bounds=(0.5, 2.0))
        m.p = pyo.Var([0, 1], initialize={0: 1.2, 1: 0.8})
        m.p.fix()
        m.y = pyo.Var([0, 1], initialize={0: 1.2, 1: 1.6})
        m.eq = pyo.Constraint([0, 1], rule=lambda m, i: m.y[i] == m.p[i] * (m.u + i))
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.u] = None
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for i in (0, 1):
            m.unknown_parameters[m.p[i]] = pyo.value(m.p[i])
            m.experiment_outputs[m.y[i]] = None
            m.measurement_error[m.y[i]] = i + 1.0
        return m


def main():
    prior = np.array([[1.0, 0.1], [0.1, 2.0]])
    for formulation in ('fim', 'sensitivity'):
        for objective in ('minimum_eigenvalue', 'log_minimum_eigenvalue'):
            doe = DesignOfExperiments(
                experiment=LinearResponseExperiment(),
                use_grey_box_objective=True,
                grey_box_fim_formulation=formulation,
                objective_option=objective,
                grey_box_eigenvalue_reference=2.0,
                prior_FIM=prior,
            )
            doe.run_doe()
            print(formulation, objective)
            print('Design:', doe.get_experiment_input_values())
            print('Minimum eigenvalue:', np.linalg.eigvalsh(doe.get_FIM())[0])


if __name__ == '__main__':
    main()
