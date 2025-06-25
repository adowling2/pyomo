#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Note: This is a temporary copy of the exisiting example to verify the new
# symbolic gradient method works as expected.
# Once this works, we will remove this file and update the test suite.

from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

from pyomo.contrib.parmest.experiment import Experiment


class PolynomialExperiment(Experiment):
    def __init__(self, data):
        """
        Arguments
        ---------
        data: object containing vital experimental information
        """
        self.data = data
        self.model = None

        #############################
        # End constructor definition

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    # Create flexible model without data
    def create_model(self):
        """
        Define the polynomial model for the experiment.

        y = a*x1 + b*x2 + c*x1*x2 + d

        Return
        ------
        m: a Pyomo.DAE model
        """

        m = self.model = pyo.ConcreteModel()

        # Define model variables

        # Input variables (independent variables)
        m.x1 = pyo.Var(bounds=(-5, 5), initialize=1)
        m.x2 = pyo.Var(bounds=(-5, 5), initialize=1)

        # Model coefficients (unknown parameters)
        m.a = pyo.Var(bounds=(-5, 5), initialize=2)
        m.b = pyo.Var(bounds=(-5, 5), initialize=-1)
        m.c = pyo.Var(bounds=(-5, 5), initialize=0.5)
        m.d = pyo.Var(bounds=(-5, 5), initialize=-1)

        # Model output (dependent variable)
        m.y = pyo.Var(initialize=0)

        # Define model equation

        @m.Constraint()
        def output_equation(m):
            return m.y == m.a * m.x1 + m.b * m.x2 + m.c * m.x1 * m.x2 + m.d

    def finalize_model(self):
        """
        This model is so simple that we do not need to finalize it.

        In a more complex example, we could discretize an ODE or DAE model,
        etc. here

        """
        pass

    def label_experiment(self):
        """
        Example for annotating (labeling) the model with a
        full experiment.
        """
        m = self.model

        # Set measurement labels
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = None

        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = 1

        # Identify design variables (experiment inputs) for the model
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.x1] = None
        m.experiment_inputs[m.x2] = None

        # Add unknown parameter labels (using nominal values from the model)
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.a, m.b, m.c, m.d])


def run_polynomial_doe():

    # Create an experiment object
    experiment = PolynomialExperiment(data=None)

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = False

    # Create the DesignOfExperiments object
    # We will not be passing any prior information in this example
    # and allow the experiment object and the DesignOfExperiments
    # call of ``run_doe`` perform model initialization.
    doe_obj = DesignOfExperiments(
        experiment,
        gradient_method="pynumero",  # Use Pynumero symbolic gradient
        fd_formula=None,
        step=1e-3,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=pyo.SolverFactory(
            'ipopt'
        ),  # If none, use default in Pyomo.DoE (ipopt with ma57)
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    doe_obj.compute_FIM()


if __name__ == "__main__":
    run_polynomial_doe()
