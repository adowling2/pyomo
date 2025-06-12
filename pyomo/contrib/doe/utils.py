#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners:
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin,
#  University of Toledo, West Virginia University, et al. All rights reserved.
#
#  NOTICE. This Software was developed under funding from the
#  U.S. Department of Energy and the U.S. Government consequently retains
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable,
#  worldwide license in the Software to reproduce, distribute copies to the
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________

import pyomo.environ as pyo

from pyomo.common.dependencies import numpy as np, numpy_available

from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData

from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd, reverse_ad
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet

import numpy as np
import pandas as pd

# Rescale FIM (a scaling function to help rescale FIM from parameter values)
def rescale_FIM(FIM, param_vals):
    """
    Rescales the FIM based on the input and parameter vals.
    It is assumed the parameter vals align with the FIM
    dimensions such that (1, i) corresponds to the i-th
    column or row of the FIM.

    Parameters
    ----------
    FIM: 2D numpy array to be scaled
    param_vals: scaling factors for the parameters

    """
    if isinstance(param_vals, list):
        param_vals = np.array([param_vals])
    elif isinstance(param_vals, np.ndarray):
        if len(param_vals.shape) > 2 or (
            (len(param_vals.shape) == 2) and (param_vals.shape[0] != 1)
        ):
            raise ValueError(
                "param_vals should be a vector of dimensions: 1 by `n_params`. The shape you provided is {}.".format(
                    param_vals.shape
                )
            )
        if len(param_vals.shape) == 1:
            param_vals = np.array([param_vals])
    else:
        raise ValueError(
            "param_vals should be a list or numpy array of dimensions: 1 by `n_params`"
        )
    scaling_mat = (1 / param_vals).transpose().dot((1 / param_vals))
    scaled_FIM = np.multiply(FIM, scaling_mat)
    return scaled_FIM


# TODO: Add swapping parameters for variables helper function
# def get_parameters_from_suffix(suffix, fix_vars=False):
#     """
#     Finds the Params within the suffix provided. It will also check to see
#     if there are Vars in the suffix provided. ``fix_vars`` will indicate
#     if we should fix all the Vars in the set or not.
#
#     Parameters
#     ----------
#     suffix: pyomo Suffix object, contains the components to be checked
#             as keys
#     fix_vars: boolean, whether or not to fix the Vars, default = False
#
#     Returns
#     -------
#     param_list: list of Param
#     """
#     param_list = []
#
#     # FIX THE MODEL TREE ISSUE WHERE I GET base_model.<param> INSTEAD OF <param>
#     # Check keys if they are Param or Var. Fix the vars if ``fix_vars`` is True
#     for k, v in suffix.items():
#         if isinstance(k, ParamData):
#             param_list.append(k.name)
#         elif isinstance(k, VarData):
#             if fix_vars:
#                 k.fix()
#         else:
#             pass  # ToDo: Write error for suffix keys that aren't ParamData or VarData
#
#     return param_list

class ExperimentGradients:
    def __init__(self, experiment_model, symbolic=True, automatic=True, verbose=False):
        """
        Initialize the ExperimentGradients class.
        Parameters
        ----------
        experiment_model : Pyomo model
            The experiment model to analyze.
        symbolic : bool, optional
            If True, perform symbolic differentiation. Default is True.
        automatic : bool, optional
            If True, perform automatic differentiation. Default is True.
        
        Performance tip:
        - If you are only interested in one type of differentiation (symbolic or automatic),
        you can set the other to False to save computation time.
        - If you will use the instance of this class to perform both symbolic and automatic differentiation,
        you can set both symbolic and automatic to True here.
        - This implementation assumes the experiment model will not be modified after this class is initialized.

        """
        
        self.model = experiment_model

        self.verbose = verbose

        self._analyze_experiment_model()

        self.jac_dict_sd = None
        self.jac_dict_ad = None
        self.jac_measurements_wrt_param = None
        
        if symbolic or automatic:
            # Analyze the experiment model to get the constraints and variables
            self._perform_differentiation(symbolic, automatic)

    def _analyze_experiment_model(self):
        """
        Partition the experiment model constraints and variables into
        sets for equality constraints, outputs (measurements), inputs,
        unknown parameters.

        This will build list of indices used for the performaning
        symbolic differentiation and automatic differentiation.
        """

        model = self.model

        # Loop over the design variables and fix them
        for v in model.experiment_inputs.keys():
            v.fix()

        # Loop over the unknown parameters and fix them
        for v in model.unknown_parameters.keys():
            v.fix()

        # Parameters
        # Create an empty component set
        param_set = ComponentSet()

        # Loop over the unknown model parameters
        for p in model.unknown_parameters.keys():
            param_set.add(p)

        # Assemble into a list
        
        param_list = list(param_set)

        # Measurements (outputs)
        # Create an empty component set
        output_set = ComponentSet()

        # Loop over the model outputs
        for o in model.experiment_outputs.keys():
            output_set.add(o)

        # Assemble into a list
        output_list = list(output_set)

        # Constraints and Variables
        # Create empty component sets
        con_set = ComponentSet() # These will be all constraints in the Pyomo model
        var_set = ComponentSet() # These will be all Pyomo variables in the Pyomo model

        # Loop over the active model constraints
        for c in model.component_data_objects(pyo.Constraint, descend_into=True, active=True):

            # Add constraint c to the constraint set
            con_set.add(c)

            # Loop over the variables in the constraint c
            # Note: changed this to include_fixed=True
            # Changed back to False to fix problem degree of freedom issues
            for v in identify_variables(c.body, include_fixed=False):
                # Add variable v to the variable set
                var_set.add(v)

        # recall that the parameters are fixed, so we did not
        # get them above. Let's add them now.
        for p in model.unknown_parameters.keys():
            if p not in var_set:
                # If the parameter is not in the variable set, add it
                var_set.add(p)

        # TODO: This may not be needed, likely remove
        # Keep track of outputs that are fixed
        outputs_fixed = ComponentSet()

        '''
        # Make sure all outputs are in the variable set
        for o in model.experiment_outputs.keys():
            if o not in var_set:
                # if the output is not in the variable set,
                # it was fixed. Let's keep track of it
                outputs_fixed.add(o)

                # And let's add it to the variable set
                var_set.add(o)

                outputs_fixed.add(o)
        '''

        # TODO: Keep track of measurements that are fixed -- work in progress

        # TODO: Keep track of mappings between measurement output and index
        # This is needed to assemble the Jacobian correctly
        # 1. Suffix. key = experiment outputs (parameters), value is position in list of variables
        # 2. Suffix. key = parameters , value is measurement error
        # 3. Use these two suffixes 
        
        # Keep track of the mapping from measurements to their index in the variable set
        measurement_mapping = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # Keep track of the mapping from parameters to their index in the variable set
        parameter_mapping = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # Assemble into lists
        con_list = list(con_set)
        var_list = list(var_set)

        # Create empty lists
        param_index = []
        model_var_index = []
        measurement_index = []

        # TODO: This is no longer needed, likely remove
        # Adding a `included` suffix to only
        # take outputs that are unfixed. This
        # makes indices match.
        measurement_error_included = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # Loop over the variables and determine which ones 
        # (and associated indices) are (a) parameters or 
        # (b) measurements
        # TODO: Does this considered fixed variables?
        # How does that change things? We fix all of our
        # experiment inputs and unknown parameters.
        for i, v in enumerate(var_set):
            # Check if the variable is a parameter
            if v in param_set:
                # If yes, record its index
                param_index.append(i)

                parameter_mapping[v] = i

            else:
                # Otherwise, it is a model variable
                model_var_index.append(i)

                # Check if the model variable is a measurement
                if v in output_set:
                    # If yes, record its index
                    measurement_index.append(i)
                    measurement_error_included[v] = model.measurement_error[v]

                    measurement_mapping[v] = i

        # Take care of measurements that were fixed
        for o in model.experiment_outputs.keys():
            if o not in var_set:
                # if the output is not in the variable set,
                # it was fixed.

                # Store the index as None
                measurement_mapping[o] = None

        # TODO: Check lengths here. The experiment model should be square if
        # the experiment inputs and unknown parameters are fixed.

        num_measurements = len(output_set)
        num_params = len(param_set)
        num_constraints = len(con_set)
        num_vars = len(var_set)
        num_inputs = len(model.experiment_inputs)

        # TODO: This is likely not needed, likely remove
        num_fixed_measurements = len(outputs_fixed)

        if self.verbose:
            print("Experiment model size:")

            print(f"  {num_vars} total variables")
            print(f"  {num_measurements} outputs (measurements)")
            print(f"  {num_inputs} inputs")
            print(f"  {num_params} unknown parameters")
            print(f"  {num_constraints} constraints\n")

        if num_vars - num_params - num_fixed_measurements != num_constraints:
            raise ValueError("The model is not square: the number of variables minus unknown parameters does not equal the number of constraints.\n" \
            "This is required for the (automatic) differentiation to work correctly.")

        # Save terms that are needed for later
        self.con_list = con_list
        self.var_list = var_list
        self.param_list = param_list

        self.param_index = param_index
        self.model_var_index = model_var_index
        self.measurement_index = measurement_index
        
        self.measurement_error_included = measurement_error_included
        
        self.num_measurements = num_measurements
        self.num_params = num_params
        self.num_constraints = num_constraints
        self.num_vars = num_vars

        self.var_set = var_set

        self.measurement_mapping = measurement_mapping
        self.parameter_mapping = parameter_mapping

    def _perform_differentiation(self, symbolic=True, automatic=True):
    
        # Initialize dictionaries to hold the Jacobian entries
        if symbolic:
            jac_dict_sd = {}
        if automatic:
            jac_dict_ad = {}

        if not symbolic and not automatic:
            raise ValueError("At least one differentiation method must be selected: symbolic or automatic.")

        # Grab data needed for the differentiation
        con_list = self.con_list
        var_list = self.var_list

        # Enumerate over the constraints
        for i, c in enumerate(con_list):
            # Check we only have equality constraints... otherwise this gets more complicated
            assert c.equality, "This function only works with equality constraints"
            
            # Perform symbolic differentiation
            if symbolic:
                der_map_sd = reverse_sd(c.body)
            
            if automatic:
                der_map_ad = reverse_ad(c.body)

            # Loop over the Pyomo variables, which includes 
            # parameters, measurements, control decisions
            for j, v in enumerate(var_list):

                # Symbolic differentiation
                if symbolic:
                    # Check if the variable is in the derivative map
                    if v in der_map_sd:
                        # Record the expression 
                        deriv = der_map_sd[v]
                    else:
                        # Otherwise, record 0
                        deriv = 0
                    # Save results in the Jacobian dictionary
                    jac_dict_sd[(i, j)] = deriv

                # Automatic differentiation
                if automatic:
                    if v in der_map_ad:
                        # Record the expression 
                        deriv = der_map_ad[v]
                    else:
                        # Otherwise, record 0
                        deriv = 0
                    # Save results in the Jacobian dictionary
                    jac_dict_ad[(i, j)] = deriv        

        if symbolic:
            self.jac_dict_sd = jac_dict_sd
        if automatic:
            self.jac_dict_ad = jac_dict_ad

    def compute_gradient_outputs_wrt_unknown_parameters(self):
        """ Perform automatic differentiation to compute the gradients of the outputs 
        with respect to the unknown parameters.
        
    
        """

        if self.jac_dict_ad is None:
            # Perform automatic differentiation if not already done
            self._perform_differentiation(symbolic=False, automatic=True)

        # Grab the necessary data from the instance
        # (this keeps variable names shorter below)
        num_constraints = self.num_constraints
        num_params = self.num_params
        num_measurements = self.num_measurements
        param_index = self.param_index
        model_var_index = self.model_var_index
        jac_dict_ad = self.jac_dict_ad
        measurement_index = self.measurement_index
    
        jac_con_wrt_param = np.zeros((num_constraints, num_params))
        for i in range(num_constraints):
            for j, p in enumerate(param_index):
                jac_con_wrt_param[i, j] = jac_dict_ad[(i, p)]

        jac_con_wrt_vars = np.zeros((num_constraints, len(model_var_index)))
        for i in range(num_constraints):
            for j, v in enumerate(model_var_index):
                jac_con_wrt_vars[i, j] = jac_dict_ad[(i, v)]

        if self.verbose:
            print(f"Jacobian of constraints with respect to parameters shape: {jac_con_wrt_param.shape}")
            print(f"Jacobian of constraints with respect to variables shape: {jac_con_wrt_vars.shape}")

        jac_vars_wrt_param = np.linalg.solve(
            jac_con_wrt_vars, -jac_con_wrt_param
        )

        jac_measurements_wrt_param = np.zeros((num_measurements, num_params))

        # print(f"Jacobian of all variables with respect to parameters:\n{jac_vars_wrt_param}")

        # TODO: What about measurements that are fixed? They should be insensitive to changes in the model parameters.

        # TODO: Need to convert the order of measurement here (corresponds to var_set) with the order in experiment_outputs
        # If the experiment_output is NOT in var_set, then it's row should be all zeros
        # Pseudocode:
        for ind, m in enumerate(self.model.experiment_outputs.keys()):

            '''

            if m not in self.var_set:
                if self.verbose:
                    # If the measurement is not in the variable set, print a message
                    # and skip it
                    print('Measurement {} is not in the variable list.'.format(m))
                # Set the row in jac_measurements_wrt_param to all zeros
                jac_measurements_wrt_param[ind, :] = 0.0
            else:
                # Find the index of the measurement in the variable list

                i = None

                # Pyomo team: Is there a better way to do this?
                for j, v in enumerate(self.var_set):
                    if v is m:
                        # Found the measurement in the variable set
                        # Get the index of the measurement in the variable list
                        # This is needed to get the row in jac_vars_wrt_param
                        # that corresponds to this measurement
                        i = j
                        break

                if self.verbose:
                        print(f"Measurement {m} found at index {i} in variable set.")
                
                jac_measurements_wrt_param[ind, :] = jac_vars_wrt_param[i, :]

                '''

            i = self.measurement_mapping[m]

            if i is None:
                jac_measurements_wrt_param[ind, :] = 0.0
            else:
                # If the measurement is in the variable set, get the row
                jac_measurements_wrt_param[ind, :] = jac_vars_wrt_param[i, :]
                
                
        # The measurement_index is the index of the measurement in the var_set

        # 1. Look over experiment_outputs
        # 2. Find index for element in var_set
        # 3. Grab the row correspond to the index from jac_vars_wrt_param
        # 4. Store in numpy array for Jacobian

        # jac_measurements_wrt_param = jac_vars_wrt_param[measurement_index, :]

        # print(f"Jacobian of measurements with respect to parameters:\n{jac_measurements_wrt_param}")

        self.jac_measurements_wrt_param = jac_measurements_wrt_param

        return jac_measurements_wrt_param
    
    def _package_jac_as_df(self, jac):
        """
        Convert a numpy array containing the Jacobian into a
        pandas DataFrame

        Arguments:
            jac: numpy array where rows are measurements and columns are parameters

        Returns:
            pandas DataFrame

        """

        var_list = self.var_list
        measurement_index = self.measurement_index
        param_index = self.param_index

        row_names = [str(o) for o in self.model.experiment_outputs.keys()]
        col_names = [str(p) for p in self.model.unknown_parameters.keys()]

        return pd.DataFrame(jac, index=row_names, columns=col_names)

    def get_numeric_sensitivity_as_df(self):
        if not self.jac_measurements_wrt_param:
            self.compute_gradient_outputs_wrt_unknown_parameters()

        return self._package_jac_as_df(self.jac_measurements_wrt_param)
    
    def get_sensitivities_from_symbolic_as_df(self):

        model = self.model

        if not hasattr(model, 'jacobian_constraint'):
            self.construct_sensitivity_constraints()

        solver = pyo.SolverFactory('ipopt')
        results1 = solver.solve(model, tee=self.verbose)

        jac = np.zeros((len(model.measurement_index), len(model.param_index)))

        for i,y in enumerate(model.measurement_index):
            for j,p in enumerate(model.param_index):
                jac[i,j] = model.jac_variables_wrt_param[y,p].value
        
        return self._package_jac_as_df(jac)

    def construct_sensitivity_constraints(self, model=None):

        if self.jac_dict_sd is None:
            # Perform symbolic differentiation if not already done
            self._perform_differentiation(symbolic=True, automatic=False)

        # Decision: Build these constraints in the model. Pyomo.DoE can look into scenarion[0]

        if model is None:
            model = self.model

        # Using the lists of indices to create Pyomo Sets
        model.param_index = pyo.Set(initialize=self.param_index)
        model.measurement_index = pyo.Set(initialize=self.measurement_index)
        model.constraint_index = pyo.Set(initialize=range(len(self.con_list)))
        model.var_index = pyo.Set(initialize=self.model_var_index)

        # Define a Pyomo variable for the Jacobian of the model variables 
        # (everything except parameters) with respect to the model parameters
        model.jac_variables_wrt_param = pyo.Var(model.var_index, model.param_index, initialize=0)

        # Calculate the Jacobian using the chain rule and total derivative definitions
        #
        # Prior comment:
        # This has an index mistake... jac_dict_sd includes the parameters, but var_index skips them
        # We need to be more careful about the indices
        #
        # New reflection:
        # var_index is built from the indices in var_list, which includes the parameters
        # I think this is okay
        @model.Constraint(model.constraint_index, model.param_index)
        def jacobian_constraint(model, i, j):
            return self.jac_dict_sd[(i,j)] == -sum(model.jac_variables_wrt_param[k,j] * self.jac_dict_sd[(i,k)] for k in model.var_index)