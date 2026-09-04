# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
# Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation
# Initiative (CCSI), and is copyright (c) 2022 by the software owners:
# TRIAD National Security, LLC., Lawrence Livermore National Security, LLC.,
# Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,
# Battelle Memorial Institute, University of Notre Dame,
# The University of Pittsburgh, The University of Texas at Austin,
# University of Toledo, West Virginia University, et al. All rights reserved.
#
# NOTICE. This Software was developed under funding from the
# U.S. Department of Energy and the U.S. Government consequently retains
# certain rights. As such, the U.S. Government has been granted for itself
# and others acting on its behalf a paid-up, nonexclusive, irrevocable,
# worldwide license in the Software to reproduce, distribute copies to the
# public, prepare derivative works, and perform publicly and display
# publicly, and to permit other to do so.
# ____________________________________________________________________________________

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

from enum import Enum
import itertools
import logging

if scipy_available and numpy_available:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

import pyomo.environ as pyo


class FIMExternalGreyBox(
    ExternalGreyBoxModel if (scipy_available and numpy_available) else object
):
    def __init__(
        self,
        doe_object,
        objective_option="determinant",
        logger_level=None,
        fim_formulation="fim",
        eigenvalue_floor=1e-8,
        softplus_beta=50.0,
        softmin_temperature=1e-2,
        shift_penalty=1e3,
        hessian_mode="exact",
        eigenvalue_reference=1.0,
    ):
        """
        Grey box model for metrics on the FIM. This methodology reduces
        numerical complexity for the computation of FIM metrics related
        to eigenvalue decomposition.

        Parameters
        ----------
        doe_object:
           Design of Experiments object that contains a built model
           (with sensitivity matrix, Q, and fisher information matrix, FIM).
           The external grey box model will utilize elements of the
           `doe_object` model to build the FIM metric with consistent naming.
        obj_option:
           String representation of the objective option. Current available
           options are: ``determinant`` (D-optimality), ``trace`` (A-optimality),
           ``minimum_eigenvalue`` (E-optimality), ``condition_number``
           (modified E-optimality).
           default: ``determinant``
        fim_formulation:
           One of ``fim``, ``sensitivity``, ``softplus_exact``, or
           ``softplus_smooth``. The default retains the lifted-FIM behavior.
        eigenvalue_floor:
           Target minimum eigenvalue for either softplus formulation.
        softplus_beta:
           Positive sharpness parameter for the softplus shift.
        softmin_temperature:
           Positive spectral soft-min temperature for ``softplus_smooth``.
        shift_penalty:
           Nonnegative objective penalty on the diagonal shift.
        hessian_mode:
           One of ``exact``, ``gauss-newton``, ``projected-psd``, or
           ``gauss-newton-psd``. Gauss-Newton modes are available only with
           the sensitivity formulation.
        eigenvalue_reference:
           Positive reference used to nondimensionalize the minimum eigenvalue
           for the ``log_minimum_eigenvalue`` objective. Default: 1.
        logger_level:
           logging level to be specified if different from doe_object's logging level.
           default: None, or equivalently, use the logging level of doe_object.

           NOTE: Use logging.DEBUG for all messages.
        """

        if doe_object is None:
            raise ValueError(
                "DoE Object must be provided to build external grey box of the FIM."
            )

        self.doe_object = doe_object

        # Grab parameter list from the doe_object model
        self._param_names = [i for i in self.doe_object.model.parameter_names]
        self._n_params = len(self._param_names)
        self._fim_input_names = list(
            itertools.combinations_with_replacement(self._param_names, 2)
        )

        if isinstance(fim_formulation, Enum):
            fim_formulation = fim_formulation.value
        valid_formulations = {"fim", "sensitivity", "softplus_exact", "softplus_smooth"}
        if fim_formulation not in valid_formulations:
            raise ValueError(
                "fim_formulation must be one of %s; received %r."
                % (sorted(valid_formulations), fim_formulation)
            )
        self.fim_formulation = fim_formulation
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.softplus_beta = float(softplus_beta)
        self.softmin_temperature = float(softmin_temperature)
        self.shift_penalty = float(shift_penalty)
        self.hessian_mode = str(hessian_mode)
        self.eigenvalue_reference = float(eigenvalue_reference)
        valid_hessian_modes = {
            "exact",
            "gauss-newton",
            "projected-psd",
            "gauss-newton-psd",
        }
        if self.hessian_mode not in valid_hessian_modes:
            raise ValueError(
                "hessian_mode must be one of %s; received %r."
                % (sorted(valid_hessian_modes), self.hessian_mode)
            )
        if (
            self.hessian_mode.startswith("gauss-newton")
            and self.fim_formulation != "sensitivity"
        ):
            raise ValueError(
                "Gauss-Newton Hessian modes require fim_formulation='sensitivity'."
            )
        if self.softplus_beta <= 0:
            raise ValueError("softplus_beta must be positive.")
        if self.softmin_temperature <= 0:
            raise ValueError("softmin_temperature must be positive.")
        if self.shift_penalty < 0:
            raise ValueError("shift_penalty must be nonnegative.")
        if self.eigenvalue_reference <= 0:
            raise ValueError("eigenvalue_reference must be positive.")

        # Check if the doe_object has model components that are required
        # TODO: is this check necessary?
        from pyomo.contrib.doe import ObjectiveLib

        objective_option = ObjectiveLib(objective_option)
        self.objective_option = objective_option

        # Create logger for FIM egb object
        self.logger = logging.getLogger(__name__)

        # If logger level is None, use doe_object's logger level
        if logger_level is None:
            logger_level = doe_object.logger.level

        self.logger.setLevel(level=logger_level)

        # Set initial values for inputs.
        self._masking_matrix = np.triu(np.ones_like(self.doe_object.fim_initial))
        if self.fim_formulation == "sensitivity":
            self._measurement_names = [i for i in self.doe_object.model.output_names]
            if self.doe_object.jac_initial is None:
                raise ValueError(
                    "jac_initial is required for the sensitivity GreyBox formulation."
                )
            jac_initial = np.asarray(self.doe_object.jac_initial, dtype=np.float64)
            expected_shape = (len(self._measurement_names), self._n_params)
            if jac_initial.shape != expected_shape:
                raise ValueError(
                    "jac_initial has shape %s; expected %s for the sensitivity "
                    "GreyBox formulation." % (jac_initial.shape, expected_shape)
                )
            self._input_values = jac_initial.flatten()
            self._prior_FIM = np.asarray(self.doe_object.prior_FIM, dtype=np.float64)
            scenario = self.doe_object.model.scenario_blocks[0]
            self._measurement_weights = np.asarray(
                [
                    1.0
                    / float(
                        scenario.measurement_error[
                            pyo.ComponentUID(name).find_component_on(scenario)
                        ]
                    )
                    ** 2
                    for name in self._measurement_names
                ],
                dtype=np.float64,
            )
        else:
            self._input_values = np.asarray(
                self.doe_object.fim_initial[self._masking_matrix > 0], dtype=np.float64
            )
        self._n_inputs = len(self._input_values)

        # The solver updates this value before requesting the Hessian of the
        # Lagrangian.  A unit default preserves the unweighted output Hessian
        # for direct users of the external model.
        self._output_con_mult_values = np.ones(self.n_outputs(), dtype=np.float64)

    def _get_raw_FIM(self):
        if self.fim_formulation == "sensitivity":
            sensitivity = self._input_values.reshape(
                len(self._measurement_names), self._n_params
            )
            return (
                sensitivity.T @ (self._measurement_weights[:, None] * sensitivity)
                + self._prior_FIM
            )

        # Grabs the current FIM subject
        # to the input values.
        # Inputs store one triangular half
        # of a symmetric FIM. Reconstruct
        # the full symmetric matrix here,
        # consistent with manuscript equation S5.
        # https://arxiv.org/abs/2604.03354v1
        upt_FIM = self._input_values

        # Create FIM in the correct way
        current_FIM = np.zeros_like(self.doe_object.fim_initial)
        # Utilize upper triangular portion of FIM
        current_FIM[np.triu_indices_from(current_FIM)] = upt_FIM
        # Construct lower triangular using the
        # current upper triangle minus the diagonal.
        current_FIM += current_FIM.transpose() - np.diag(np.diag(current_FIM))

        return current_FIM

    @staticmethod
    def _stable_softplus(value, beta):
        return np.logaddexp(0.0, beta * value) / beta

    def _shift_information(self, current_FIM, derivative_order=2):
        """Return shift, gradient, and Hessian with respect to packed FIM inputs."""
        eigenvalues, eigenvectors = np.linalg.eigh(current_FIM)
        if self.fim_formulation == "softplus_exact":
            minimum_value = eigenvalues[0]
            weights = None
        else:
            temperature = self.softmin_temperature
            scaled = -eigenvalues / temperature
            scaled -= np.max(scaled)
            weights = np.exp(scaled)
            weights /= np.sum(weights)
            minimum_value = -temperature * scipy.special.logsumexp(
                -eigenvalues / temperature
            )

        argument = self.eigenvalue_floor - minimum_value
        sigmoid = scipy.special.expit(self.softplus_beta * argument)
        shift = self._stable_softplus(argument, self.softplus_beta)
        if derivative_order == 0:
            return shift, None, None

        n_fim_inputs = len(self._fim_input_names)
        basis = []
        for row_name, col_name in self._fim_input_names:
            row = self._param_names.index(row_name)
            col = self._param_names.index(col_name)
            matrix = np.zeros_like(current_FIM)
            matrix[row, col] = 1.0
            matrix[col, row] = 1.0
            basis.append(matrix)

        if self.fim_formulation == "softplus_exact":
            minimum_vector = eigenvectors[:, 0]
            eigenvalue_tolerance = np.sqrt(np.finfo(float).eps) * max(
                1.0, np.max(np.abs(eigenvalues))
            )
            if (
                self._n_params > 1
                and abs(eigenvalues[1] - eigenvalues[0]) <= eigenvalue_tolerance
            ):
                raise ValueError(
                    "softplus_exact is not differentiable at a repeated minimum "
                    "eigenvalue; use softplus_smooth instead."
                )
            minimum_gradient = np.asarray(
                [minimum_vector @ matrix @ minimum_vector for matrix in basis]
            )
            if derivative_order == 1:
                return shift, -sigmoid * minimum_gradient, None
            minimum_hessian = np.zeros((n_fim_inputs, n_fim_inputs))
            for other in range(1, self._n_params):
                denominator = minimum_value - eigenvalues[other]
                other_vector = eigenvectors[:, other]
                projections = np.asarray(
                    [other_vector @ matrix @ minimum_vector for matrix in basis]
                )
                minimum_hessian += (
                    2.0 * np.outer(projections, projections) / denominator
                )
        else:
            transformed_basis = [
                eigenvectors.T @ matrix @ eigenvectors for matrix in basis
            ]
            minimum_gradient = np.asarray(
                [np.dot(weights, np.diag(matrix)) for matrix in transformed_basis]
            )
            if derivative_order == 1:
                return shift, -sigmoid * minimum_gradient, None
            eigenvalue_hessian = (
                -(np.diag(weights) - np.outer(weights, weights)) / temperature
            )
            minimum_hessian = np.zeros((n_fim_inputs, n_fim_inputs))
            for row, row_matrix in enumerate(transformed_basis):
                row_diagonal = np.diag(row_matrix)
                for col, col_matrix in enumerate(transformed_basis):
                    value = row_diagonal @ eigenvalue_hessian @ np.diag(col_matrix)
                    for i in range(self._n_params):
                        for j in range(i + 1, self._n_params):
                            gap = eigenvalues[i] - eigenvalues[j]
                            if abs(gap) <= np.finfo(float).eps:
                                divided_difference = -weights[i] / temperature
                            else:
                                divided_difference = (weights[i] - weights[j]) / gap
                            value += (
                                2.0
                                * divided_difference
                                * row_matrix[i, j]
                                * col_matrix[i, j]
                            )
                    minimum_hessian[row, col] = value

        shift_gradient = -sigmoid * minimum_gradient
        shift_hessian = (
            self.softplus_beta
            * sigmoid
            * (1.0 - sigmoid)
            * np.outer(minimum_gradient, minimum_gradient)
            - sigmoid * minimum_hessian
        )
        return shift, shift_gradient, shift_hessian

    def _get_FIM(self):
        current_FIM = self._get_raw_FIM()
        if self.fim_formulation.startswith("softplus_"):
            shift, _, _ = self._shift_information(current_FIM, derivative_order=0)
            current_FIM = current_FIM + shift * np.eye(self._n_params)
        return current_FIM

    def _shift_penalty_sign(self):
        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option in (ObjectiveLib.trace, ObjectiveLib.condition_number):
            return 1.0
        return -1.0

    def regularization_diagnostics(self):
        """Return terminal softplus-shift activity and exact-penalty metrics."""
        if not self.fim_formulation.startswith("softplus_"):
            return None
        raw_fim = self._get_raw_FIM()
        shift, _, _ = self._shift_information(raw_fim, derivative_order=0)
        effective_fim = raw_fim + shift * np.eye(self._n_params)
        raw_eigenvalues = np.linalg.eigvalsh(raw_fim)
        effective_eigenvalues = raw_eigenvalues + shift
        argument = self.eigenvalue_floor
        if self.fim_formulation == "softplus_exact":
            spectral_minimum = raw_eigenvalues[0]
        else:
            spectral_minimum = -self.softmin_temperature * scipy.special.logsumexp(
                -raw_eigenvalues / self.softmin_temperature
            )
        argument -= spectral_minimum
        activity = scipy.special.expit(self.softplus_beta * argument)

        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            inverse = np.linalg.pinv(effective_fim)
            shift_benefit_slope = np.trace(inverse @ inverse)
        elif self.objective_option == ObjectiveLib.determinant:
            shift_benefit_slope = np.trace(np.linalg.pinv(effective_fim))
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            shift_benefit_slope = 1.0
        elif self.objective_option == ObjectiveLib.log_minimum_eigenvalue:
            shift_benefit_slope = 1.0 / effective_eigenvalues[0]
        elif self.objective_option == ObjectiveLib.condition_number:
            shift_benefit_slope = (
                1.0 / effective_eigenvalues[0]
                - 1.0 / effective_eigenvalues[-1]
            )
        else:
            shift_benefit_slope = 0.0

        return {
            "Raw FIM": raw_fim.tolist(),
            "Effective FIM": effective_fim.tolist(),
            "Raw FIM Eigenvalues": raw_eigenvalues.tolist(),
            "Effective FIM Eigenvalues": effective_eigenvalues.tolist(),
            "GreyBox Spectral Minimum": float(spectral_minimum),
            "GreyBox Diagonal Shift": float(shift),
            "GreyBox Shift Activity": float(activity),
            "GreyBox Shift Penalty Contribution": float(
                self.shift_penalty * shift
            ),
            "GreyBox Shift Benefit Slope": float(shift_benefit_slope),
            "GreyBox Shift Penalty Margin": float(
                self.shift_penalty - shift_benefit_slope
            ),
        }

    def _reorder_pairs(self, i, j, k, l):
        # Reorders the pairs (i, j) and
        # (k, l) for considering only
        # the symmetric portion of the FIM
        # while calculating the Hessian

        # If the pairs ((i, j), (k, l)) are not
        # in increasing order, we reorder
        # the pairs.
        if i > j:
            if k > l:
                return [j, i, l, k]
            else:
                return [j, i, k, l]
        else:
            if k > l:
                return [i, j, l, k]
        return [i, j, k, l]

    def input_names(self):
        # Cartesian product gives us matrix indices flattened in row-first format
        # Can use itertools.combinations(self._param_names, 2) with added
        # diagonal elements, or do double for loops if we switch to upper triangular
        if self.fim_formulation == "sensitivity":
            return list(itertools.product(self._measurement_names, self._param_names))
        return self._fim_input_names

    def equality_constraint_names(self):
        # TODO: Are there any objectives that will have constraints?
        return []

    def output_names(self):
        # TODO: add output name for the variable. This may have to be
        # an input from the user. Or it could depend on the usage of
        # the ObjectiveLib Enum object, which should have an associated
        # name for the objective function at all times.
        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            obj_name = "A-opt"
        elif self.objective_option == ObjectiveLib.pseudo_trace:
            obj_name = "pseudo-A-opt"
        elif self.objective_option == ObjectiveLib.determinant:
            obj_name = "log-D-opt"
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            obj_name = "E-opt"
        elif self.objective_option == ObjectiveLib.log_minimum_eigenvalue:
            obj_name = "log-E-opt"
        elif self.objective_option == ObjectiveLib.condition_number:
            obj_name = "ME-opt"
        else:
            ObjectiveLib(self.objective_option)
        return [obj_name]

    def set_input_values(self, input_values):
        # Set initial values to be flattened initial FIM (aligns with input names)
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        # TODO: are there any objectives that will have constraints?
        return None

    def evaluate_outputs(self):
        # Evaluates the objective value for the specified
        # ObjectiveLib type.
        current_FIM = self._get_FIM()

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        # Change objective value based on ObjectiveLib type.
        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            obj_value = np.trace(np.linalg.pinv(M))
        elif self.objective_option == ObjectiveLib.pseudo_trace:
            obj_value = np.trace(M)
        elif self.objective_option == ObjectiveLib.determinant:
            sign, logdet = np.linalg.slogdet(M)
            obj_value = logdet
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            obj_value = np.linalg.eigvalsh(M)[0]
        elif self.objective_option == ObjectiveLib.log_minimum_eigenvalue:
            minimum_eigenvalue = np.linalg.eigvalsh(M)[0]
            if minimum_eigenvalue <= 0:
                raise ValueError(
                    "log_minimum_eigenvalue requires a positive-definite "
                    "information matrix."
                )
            obj_value = np.log(minimum_eigenvalue / self.eigenvalue_reference)
        elif self.objective_option == ObjectiveLib.condition_number:
            eig, _ = np.linalg.eig(M)
            obj_value = np.log(np.abs(np.max(eig) / np.min(eig)))
        else:
            ObjectiveLib(self.objective_option)

        if self.fim_formulation.startswith("softplus_"):
            shift, _, _ = self._shift_information(
                self._get_raw_FIM(), derivative_order=0
            )
            obj_value += self._shift_penalty_sign() * self.shift_penalty * shift

        return np.asarray([obj_value], dtype=np.float64)

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # Set initial values of the inputs/outputs
        # This will depend on the objective used

        # Initialize GreyBox inputs in the same order used by set_input_values.
        for ind, val in enumerate(self.input_names()):
            pyomo_block.inputs[val] = self._input_values[ind]

        # Initialize log_determinant value
        from pyomo.contrib.doe import ObjectiveLib

        # Calculate initial values for the output
        output_value = self.evaluate_outputs()[0]

        # Set the value of the output for the given
        # objective function.
        if self.objective_option == ObjectiveLib.trace:
            pyomo_block.outputs["A-opt"] = output_value
        elif self.objective_option == ObjectiveLib.pseudo_trace:
            pyomo_block.outputs["pseudo-A-opt"] = output_value
        elif self.objective_option == ObjectiveLib.determinant:
            pyomo_block.outputs["log-D-opt"] = output_value
        elif self.objective_option == ObjectiveLib.minimum_eigenvalue:
            pyomo_block.outputs["E-opt"] = output_value
        elif self.objective_option == ObjectiveLib.log_minimum_eigenvalue:
            pyomo_block.outputs["log-E-opt"] = output_value
        elif self.objective_option == ObjectiveLib.condition_number:
            pyomo_block.outputs["ME-opt"] = output_value

    def evaluate_jacobian_equality_constraints(self):
        # TODO: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return None

    def _objective_gradient_matrix(self, M):
        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            Minv = np.linalg.pinv(M)
            # Derivative formula of A-optimality
            # is -inv(FIM) @ inv(FIM). Add reference to
            # pyomo.DoE 2.0 manuscript S.I.
            jac_M = -Minv @ Minv
        elif self.objective_option == ObjectiveLib.pseudo_trace:
            jac_M = np.eye(self._n_params, dtype=np.float64)
        elif self.objective_option == ObjectiveLib.determinant:
            Minv = np.linalg.pinv(M)
            # Derivative formula derived using tensor
            # calculus. Add reference to pyomo.DoE 2.0
            # manuscript S.I.
            jac_M = 0.5 * (Minv + Minv.transpose())
        elif self.objective_option in (
            ObjectiveLib.minimum_eigenvalue,
            ObjectiveLib.log_minimum_eigenvalue,
        ):
            eig_vals, eig_vecs = np.linalg.eigh(M)
            # Obtain minimum eigenvalue location
            min_eig_loc = np.argmin(eig_vals)

            # Grab eigenvector associated with
            # the minimum eigenvalue and make
            # it a matrix. This is so we can
            # use matrix operations later in
            # the code.
            min_eig_vec = np.array([eig_vecs[:, min_eig_loc]])

            # Calculate the derivative matrix.
            # This is the expansion product of
            # the eigenvector we grabbed in
            # the previous line of code.
            jac_M = min_eig_vec * np.transpose(min_eig_vec)
            if self.objective_option == ObjectiveLib.log_minimum_eigenvalue:
                minimum_eigenvalue = eig_vals[min_eig_loc]
                if minimum_eigenvalue <= 0:
                    raise ValueError(
                        "log_minimum_eigenvalue requires a positive-definite "
                        "information matrix."
                    )
                jac_M /= minimum_eigenvalue
        elif self.objective_option == ObjectiveLib.condition_number:
            eig_vals, eig_vecs = np.linalg.eigh(M)
            # Obtain minimum (and maximum) eigenvalue location(s)
            min_eig_loc = np.argmin(eig_vals)
            max_eig_loc = np.argmax(eig_vals)

            min_eig = np.min(eig_vals)
            max_eig = np.max(eig_vals)

            # Grab eigenvector associated with
            # the min (and max) eigenvalue and make
            # it a matrix. This is so we can
            # use matrix operations later in
            # the code.
            min_eig_vec = np.array([eig_vecs[:, min_eig_loc]])
            max_eig_vec = np.array([eig_vecs[:, max_eig_loc]])

            # Calculate the derivative matrix.
            # Similar to minimum eigenvalue,
            # this computation involves two
            # expansion products.
            min_eig_term = min_eig_vec * np.transpose(min_eig_vec)
            max_eig_term = max_eig_vec * np.transpose(max_eig_vec)

            # Combining the expression
            jac_M = 1 / max_eig * max_eig_term - 1 / min_eig * min_eig_term
        else:
            ObjectiveLib(self.objective_option)
        return jac_M

    def _sensitivity_to_fim_jacobian(self):
        sensitivity = self._input_values.reshape(
            len(self._measurement_names), self._n_params
        )
        derivative = np.zeros((len(self._fim_input_names), self._n_inputs))
        for fim_index, (row_name, col_name) in enumerate(self._fim_input_names):
            row = self._param_names.index(row_name)
            col = self._param_names.index(col_name)
            for measurement in range(len(self._measurement_names)):
                weight = self._measurement_weights[measurement]
                if row == col:
                    derivative[fim_index, measurement * self._n_params + row] = (
                        2.0 * weight * sensitivity[measurement, row]
                    )
                else:
                    derivative[fim_index, measurement * self._n_params + row] = (
                        weight * sensitivity[measurement, col]
                    )
                    derivative[fim_index, measurement * self._n_params + col] = (
                        weight * sensitivity[measurement, row]
                    )
        return derivative

    @staticmethod
    def _pack_symmetric_gradient(gradient):
        packed = 2.0 * gradient - np.diag(np.diag(gradient))
        return packed[np.triu_indices_from(packed)]

    def evaluate_jacobian_outputs(self):
        # Compute the objective gradient with respect to the selected GreyBox
        # inputs and return the sparse row expected by PyNumero.
        M = np.asarray(self._get_FIM(), dtype=np.float64).reshape(
            self._n_params, self._n_params
        )
        gradient_matrix = self._objective_gradient_matrix(M)
        packed_gradient = self._pack_symmetric_gradient(gradient_matrix)

        if self.fim_formulation == "sensitivity":
            jacobian = packed_gradient @ self._sensitivity_to_fim_jacobian()
        elif self.fim_formulation.startswith("softplus_"):
            _, shift_gradient, _ = self._shift_information(
                self._get_raw_FIM(), derivative_order=1
            )
            jacobian = (
                packed_gradient
                + (
                    np.trace(gradient_matrix)
                    + self._shift_penalty_sign() * self.shift_penalty
                )
                * shift_gradient
            )
        else:
            jacobian = packed_gradient

        rows = np.zeros(len(jacobian), dtype=int)
        cols = np.arange(len(jacobian))

        return scipy.sparse.coo_matrix(
            (jacobian, (rows, cols)), shape=(1, self._n_inputs)
        )

    # Beyond here is for Hessian information
    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        # TODO: Do any objectives require constraints?
        # Assert lengths match
        self._eq_con_mult_values = np.asarray(
            eq_con_multiplier_values, dtype=np.float64
        )

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        output_con_multiplier_values = np.asarray(
            output_con_multiplier_values, dtype=np.float64
        )
        assert self.n_outputs() == len(output_con_multiplier_values)
        self._output_con_mult_values = output_con_multiplier_values

    def evaluate_hessian_equality_constraints(self):
        # Returns coo_matrix of the correct shape
        # No constraints so this returns `None`
        return None

    def evaluate_hessian_outputs(self):
        # Compute the hessian of the objective function with
        # respect to the fisher information matrix. Then, return
        # a coo_matrix that aligns with what IPOPT will expect.
        current_FIM = self._get_FIM()

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        # We will store the Hessian values in
        # vectorized (flattened) format. The length
        # of the vectorized Hessian for the symmetric
        # FIM representation scales by the number of
        # unknown parameters.
        hess_array_length = round(
            (((self._n_params + 1) * self._n_params / 2) + 1)
            * (((self._n_params + 1) * self._n_params / 2))
            / 2
        )

        # Initializing lists of the correct length
        # for the hessian values and the row and column
        # of these data in the coo matrix to be returned
        hess_vals = [0] * hess_array_length
        hess_rows = [0] * hess_array_length
        hess_cols = [0] * hess_array_length

        # We are utilizing the symmetric Hessian, but we
        # must consider the contribution from all elements.
        # Therefore, we are required to use the full product
        # space of the parameter names (full FIM) to compute
        # the Hessian of the symmetric FIM.
        full_input_names = itertools.product(self._param_names, repeat=2)

        # Here, we use combination with replacement to only
        # consider the upper triangle of the Hessian for the
        # full FIM. We will map these second derivative values
        # back onto the symmetric FIM Hessian.
        input_differentials_2D = itertools.combinations_with_replacement(
            full_input_names, 2
        )

        from pyomo.contrib.doe import ObjectiveLib

        if self.objective_option == ObjectiveLib.trace:
            # Grab Inverse
            Minv = np.linalg.pinv(M)

            # Also grab inverse squared
            Minv_sq = Minv @ Minv

            for current_differential in input_differentials_2D:
                d1, d2 = current_differential

                # Grabbing the ordered quadruple (i, j, k, l)
                # `location` here refers to the index in the
                # self._param_names list
                #
                # i is the location of the first element of d1
                # j is the location of the second element of d1
                # k is the location of the first element of d2
                # l is the location of the second element of d2
                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # New Formula (tested with finite differencing)
                # Will be cited from the Pyomo.DoE 2.0 paper
                hess_contribution = (Minv[i, l] * Minv_sq[k, j]) + (
                    Minv_sq[i, l] * Minv[k, j]
                )

                # Since we are considering the full matrix in
                # this loop, we need to point the contribution
                # to the correct index for the symmetric FIM
                # Hessian.
                reordered_ijkl = self._reorder_pairs(i, j, k, l)
                d1_symmetric = (
                    self._param_names[reordered_ijkl[0]],
                    self._param_names[reordered_ijkl[1]],
                )
                d2_symmetric = (
                    self._param_names[reordered_ijkl[2]],
                    self._param_names[reordered_ijkl[3]],
                )

                # Identify what index of the symmetric FIM
                # Hessian arrays need to be updated.
                # Note: we are only interested in building
                # the lower triangular portion of the Hessian.
                row = max(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                col = min(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                flattened_row_col_index = (row + 1) * row // 2 + col

                # Hessian needs to be handled carefully because of
                # the ``missing`` components from the full FIM
                # when only passing a symmetric version of the FIM.
                #
                # When we reordered (i, j, k, l), we are correctly
                # pointing to which index needs to be contributed to.
                # However, when an element that is not included
                # is being mapped to a diagonal element of the
                # symmetric FIM hessian from the full FIM hessian,
                # it needs to be counted twice. This only occurs
                # when (i != j) and (k != l) and (i, j) and (k, l)
                # are the conjugate of one another:
                # (i == l) and (j == k).
                #
                # Otherwise, we only add the element once.

                # Standard addition
                hess_vals[flattened_row_col_index] += hess_contribution

                # Duplicate check and addition if
                # criteria is satisfied.
                if ((i != j) and (k != l)) and ((i == l) and (j == k)):
                    hess_vals[flattened_row_col_index] += hess_contribution

                hess_rows[flattened_row_col_index] = row
                hess_cols[flattened_row_col_index] = col

        elif self.objective_option == ObjectiveLib.determinant:
            # Grab inverse
            Minv = np.linalg.pinv(M)

            for current_differential in input_differentials_2D:
                # Row, Col and i, j, k, l values are
                # obtained identically as in the trace
                # for loop above.
                d1, d2 = current_differential

                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # New Formula (tested with finite differencing)
                # Will be cited from the Pyomo.DoE 2.0 paper
                hess_contribution = -(Minv[i, l] * Minv[k, j])

                # Since we are considering the full matrix in
                # this loop, we need to point the contribution
                # to the correct index for the symmetric FIM
                # Hessian.
                reordered_ijkl = self._reorder_pairs(i, j, k, l)
                d1_symmetric = (
                    self._param_names[reordered_ijkl[0]],
                    self._param_names[reordered_ijkl[1]],
                )
                d2_symmetric = (
                    self._param_names[reordered_ijkl[2]],
                    self._param_names[reordered_ijkl[3]],
                )

                # Identify what index of the symmetric FIM
                # Hessian arrays need to be updated
                row = max(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                col = min(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                flattened_row_col_index = (row + 1) * row // 2 + col

                # Hessian needs to be handled carefully because of
                # the ``missing`` components when only passing
                # a symmetric version of the FIM. For a more
                # detailed explanation, please see the trace
                # for loop above
                hess_vals[flattened_row_col_index] += hess_contribution

                # Duplicate check and addition
                if ((i != j) and (k != l)) and ((i == l) and (j == k)):
                    hess_vals[flattened_row_col_index] += hess_contribution

                hess_rows[flattened_row_col_index] = row
                hess_cols[flattened_row_col_index] = col

        elif self.objective_option in (
            ObjectiveLib.minimum_eigenvalue,
            ObjectiveLib.log_minimum_eigenvalue,
        ):
            # Grab eigenvalues and eigenvectors
            # Also need the min location
            all_eig_vals, all_eig_vecs = np.linalg.eigh(M)
            min_eig_loc = np.argmin(all_eig_vals)

            # Grabbing min eigenvalue and corresponding
            # eigenvector
            min_eig = all_eig_vals[min_eig_loc]
            min_eig_vec = np.array([all_eig_vecs[:, min_eig_loc]])
            if (
                self.objective_option == ObjectiveLib.log_minimum_eigenvalue
                and min_eig <= 0
            ):
                raise ValueError(
                    "log_minimum_eigenvalue requires a positive-definite "
                    "information matrix."
                )

            for current_differential in input_differentials_2D:
                # Row, Col and i, j, k, l values are
                # obtained identically as in the trace
                # for loop above.
                d1, d2 = current_differential

                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # For loop to iterate over all
                # eigenvalues/vectors
                hess_contribution = 0
                for curr_eig in range(len(all_eig_vals)):
                    # Skip if we are at the minimum
                    # eigenvalue. Denominator is
                    # zero.
                    if curr_eig == min_eig_loc:
                        continue

                    # Formula derived in Pyomo.DoE Paper
                    hess_contribution += (
                        1
                        * (
                            min_eig_vec[0, i]
                            * all_eig_vecs[j, curr_eig]
                            * min_eig_vec[0, l]
                            * all_eig_vecs[k, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )
                    hess_contribution += (
                        1
                        * (
                            min_eig_vec[0, k]
                            * all_eig_vecs[i, curr_eig]
                            * min_eig_vec[0, j]
                            * all_eig_vecs[l, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )

                if self.objective_option == ObjectiveLib.log_minimum_eigenvalue:
                    first_d1 = min_eig_vec[0, i] * min_eig_vec[0, j]
                    first_d2 = min_eig_vec[0, k] * min_eig_vec[0, l]
                    hess_contribution = (
                        hess_contribution / min_eig
                        - first_d1 * first_d2 / min_eig**2
                    )

                # Since we are considering the full matrix in
                # this loop, we need to point the contribution
                # to the correct index for the symmetric FIM
                # Hessian.
                reordered_ijkl = self._reorder_pairs(i, j, k, l)
                d1_symmetric = (
                    self._param_names[reordered_ijkl[0]],
                    self._param_names[reordered_ijkl[1]],
                )
                d2_symmetric = (
                    self._param_names[reordered_ijkl[2]],
                    self._param_names[reordered_ijkl[3]],
                )

                # Identify what index of the symmetric FIM
                # Hessian arrays need to be updated
                row = max(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                col = min(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                flattened_row_col_index = (row + 1) * row // 2 + col

                # Hessian needs to be handled carefully because of
                # the ``missing`` components when only passing
                # a symmetric version of the FIM. See trace for loop
                # for more detailed explanation
                hess_vals[flattened_row_col_index] += hess_contribution

                # Duplicate check and addition
                if ((i != j) and (k != l)) and ((i == l) and (j == k)):
                    hess_vals[flattened_row_col_index] += hess_contribution

                hess_rows[flattened_row_col_index] = row
                hess_cols[flattened_row_col_index] = col

        elif self.objective_option == ObjectiveLib.condition_number:
            # Hessian for log condition number has 4
            # terms. The first and third terms are
            # multiples of the second derivative of the
            # maximum and minimum eigenvalues, respectively
            # The other two are tensor products
            # of the first derivative of the maximum
            # eigenvalue with itself, and the minimum
            # eigenvalue with itself.
            #
            # Grab eigenvalues and eigenvectors
            # Also need the max and min locations
            all_eig_vals, all_eig_vecs = np.linalg.eig(M)
            min_eig_loc = np.argmin(all_eig_vals)
            max_eig_loc = np.argmax(all_eig_vals)

            # Grabbing min eigenvalue and corresponding
            # eigenvector
            min_eig = all_eig_vals[min_eig_loc]
            min_eig_vec = np.array([all_eig_vecs[:, min_eig_loc]])

            # Grabbing max eigenvalue and corresponding
            # eigenvector
            max_eig = all_eig_vals[max_eig_loc]
            max_eig_vec = np.array([all_eig_vecs[:, max_eig_loc]])

            for current_differential in input_differentials_2D:
                # Row, Col and i, j, k, l values are
                # obtained identically as in the trace
                # for loop above.
                d1, d2 = current_differential

                i = self._param_names.index(d1[0])
                j = self._param_names.index(d1[1])
                k = self._param_names.index(d2[0])
                l = self._param_names.index(d2[1])

                # For loop to iterate over all
                # eigenvalues/vectors for first
                # term (second derivative of
                # maximum eigenvalue)
                log_cond_term_1 = 0
                for curr_eig in range(len(all_eig_vals)):
                    # Skip if we are at the maximum
                    # eigenvalue. Denominator is
                    # zero.
                    if curr_eig == max_eig_loc:
                        continue

                    # Formula derived in Pyomo.DoE Paper
                    log_cond_term_1 += (
                        1
                        * (
                            max_eig_vec[0, i]
                            * all_eig_vecs[j, curr_eig]
                            * max_eig_vec[0, l]
                            * all_eig_vecs[k, curr_eig]
                        )
                        / (max_eig - all_eig_vals[curr_eig])
                    )
                    log_cond_term_1 += (
                        1
                        * (
                            max_eig_vec[0, k]
                            * all_eig_vecs[i, curr_eig]
                            * max_eig_vec[0, j]
                            * all_eig_vecs[l, curr_eig]
                        )
                        / (max_eig - all_eig_vals[curr_eig])
                    )

                # For loop to iterate over all
                # eigenvalues/vectors for third
                # term (second derivative of
                # minimum eigenvalue)
                log_cond_term_3 = 0
                for curr_eig in range(len(all_eig_vals)):
                    # Skip if we are at the minimum
                    # eigenvalue. Denominator is
                    # zero.
                    if curr_eig == min_eig_loc:
                        continue

                    # Formula derived in Pyomo.DoE Paper
                    log_cond_term_3 += (
                        1
                        * (
                            min_eig_vec[0, i]
                            * all_eig_vecs[j, curr_eig]
                            * min_eig_vec[0, l]
                            * all_eig_vecs[k, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )
                    log_cond_term_3 += (
                        1
                        * (
                            min_eig_vec[0, k]
                            * all_eig_vecs[i, curr_eig]
                            * min_eig_vec[0, j]
                            * all_eig_vecs[l, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )

                # Computing each term of the hessian formula
                # Second derivative of max eigenvalue term
                log_cond_term_1 = 1 / max_eig * log_cond_term_1

                # First derivative of max eigenvalue term
                log_cond_term_2 = (
                    1
                    / (max_eig**2)
                    * (max_eig_vec[0, l] * max_eig_vec[0, k])
                    * (max_eig_vec[0, j] * max_eig_vec[0, i])
                )

                # Second derivative of min eigenvalue term
                log_cond_term_3 = 1 / min_eig * log_cond_term_3

                # First derivative of min eigenvalue term
                log_cond_term_4 = (
                    1
                    / (min_eig**2)
                    * (min_eig_vec[0, l] * min_eig_vec[0, k])
                    * (min_eig_vec[0, j] * min_eig_vec[0, i])
                )

                # Combining all the components
                hess_contribution = (
                    log_cond_term_1
                    - log_cond_term_2
                    - log_cond_term_3
                    + log_cond_term_4
                )

                # Since we are considering the full matrix in
                # this loop, we need to point the contribution
                # to the correct index for the symmetric FIM
                # Hessian.
                reordered_ijkl = self._reorder_pairs(i, j, k, l)
                d1_symmetric = (
                    self._param_names[reordered_ijkl[0]],
                    self._param_names[reordered_ijkl[1]],
                )
                d2_symmetric = (
                    self._param_names[reordered_ijkl[2]],
                    self._param_names[reordered_ijkl[3]],
                )

                # Identify what index of the symmetric FIM
                # Hessian arrays need to be updated
                row = max(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                col = min(
                    self._fim_input_names.index(d1_symmetric),
                    self._fim_input_names.index(d2_symmetric),
                )
                flattened_row_col_index = (row + 1) * row // 2 + col

                # Hessian needs to be handled carefully because of
                # the ``missing`` components when only passing
                # a symmetric version of the FIM. See trace for loop
                # for more detailed explanation
                hess_vals[flattened_row_col_index] += hess_contribution

                # Duplicate check and addition
                if ((i != j) and (k != l)) and ((i == l) and (j == k)):
                    hess_vals[flattened_row_col_index] += hess_contribution

                hess_rows[flattened_row_col_index] = row
                hess_cols[flattened_row_col_index] = col
        else:
            ObjectiveLib(self.objective_option)

        n_fim_inputs = len(self._fim_input_names)
        packed_hessian = scipy.sparse.coo_matrix(
            (np.asarray(hess_vals), (hess_rows, hess_cols)),
            shape=(n_fim_inputs, n_fim_inputs),
        ).toarray()

        if self.fim_formulation == "fim":
            output_hessian = scipy.sparse.coo_matrix(packed_hessian)
        else:
            # The formulas above populate one triangle. The chain rules below
            # use the conventional full symmetric Hessian.
            packed_hessian = (
                packed_hessian + packed_hessian.T - np.diag(np.diag(packed_hessian))
            )
            M = np.asarray(self._get_FIM(), dtype=np.float64)
            gradient_matrix = self._objective_gradient_matrix(M)
            packed_gradient = self._pack_symmetric_gradient(gradient_matrix)

            if self.fim_formulation == "sensitivity":
                derivative = self._sensitivity_to_fim_jacobian()
                transformed_hessian = derivative.T @ packed_hessian @ derivative
                if not self.hessian_mode.startswith("gauss-newton"):
                    for measurement, weight in enumerate(self._measurement_weights):
                        block = np.zeros((self._n_params, self._n_params))
                        for fim_index, (row_name, col_name) in enumerate(
                            self._fim_input_names
                        ):
                            row = self._param_names.index(row_name)
                            col = self._param_names.index(col_name)
                            if row == col:
                                block[row, row] += (
                                    2.0 * weight * packed_gradient[fim_index]
                                )
                            else:
                                value = weight * packed_gradient[fim_index]
                                block[row, col] += value
                                block[col, row] += value
                        start = measurement * self._n_params
                        transformed_hessian[
                            start : start + self._n_params,
                            start : start + self._n_params,
                        ] += block
            else:
                _, shift_gradient, shift_hessian = self._shift_information(
                    self._get_raw_FIM()
                )
                diagonal_direction = np.asarray(
                    [float(row == col) for row, col in self._fim_input_names]
                )
                transformation = np.eye(n_fim_inputs) + np.outer(
                    diagonal_direction, shift_gradient
                )
                transformed_hessian = (
                    transformation.T @ packed_hessian @ transformation
                    + (
                        np.trace(gradient_matrix)
                        + self._shift_penalty_sign() * self.shift_penalty
                    )
                    * shift_hessian
                )

            output_hessian = scipy.sparse.coo_matrix(np.tril(transformed_hessian))

        # The ExternalGreyBoxModel contract requires the multiplier-weighted
        # output contribution to the Hessian of the Lagrangian.
        weighted_hessian = self._output_con_mult_values[0] * output_hessian
        if self.hessian_mode.endswith("projected-psd"):
            lower = weighted_hessian.toarray()
            full = lower + lower.T - np.diag(np.diag(lower))
            eigenvalues, eigenvectors = np.linalg.eigh(full)
            projected = (
                eigenvectors * np.maximum(eigenvalues, 0.0)
            ) @ eigenvectors.T
            weighted_hessian = scipy.sparse.coo_matrix(np.tril(projected))
        return weighted_hessian
