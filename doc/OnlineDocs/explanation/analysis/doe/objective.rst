.. _doe_objectives:

Objective Options
=================

Grey-box Hessian interface
--------------------------

When using ``use_grey_box_objective=True``, the FIM metric is represented by
``FIMExternalGreyBox``. Its ``evaluate_hessian_outputs()`` method returns the
lower-triangular sparse Hessian multiplied by the output-constraint multiplier,
as required by the ``ExternalGreyBoxModel`` interface. The solver supplies this
multiplier through ``set_output_constraint_multipliers()`` when assembling the
Hessian of the Lagrangian.

For direct evaluation outside a solver, the multiplier defaults to one, giving
the unweighted metric Hessian. Call ``set_output_constraint_multipliers([1.0])``
to restore that behavior after a different multiplier has been set. Multipliers
do not change the metric value or its Jacobian; the pseudo-A-optimal metric is
linear in the FIM and has a zero Hessian with respect to FIM inputs for every
multiplier. With sensitivity inputs it is quadratic and generally has a nonzero
Hessian.

Grey-box inputs: information matrix or sensitivity Jacobian
----------------------------------------------------------

With ``use_grey_box_objective=True``, choose the external model's inputs with
``grey_box_fim_formulation``:

* ``"fim"`` (default): pass the upper triangle of the symmetric information
  matrix :math:`M`. The optimization model includes variables for its entries
  and equality constraints defining their values.
* ``"sensitivity"``: pass the measurement-by-parameter Jacobian :math:`J` and
  construct the information matrix inside the GreyBox:

  .. math::

     M(J) = J^\mathsf{T} W J + M_{\mathrm{prior}},
     \qquad W = \operatorname{diag}(1/\sigma_i^2).

The sensitivity formulation uses the same measurement errors, prior, parameter
scaling, and finite-difference scenarios as the FIM formulation. The prior must
be supplied in the same parameter coordinates and scaling as the Jacobian. It does not
select k_aug or change how response sensitivities are calculated. Inputs follow
the model's output order, then parameter order. The lifted FIM constraints are
deactivated; ``get_FIM()`` and the saved FIM results reconstruct information from
the current sensitivity variables. The inactive ``model.fim`` variables are not
updated during optimization and should not be used for reporting in this mode.

For :math:`n_p` parameters and :math:`n_y` measurements, the FIM formulation
passes :math:`n_p(n_p+1)/2` inputs, while the sensitivity formulation passes
:math:`n_y n_p`. The latter can therefore require a larger external Hessian.

All existing GreyBox criteria can use either input formulation. The default
objective remains D-optimality; selecting ``"minimum_eigenvalue"`` continues to
use raw E-optimality. The enum ``GreyBoxFIMFormulation`` is also exported from
``pyomo.contrib.doe`` for selecting ``fim`` or ``sensitivity``.

A finite symmetric positive-semidefinite prior is required for sensitivity
inputs. A positive-definite prior guarantees a positive-definite information
matrix for every finite :math:`J`. Without such a prior, the combined information
must have full rank wherever a criterion requiring positive definiteness is
evaluated. Passing :math:`J` prevents an independently varying lifted matrix
from becoming indefinite, but does not guarantee identifiability or convergence.
No eigenvalue clipping or regularization is applied.

Native logarithmic E-optimality
-------------------------------

Select ``objective_option="log_minimum_eigenvalue"`` to maximize

.. math::

   \phi(M) = \log\left(\frac{\lambda_{\min}(M)}{\lambda_{\mathrm{ref}}}\right).

This uses the natural logarithm and requires ``use_grey_box_objective=True``.
``grey_box_eigenvalue_reference`` supplies the finite positive reference
:math:`\lambda_{\mathrm{ref}}` (default 1). Changing the reference only adds a
constant to the objective; it does not change its derivatives or optimum.
Raw E and log-E rank positive-definite information matrices identically, but
their numerical optimization paths can differ. Saved ``log10 E-opt`` statistics
retain their existing base-10 convention and report the physical minimum
eigenvalue independently of this reference.

For example, given a labeled experiment and a compatible prior::

    doe = DesignOfExperiments(
        experiment=experiment,
        use_grey_box_objective=True,
        grey_box_fim_formulation="sensitivity",  # pass J; use "fim" to pass M
        objective_option="log_minimum_eigenvalue",
        grey_box_eigenvalue_reference=1.0,
        prior_FIM=prior,
    )
    doe.run_doe()
    information = doe.get_FIM()

Log-E values require a positive-definite information matrix. Its exact gradient
and Hessian additionally require a simple (nonrepeated) minimum eigenvalue.
Invalid domains and numerically indistinguishable repeated minima raise
``ValueError`` rather than supplying clipped values or undefined derivatives.
The logarithm does not smooth eigenvalue crossings. With FIM inputs, infeasible
solver iterates may leave the positive-definite domain even when the initial
matrix is valid; an exception is not a barrier or a solver recovery mechanism.

For a simple smallest eigenvalue :math:`\lambda` and normalized eigenvector
:math:`v`, the matrix gradient is :math:`G = vv^\mathsf{T}/\lambda`. In sensitivity
coordinates the gradient is :math:`2WJG`. The exact Hessian includes both the
transformed matrix Hessian and the second derivative of
:math:`J^\mathsf{T}WJ`. In particular, that second term need not vanish at
:math:`J=0`. The complete Hessian is multiplied by the output-constraint
multiplier supplied by PyNumero, as for the existing FIM formulation.

A small generic comparison with a known optimum is available in
``pyomo.contrib.doe.examples.grey_box_e_optimality``. It exercises raw/log E and
both input formulations and requires IPOPT, CyIpopt, and PyNumero ASL::

    python -m pyomo.contrib.doe.examples.grey_box_e_optimality
