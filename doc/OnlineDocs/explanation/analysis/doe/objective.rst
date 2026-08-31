.. _doe_objectives:

Objective Options
=================

.. note::

    Detailed descriptions and example code for the objective options in Pyomo.DoE will be added in a future update.

GreyBox FIM formulations
------------------------

When ``use_grey_box_objective=True``, ``grey_box_fim_formulation`` selects the
variables passed to the external objective and the treatment of an indefinite
Fisher information matrix:

``fim``
    Retains the original lifted-FIM behavior. The upper triangle of the Pyomo
    ``fim`` variable is passed to the external model. This option is the default
    and provides a benchmark for comparing alternative formulations. It does
    not enforce positive definiteness. In particular, the D-optimality output
    uses the logarithm returned by ``numpy.linalg.slogdet``; the determinant
    sign is not part of the output.

``sensitivity``
    Passes the sensitivity matrix :math:`J` to the external model, which forms
    :math:`J^T W J + F_{\mathrm{prior}}` internally. The lifted FIM variables
    and constraints are excluded from the optimization problem. This matrix is
    positive semidefinite when :math:`W` and :math:`F_{\mathrm{prior}}` are
    positive semidefinite, and is positive definite when the combined
    information has full rank.

``softplus_exact``
    Passes the lifted FIM and evaluates the objective at
    :math:`F + sI`, where

    .. math::

       s = \frac{1}{\beta}\log\left(1 +
           \exp\left(\beta(\epsilon-\lambda_{\min}(F))\right)\right).

    This option uses analytic first- and second-order derivatives of a simple
    minimum eigenvalue. Its Hessian is undefined when the minimum eigenvalue is
    repeated; use ``softplus_smooth`` in that case.

``softplus_smooth``
    Replaces the exact minimum eigenvalue above with the smooth spectral
    approximation

    .. math::

       -\tau\log\sum_i\exp(-\lambda_i(F)/\tau).

    This variant remains differentiable at repeated eigenvalues.

The two shifted formulations penalize the required shift. The penalty is
subtracted from maximized criteria and added to minimized criteria. Configure
the eigenvalue target, softplus sharpness, smooth-minimum temperature, and
penalty weight with ``grey_box_eigenvalue_floor``,
``grey_box_softplus_beta``, ``grey_box_softmin_temperature``, and
``grey_box_shift_penalty``, respectively. The penalty weight is not invariant
to arbitrary rescaling of the FIM or objective, so it should be selected and
reported with the model scaling.

The small :mod:`pyomo.contrib.doe.examples.grey_box_spd_example` benchmark
isolates the indefinite-matrix failure mode in one design variable. It can be
run directly to compare the original lifted formulation with the penalized
smooth-shift formulation.

GreyBox Hessian safeguards
--------------------------

``grey_box_hessian_mode`` controls the curvature returned by the external
objective. Its default, ``exact``, returns the analytic Hessian. With the
``sensitivity`` FIM formulation, ``gauss-newton`` retains the exact gradient
but drops the second derivative of the quadratic map
:math:`J\mapsto J^T W J`. The resulting approximation is

.. math::

   \left(\frac{\partial F}{\partial J}\right)^T
   \nabla_F^2 f
   \left(\frac{\partial F}{\partial J}\right).

For a convex canonical FIM metric, this approximation is positive
semidefinite. ``projected-psd`` instead projects the multiplier-weighted exact
GreyBox Hessian onto the positive semidefinite cone.
``gauss-newton-psd`` combines both operations. These modes safeguard only the
GreyBox objective contribution; they do not modify curvature from the
experiment model's dynamic or algebraic constraints.

Softplus result diagnostics
---------------------------

Softplus formulations record both the raw and effective FIM, their
eigenvalues, the terminal diagonal shift and its activity, the penalty
contribution, and the local marginal objective benefit of an additional
diagonal shift. The reported ``GreyBox Shift Penalty Margin`` is the configured
penalty coefficient minus that marginal benefit. A negative margin indicates
that the regularized criterion locally rewards additional shift more strongly
than the objective penalizes it. Penalty adequacy is criterion- and
scale-dependent; it should not be inferred from the shift magnitude alone.
