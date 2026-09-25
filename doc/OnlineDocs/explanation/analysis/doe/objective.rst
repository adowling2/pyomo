.. _doe_objectives:

Objective Options
=================

.. note::

    Detailed descriptions and example code for the objective options in Pyomo.DoE will be added in a future update.

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
linear in the FIM and has a zero Hessian for every multiplier.
