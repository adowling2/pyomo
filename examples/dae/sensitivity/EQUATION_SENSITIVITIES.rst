Explicit sensitivity-equation track: design boundary
=====================================================

G7, 30 October 2026. Simulator.compile delegates adaptive first-derivative
integration to CasADi/CVODES/IDAS. It supports direct and adjoint products;
it does not add a Pyomo transformation that returns sensitivity constraints
or adjoint variables. The standalone goal identifies that transformation as
a possible second deliverable, not a prerequisite for the native map.

Independent equations
---------------------

For xdot=f(t,x,z,p,u), g(t,x,z,p,u)=0 on a regular index-one branch,
parameter direction v gives differential and algebraic sensitivities S,Z:

.. code-block:: text

   Sdot = f_x S + f_z Z + f_p v
   0    = g_x S + g_z Z + g_p v
   S(t0) = x0_p v

For a control-amplitude direction, add f_u du and g_u du on its fixed
interval. The algebraic initial derivative satisfies this differentiated
residual. It is not a derivative with respect to the Newton guess. At fixed
switches S is continuous; Z can jump with the input. Observation/terminal
algebraic costs must include their direct parameter and amplitude terms.

A locally reduced adjoint can be expressed without forming matrix inverses.
Solve g_z Z_x=-g_x and g_z Z_p=-g_p, and form reduced derivatives
fbar_x=f_x+f_z Z_x, fbar_p=f_p+f_z Z_p. For
J=Phi(x(T),z(T),p)+integral L(t,x,z,p)dt:

.. code-block:: text

   -lambda_dot = fbar_x^T lambda + Lbar_x^T
   lambda(T)   = Phi_x^T + Z_x(T)^T Phi_z^T
   grad_p J    = Phi_p^T + Z_p(T)^T Phi_z^T
                 + x0_p^T lambda(t0)
                 + integral (fbar_p^T lambda + Lbar_p^T) dt

Here Lbar derivatives also include implicit z dependence. A fixed-grid
amplitude uses analogous Z_u and fbar_u terms on its interval. Intermediate
observation costs induce adjoint jumps; a right-side algebraic observation
uses the new amplitude's direct term. State resets, switching-time
sensitivities and branch transitions require different mathematics and
remain excluded. These formulas assume real, sufficiently smooth residuals
and a fixed nonsingular branch.

Bounded follow-up proposal
--------------------------

If users need to inspect, collocate or optimize the sensitivity equations,
first prototype one selected parameter direction on the existing scalar ODE
and regular index-one DAE examples. Preserve the original model and return
ordinary Pyomo components, with explicit initial derivative equations and
stable component mapping. Check the transformation against hand equations,
the native compiled map and the discrete Radau oracle before generalizing.
An adjoint transformation would additionally need an objective/observation
contract and backward jump/terminal conditions; it should follow the direct
prototype, not be hidden inside compile.

The existing test_explicit_sensitivity_equations in test_simulator_compile.py
already integrates hand-written ODE tangent equations as an independent check.
DAE_COMPARISON.rst provides separate implicit discrete direct/adjoint solves.
Neither is advertised as a general Pyomo transformation. No second adaptive
integrator, IC callback framework or result class is warranted for this goal.
