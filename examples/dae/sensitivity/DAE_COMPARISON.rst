Public DAE and implicit collocation comparison
===============================================

G7, 30 October 2026. This closes the standalone goal's remaining refined
DAE collocation comparison. Production Simulator code is unchanged from
17a021b54. The optional research script dae_comparison.py uses PyNumero/ASL,
NumPy and SciPy. Core sensitivity tests and runnable examples do not need
ASL. No optimization solver is used.

The same original equations appear in both models:
xdot=-k*z, qdot=x, z=a*x, x0=k*k, q0=0. The cost is
J=2*x(T)+3*q(T)+4*z(T)+k*k+0.2*a*a, with T=1.2, k=0.7 and a=1.4.
The public IDAS map composes nonlinear initial states and the objective in
CasADi. The discrete model retains k/a as free Var columns and includes the
initial-state equation in the residual. It keeps the algebraic equation
rather than replacing z with a*x in the discrete system.

At fixed parameters, two sparse linear updates solve the collocation state
equations. Implicit direct derivatives solve R_y*S=-R_p; an independent
transposed solve gives lambda from R_y^T*lambda=J_y^T and gradient
J_p-R_p^T*lambda. Both include the algebraic terminal cost and the IC chain
rule. Four-step finite differences differentiate the solved discrete cost.
Closed-form elimination is used only in the independent continuous oracle.

Both CasADi versions give the following dimensionless errors, normalized
by one plus the absolute reference value:

.. list-table:: Radau3 mesh refinement
   :header-rows: 1

   * - Elements
     - Total cost gradient versus continuous
     - Best discrete FD versus implicit
     - Direct versus discrete adjoint
   * - 4
     - 9.74e-7
     - 1.66e-11
     - 1.48e-16
   * - 8
     - 3.12e-8
     - 8.28e-11
     - 3.85e-16
   * - 16
     - 9.89e-10
     - 1.59e-11
     - 4.44e-16

All recorded continuous terminal, cost, terminal-sensitivity and cost-gradient
errors decrease at every refinement. The finest terminal-sensitivity error
is below 5.8e-10. State residuals are below 1e-10. The public compiled map's
largest analytic error across primal, full terminal JVPs and cost JVP/VJP
checks is below 5.6e-11 on both versions. This connects public DAE derivatives,
continuous mathematics and converging discrete equations; finite-mesh
collocation derivatives are not equated with adaptive IDAS derivatives.

Reproduce in either dependency environment:

.. code-block:: sh

   PYTHONPATH="$PWD" <env-python> examples/dae/sensitivity/dae_comparison.py \
       --output /tmp/dae-comparison.json

The script fails clearly if ASL is absent. Versioned results are
results/dae_comparison-3.8.1.json and results/dae_comparison-3.8.2.json. They
record source and loaded ASL hashes, actual environment versions, independent
errors and FD sweeps. The ASL binary is the existing externally discovered
library; this is not fresh-build or cross-platform qualification. This
comparison covers the autonomous linear DAE with nonlinear ICs; controlled
and nonlinear branch correctness has separate analytic/FD public evidence
in DAE_CONTROLS.rst and DAE_DECISION.rst.
