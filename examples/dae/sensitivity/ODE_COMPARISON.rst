ODE collocation and map-cost comparison
=======================================

G2, 30 October 2026. Results are in results/ode_comparison-3.8.1.json and
results/ode_comparison-3.8.2.json. The coarser-tolerance pilot is retained
separately in results/ode_comparison_pilot-3.8.2.json with gate_passed=false.
No production Simulator change was needed for this checkpoint.

Public compile and independent discrete equations
-------------------------------------------------

The common model is xdot=-k*x, qdot=x, x0=k*k, q0=0, with
J=2*x(1.2)+3*q(1.2)+k*k at k=0.7. The public compiled map uses a mutable
Param and symbolic IC composition. Both derivative modes agree with the
analytic total derivative 4.893876838802868 within 1.4e-10.

The independent PyNumero model uses a free Var for k so that the equation
Jacobian retains its parameter column. The nonlinear IC equation is part
of the residual system. At fixed k the state equations are linear; two
sparse linear updates solve them without calling an optimizer. Implicit
differentiation includes the initial-state equation and the explicit
objective term. The discrete derivative is checked against a four-step
finite-difference sweep of the discrete cost.

Three-point Radau meshes with 4, 8 and 16 elements give continuous derivative
errors of 1.67e-7, 5.32e-9 and 1.68e-10. The minimum discrete FD errors are
below 1.4e-10 and all state residuals are below 1e-10. This directly connects
the public compile result, exact continuous mathematics and a converging
discrete calculation. It does not equate a discretization derivative with
an adaptive integrator derivative at a finite mesh.

The ASL extension is an existing library, not a fresh build. Each report
records its absolute path and hash. ASL is required only for this optional
comparison and is not a dependency of compile() or its core tests.

Construction and execution measurements
---------------------------------------

The benchmark uses xdot=k*x+u, qdot=x, x0=1.3, q0=0 and k=-0.7. Controlled
maps receive constant amplitudes 0.1 in all intervals; uncontrolled maps
capture the same 0.1 input numerically. Both therefore represent the same
continuous dynamics. The controlled VJP also returns every amplitude
partial, so its derivative work is not identical to the uncontrolled map.

Construction medians use three compile calls, excluding Pyomo model creation
and Simulator construction. Execution medians use five calls after a warm-up;
derivative-function construction is recorded separately. Benchmark cases run
in bounded subprocesses. Final environment reports were generated serially
to avoid competing benchmark processes. Timings describe one machine and
are not unit-test thresholds or cross-platform guarantees.

Representative CasADi 3.8.2 medians at reltol=1e-11, abstol=1e-13, in ms:

.. list-table:: Compiled affine ODE map costs
   :header-rows: 1
   :widths: 20 20 15 15 15 15

   * - Observations
     - Control intervals
     - Compile
     - Primal
     - JVP
     - VJP
   * - 10
     - 0
     - 0.53
     - 0.16
     - 0.74
     - 0.73
   * - 10
     - 4
     - 1.17
     - 0.91
     - 4.33
     - 4.37
   * - 100
     - 100
     - 1.82
     - 8.39
     - 45.36
     - 48.01
   * - 400
     - 100
     - 2.20
     - 17.57
     - 105.84
     - 107.89
   * - 100
     - 400
     - 2.27
     - 18.13
     - 108.25
     - 108.22
   * - 400
     - 0
     - 0.52
     - 0.18
     - 0.82
     - 0.81

The controlled cost grows with the union of observations and switches.
Each normalized interval restarts CVODES, discarding multistep history.
Uncontrolled maps use one native multi-observation integration and are
much cheaper for this small model. At the tested sizes, construction remains
small and runtime growth is consistent with interval work rather than
explosive expression expansion. This does not measure large-state models,
memory scaling or production optimization workloads. No broad optimization
or alternate implementation is justified by this small benchmark alone.

Numerical accuracy under grid refinement
----------------------------------------

The initial pilot used reltol=1e-9, abstol=1e-11. Its 400-observation,
100-control-interval case has primal error 3.57e-7 and terminal parameter
adjoint error 3.08e-7, exceeding this checkpoint's 3e-7 accuracy target.
The 100-observation, 400-interval case also misses the primal target.
Those results remain visible with accuracy_passed=false; all experiments
completed, but the pilot did not pass the gate.

An explicitly requested tighter pair, reltol=1e-11 and abstol=1e-13, brings
all nine benchmark cases below 4.2e-9 for primal and derivative errors on
both versions. This is measured tolerance refinement, not a change to the
library defaults or a guarantee that a local tolerance bounds global map
error. More observation/control intervals add local integration errors.
Users should check convergence for the complete composed map and scale the
model appropriately. The implementation does not silently tighten settings
according to grid length.

Reproduction and gate
---------------------

Run from the repository root in each CasADi environment:

.. code-block:: sh

   PYTHONPATH="$PWD" <env-python> examples/dae/sensitivity/ode_comparison.py \
       --tolerance 1e-11 --output /tmp/ode-comparison.json

To reproduce the coarse pilot, set --tolerance 1e-9. Inspect gate_passed and
per-case accuracy_passed, not just the process exit code. If ASL is absent,
the report says available=false and the comparison gate remains unpassed;
core compile sensitivity tests can still run without ASL.

G2 passes for the recorded tight-setting matrix: the public derivative is
analytic, discrete derivatives agree with discrete FD, mesh refinement
converges, and construction/evaluation/derivative costs have been measured
with their limits stated. The pilot is qualified separately. Next is G3
ODE documentation, compatibility checks and provisional API closeout.
