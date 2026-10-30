Development checkpoint log
==========================

30 October 2026: G0 baseline and execution record
-------------------------------------------------

Status: passed; this entry and DEVELOPMENT_GOAL.rst are included in the G0
plan commit. Next checkpoint: G1 ODE numerical qualification.

The starting tree is clean at f9a93159b on pyomo-dae-sensitivity. Work package
C supports fixed-grid control amplitudes and total derivatives through
symbolically composed initial conditions. Work package B and the earlier
Simulator fixes are committed in 74e516a11.

Existing evidence: results/controls_validation.json records 265 tests and
232 subtests passing with each of CasADi 3.8.1 and 3.8.2 on Python 3.12.14,
macOS ARM64. Seven tests pass and 120 skip in the lean optional-dependency
check. Four guide blocks and both ODE examples pass on both versions.
These are historical baseline results, not proof of future changes.

Known issues to resolve or qualify: controlled large-grid cost is unmeasured;
the collocation probe does not yet call compile(); a recorded native BDF
adjoint failure occurs for the 1000:1 decay case; native algebraic-terminal
DAE adjoints remain inaccurate in supplied probes. Current compile() refuses
DAEs, vector absolute weights and switch-time derivatives.

The goal has explicit gates G0-G7, acceptance thresholds, failure paths,
commit boundaries and a local-versus-remote validation distinction. Production
DAE changes are gated on ODE closeout and a native initialization/output
contract. This plan does not authorize pushing or external messages.

30 October 2026: G1 ODE numerical qualification
-----------------------------------------------

Status: passed. The G0
plan commit is 70fa93fb3. Numerical evidence and two new test methods are
included in the upcoming G1 checkpoint commit; production code is unchanged.

The nonlinear shared-control model has states six orders of magnitude apart.
Exact state/derivative checks, two-direction products and scaled identities
pass on both versions; tolerance refinement reduces errors below 2.2e-10.
The total IC/control composition passes analytic checks and a four-step FD
sweep, with minimum maximum derivative error below 2.0e-10.

The stiff-adjoint failure reduces to a scalar native equation; default BDF
primal/direct calculations pass but reverse integration fails identically
through native and public graphs. Explicit Adams passes. Checkpoint spacing
one helps the original coupled model but not the scalar reduction. See
ODE_NUMERICS.rst and results/ode_numerics-<version>.json for the 24-case
per-version matrix, options, exceptions, timeouts and source hashes.

Commands: run ode_qualification.py once per environment; run pytest on the
full pyomo/dae/tests directory. CasADi 3.8.2 passes 267 tests and 236 subtests.
CasADi 3.8.1 also passes 267 tests and 236 subtests in an isolated working
directory. Black, typo checks, standalone RST parsing and diff checks pass.
The first parallel 3.8.1 full-suite run collided with the other run's
existing diffset.dat fixture and had one FileNotFoundError. The rerun uses
an isolated working directory. Future full-suite matrix runs must isolate
working directories or run sequentially.

Next action: G2 public-interface collocation comparison and controlled-grid
cost measurements. Current source remains the C implementation; the new
numerical qualification needed no production fix.
