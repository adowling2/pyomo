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

Status: passed. The G0 plan commit is 70fa93fb3. Numerical evidence and two
new test methods are committed in 3f66b87d2; production code is unchanged.

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

30 October 2026: G2 collocation and cost comparison
---------------------------------------------------

Status: passed. G1 is committed as 3f66b87d2.
The G2 script, reports and interpretation are committed in 5adc03eac.
Production Simulator code is unchanged.

The public map and independently differentiated collocation equations use
the same nonlinear IC, integral state and explicit cost. Direct and adjoint
continuous derivatives agree with the exact value within 1.4e-10. Radau
4/8/16-element derivative errors decrease from 1.67e-7 to 5.32e-9 to
1.68e-10; discrete FD agreement is below 1.4e-10. Existing ASL binary path
and hash are recorded; no optimizer was used.

Nine grid sizes cover 10-400 observations and 0-400 control intervals.
The coarse pilot exposes accumulated integration error above 3e-7 and
retains gate_passed=false. Explicit tolerance refinement to 1e-11/1e-13
passes every recorded primal/JVP/VJP check below 4.2e-9 on both versions.
Final timing matrices run serially after exploratory parallel runs; results
show controlled runtime increasing with interval work and much cheaper
uncontrolled multi-observation calls. See ODE_COMPARISON.rst for the measured
cost and accuracy limits. No default or algorithm was changed.

Command: run ode_comparison.py once per environment with --tolerance 1e-11,
plus a retained 3.8.2 --tolerance 1e-9 pilot. Next action: G3 local Sphinx
build, final ODE suite/optional-dependency checks and contribution packaging.
Black, typo checks, RST parsing, diff checks and source-hash verification pass.

30 October 2026: G3 ODE closeout
--------------------------------

Status: passed; this entry is included in the G3 checkpoint commit.
G2 is committed as 5adc03eac.
The ODE API is unchanged and the checkpoint's guide update explains local
tolerance accumulation, dense-grid cost and consistent state/seed scaling.

All three full suites pass 267 tests and 236 subtests: Python 3.12.14 with
CasADi 3.8.1/3.8.2, plus Python 3.14.3 with 3.8.2. The lean environment
runs 129 tests: seven pass and 122 skip. Fresh modified-line coverage is
208/212 (98.1%). The instrumented run's multiprocessing shutdown diagnostic
is retained separately from the successful numerical checks.

The focused Sphinx 9.1.1 build passes with warnings treated as errors.
check_ode_docs.py reproduces its temporary-copy setup and verifies example
download hashes, all four guide snippets and both analytic examples.
No source configuration changes were made for unrelated theme/bibliography
warnings in exploratory builds. See ODE_CLOSEOUT.rst for the exact build
scope, contribution breakdown, release-note draft and remaining remote-CI
qualification. No push, PR or external message has been sent.

Packaging/style/hash checks pass; results/ode_closeout.json records current
sources and commands. Next action: begin native G4 DAE experiments and write
DAE_DECISION.rst before any DAE production code.

30 October 2026: G4 native DAE contract
---------------------------------------

Status: passed as a native design gate; this entry is included in the G4
checkpoint commit. G3 is committed as f2fcaa0cc. Production Simulator code still rejects DAEs. The native
experiment runner imports only CasADi/NumPy; no Pyomo, ASL or optimizer.

There are 58 isolated experiments per version, including linear, nonlinear
and two-algebraic-variable residuals; one/multiple direct and adjoint modes;
nonlinear IC composition; integral/terminal/mixed costs; four-step FD sweeps;
branch and guess changes; initial-only maps; invalid/nonfinite inputs and
structural/local singularity. Raw algebraic-output adjoints are inaccurate,
and a multi-observation native map can fail at adjoint weight changes.

Generic implicit reconstruction fixes the recorded terminal output path.
Normalized observation-interval composition also resolves the recorded
mixed-weight failures. Both mechanisms reuse native solvers. All proposed
modes pass on all three models at 1e-10/1e-12 and 1e-11/1e-13 settings on
both versions, with analytic/product errors below 1e-9. Initial state values
are unchanged; guess derivatives are locally zero on either branch.

A 1e-12 coupled total-cost adjoint still fails native consistency, and the
near-singular model fails native integration. These are retained as explicit
limitations. The design does not silently relax tolerances or restrict away
algebraic objectives to preserve an incorrect output derivative.

DAE_DECISION.rst defines the proposed z0 guess input, consistent zf output,
initialization/output mathematics, required IDAS options and minimal G5
boundary. It must be committed before production changes. Next action:
verify the G4 aggregate, commit the native decision, then implement G5 with
public-interface regression tests.
