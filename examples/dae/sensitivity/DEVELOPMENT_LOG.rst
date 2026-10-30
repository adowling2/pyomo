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

30 October 2026: G5 public index-one DAE bridge
-----------------------------------------------

Status: passed; this entry is included in the G5 implementation commit.
G4's native decision was committed in 2ad776409 before these production
changes. The public compile method now accepts explicit integrator='idas',
adds the native z0 guess input and consistent zf output, and retains the
ODE API. No compile keyword, result wrapper or derivative algorithm was
added. Existing simulate behavior is unchanged.

Sixteen new public-interface test methods cover the three native model
families, mixed observation weights, multiple directions, total nonlinear
IC/cost derivatives and four-step FD checks, both branches, guess invariance,
initial-only/sparse/dense grids, indexed ordering, empty/uninitialized
parameters, nonlinear parameter expressions used only in g, invalid and
singular systems, numerical-policy refusals and state/options preservation.
A ten-variable dense algebraic block checks the generic root/Jacobian path.

All three full DAE suites pass 283 tests and 271 subtests: Python 3.12.14
with CasADi 3.8.1/3.8.2 and Python 3.14.3 with 3.8.2. The lean check runs
145 tests: seven pass and 138 skip. Fresh coverage is 68/68 modified
executable lines and 42/42 DAE helper statements. The instrumented run's
existing semaphore cleanup diagnostic is recorded separately.

Shifted-time tests exposed a further native IDAS explicit-time adjoint
failure. The pure-CasADi reduction is dae_time_reproducer.py, with 14 cases
per version. An internal physical clock makes those equations autonomous
without changing public state layouts, tolerances or backend. Tight-setting
rate adjoint errors are below 2e-12. Disabling quadrature error control gives
about 5.23e-3 error and remains forbidden. Coarse clocked primals miss the
fixed accuracy target and remain explicitly flagged as such.

The default symbolic determinant check expands rapidly for dense matrices.
Selecting CasADi's symbolicqr determinant avoids that expansion while
preserving finite/nonzero local checks. The bounded guard reproducer retains
construction and regular/singular evaluation evidence. No timing threshold
was added to core tests. See the G5 addendum in DAE_DECISION.rst.

Five guide blocks, three analytic runnable examples and the focused
Sphinx guide/API build pass; downloaded example hashes are verified.
The new example is examples/dae/simulator_dae_sensitivity_example.py.
Black, typo checks, standalone RST parsing, diff and evidence-hash checks
pass. results/dae_compile_validation.json records this source snapshot.

G5 does not enable DAE controls; a clear refusal remains until G6. Next:
qualify fixed-grid DAE amplitudes, right-side algebraic observations and
phase consistency derivatives, then implement that bounded extension.

30 October 2026: G6 native control qualification
------------------------------------------------

Status: active; native qualification and switch contract passed, public
implementation pending. G5 is committed in 78818eef2.

DAE_CONTROLS.rst fixes right-held interval amplitudes, continuous differential
states, right-side algebraic observations at internal switches and the last
interval at the terminal time. An interval starting at the final observation
can immediately affect its algebraic value even before affecting dynamics.

The pure-CasADi dae_control_probe.py executes 17 bounded cases per version.
Exact affine and nonlinear phase formulas check all independent initial,
parameter and amplitude derivatives, two adjoint weights, identities,
branch/guess semantics, and FD sweeps on several grids. The main-setting
analytic/product maximum is below 4.7e-9 and FD minima below 4.6e-7 on both
versions. The coarse-tolerance sweep remains separately qualified.

The probe, convention note and versioned native records are included in
this intermediate G6 evidence commit. Production DAE controls are still
refused. Next action: extend the existing DAE helper and add independent
public tests, preserving the committed switch convention and ODE behavior.

30 October 2026: G6 public segmented DAE controls
-------------------------------------------------

Status: passed; this entry is included in the G6 implementation commit.
The convention/native gate was committed first in ddcc1d8d8. The public DAE
helper now reuses phase consistency and implicit observation reconstruction
with selected amplitudes. No compile keyword, result class, solver fallback
or change to simulate was added. The source diff is bounded to control
packing, phase selection and right-side algebraic observation reconstruction.

Ten new public test methods cover the planned switch/branch/IC/control
cases, derivative products and analytic/FD checks. Full suites pass 293
tests and 300 subtests in each of the three numerical environments. The lean
run passes seven tests with 148 skips. Modified executable coverage is 26/26;
the DAE helper is 60/60, and Simulator is 666/692 (96.24%). Instrumented
semaphore cleanup diagnostics remain recorded separately from passing tests.

A zero-rate affine case exposed a native IDAS adjoint iteration failure.
The one-interval pure native reduction and both versioned result files retain
that failure. Explicit tighter tolerances and normalized step control pass
the complete public zero-rate derivative test; production has no conditional
option change. See DAE_CONTROLS.rst for the failed and verified configurations.

Five guide blocks and four executable analytic examples pass on all three
environments. The focused Sphinx build passes with warnings treated as
errors and four download hashes verified. Black, typos, RST parsing and
diff checks pass. results/dae_controls_validation.json identifies this exact
snapshot and reproduction commands. Historical records are preserved.

Next action: G7 final support matrix, contribution/release draft and disclosure,
full standalone-goal requirement audit (including DAE collocation comparison),
then final local validation and a focused packaging commit. No publication
or remote CI is authorized or claimed.

30 October 2026: G7 independent DAE collocation comparison
----------------------------------------------------------

Status: active; the remaining standalone-goal DAE collocation check passes.
G6 is committed in 17a021b54. No production changes follow that checkpoint.

The optional dae_comparison.py retains the same original implicit DAE in
public and discrete models, with nonlinear ICs and a mixed differential/
integral/algebraic terminal cost. Independent implicit direct and transposed
adjoint solves agree to roundoff. Four-step discrete FD minima are below
8.3e-11. The total-cost gradient's continuous error decreases from 9.74e-7
to 3.12e-8 to 9.89e-10 on 4/8/16 Radau3 elements. The public map passes
all continuous primal/direct/adjoint checks below 5.6e-11 on both versions.

DAE_COMPARISON.rst and the two versioned records retain the scope, equations,
source/ASL hashes and commands. This optional comparison uses the existing
ASL library and no optimizer. Black, typos, RST and hash checks pass. These
files are included in an intermediate G7 evidence commit. Next: finish the
support/requirement matrix and local contribution packaging before declaring
G7 or the overall goal complete.

30 October 2026: G7 final local contribution readiness
------------------------------------------------------

Status: passed for the bounded local goal; this entry is included in the
final packaging commit. The DAE collocation evidence is committed in
311630d2f, after the passing G6 implementation in 17a021b54. Production,
core tests, user guide and runnable examples remain byte-identical to the
G6-validated versions; their full-suite/document results are reused without
claiming a new execution. The new comparison and audit scripts ran, and
all final notes, formatting, typo, whitespace and current hash checks pass.

CONTRIBUTION_READINESS.rst contains the final public support table, all
standalone-goal required-test families mapped to executed evidence, measured
native limits, release exclusions and a proposed review breakdown. Its
release text and PR-description draft remain local; human review-process
and legal acknowledgements are not asserted on behalf of the author.
EQUATION_SENSITIVITIES.rst describes the separate direct/adjoint equation
transformation track and explains why no transformation framework is needed
inside compile. AI assistance is disclosed for code, tests and documentation.

The historical hash audit recovers 108/110 repository source claims through
311630d2f. Two pre-first-commit interface-audit hashes are not recoverable from
retained Git snapshots. The audit flags them explicitly; old records are
unchanged. They are historical observations rather than final qualification
sources. Every current runtime/test/doc hash is verified. Final modified-line
coverage is 286/290 (98.62%), with the four defensive guards named in the
readiness note. Existing instrumented shutdown diagnostics remain qualified.

Checkpoint gates: G0 70fa93fb3; G1 3f66b87d2; G2 5adc03eac; G3 f2fcaa0cc;
G4 2ad776409; G5 78818eef2; G6 native ddcc1d8d8 and public 17a021b54;
G7 comparison 311630d2f and final packaging in this commit. A/B 74e516a11
and C f9a93159b remain the prior implementation checkpoints. G3 preceded
all production DAE work, and G4/G6 native decisions preceded their public
extensions. No gate relies on an expected native failure being counted as
accurate support.

All locally required G0-G7 work is complete. Linux/Windows and the remaining
supported-Python release matrix, whole-manual/external-link checks, human
review and any upstream submission remain explicitly unexecuted. They are
outside this local goal and require the appropriate infrastructure and/or
user authorization. No push, PR, issue or external message was sent.

30 October 2026: Single-PR draft and independent correctness audit
------------------------------------------------------------------

The user selected one targeted Pyomo.DAE PR covering the Simulator fixes,
ODE/DAE maps and both native sensitivity modes. PR_DESCRIPTION.md was drafted
in 35c6d4884 using the repository template, with numerical validation details
and provisional human-review/legal declarations. A separate read-only worker
audited the entire combined diff and found three correctness categories missed
by the earlier suites: endpoint/switch handling, algebraic determinant/rank
checks and complex zvode IC/input data.

All three were fixed in 8cbe3070d with nine additional collected regression
methods. The parent ran all three full suites: 302 tests and 320 subtests pass
per environment. The worker independently verified the fixes on both versions
and reran the stable-source full 3.8.2 suite with the same result. It reports
no outstanding production findings. Modified coverage is 304/308 (98.70%),
including all 24 audit-fix executable lines. PR_AUDIT.rst and
results/pr_audit_validation.json record the exact source and checks.

The Markdown draft now reflects the corrected implementation and current
validation, rather than historical G7 counts. Human review, cross-platform
qualification and submission declarations remain pending. All development
artifacts are retained until the user is satisfied and authorizes cleanup;
committed history will preserve that evidence. No PR was opened or published.
