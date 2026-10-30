ODE closeout and index-one DAE development goal
===============================================

Started 30 October 2026 on branch ``pyomo-dae-sensitivity``.
Baseline: ``f9a93159b`` (work package C), following ``74e516a11`` (A/B).
The current implementation and historical evidence are described in
``API_DECISION.rst`` and ``results/controls_validation.json``.

Objective
---------

Finish a small, reviewable ODE Simulator.compile contribution, then develop
and qualify a semi-explicit index-one DAE extension. Preserve the corrected
Simulator behavior and the native CasADi Function interface. Verify direct
and adjoint products against independent mathematics before advertising a
capability. Commit completed checkpoints as work proceeds.

Plan for roughly 15-32 hours of focused development, spread over several
sessions or days. These are effort estimates, not deadlines or a reason to
run unnecessary experiments. Native dependency failures may require a later
upstream resolution. Record that limitation rather than forcing a result.

Working rules
-------------

* Keep production changes surgical. Reuse existing conversion and graph
  assembly. Do not add a result/registry class, an IC callback framework or
  a numerical integration algorithm merely to support this work.
* Preserve model values, fixity, activation, caller options and prior
  simulation results on success and failure. Correct demonstrated bugs;
  compatibility does not require preserving an incorrect derivative.
* Keep quadrature error control required. Retain scalar tolerances and the
  verified version policy until a separate experiment justifies a change.
  Never hide a failure by relaxing all tolerances, disabling error control,
  changing reference values, skipping a supported test or switching solvers
  silently.
* Use analytic checks first, independent discrete/finite-difference checks
  second, and forward-adjoint identities as an additional check. Agreement
  between two derivative modes alone is insufficient.
* Test CasADi 3.8.1 and 3.8.2 in separate environments. Record versions,
  options, source hashes, commands, normalized errors and process outcomes.
  Known native failure experiments run in bounded subprocesses, with core
  dumps disabled. Expected-failure observations are not passing capability
  claims.
* Follow the repository contribution guide, Black, typo checks, guarded
  optional imports and pyomo.common.unittest. Use meaningful tests without
  mocks. Retain AI-assistance disclosure for any future submission.
* Make a focused commit at each passing checkpoint or a useful, reproducible
  dependency investigation. Separate experimental evidence from production
  changes. Do not amend completed commits, rewrite history, push, publish a
  PR or contact maintainers without further authorization.

Checkpoints and gates
---------------------

G0: Baseline and execution record (0.5-1 hour)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deliver this goal and ``DEVELOPMENT_LOG.rst``. Record the clean starting
commit, existing test results, numerical policies and unresolved issues.
Define where each later checkpoint will store reproducible evidence.

Gate: the plan distinguishes existing functionality, experiments and public
support. Every stage has an acceptance test, a failure path and a commit
boundary. Start G1 after committing the plan.

G1: ODE numerical qualification (1-3 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a nonlinear controlled ODE with a closed-form reference and state scales
separated by at least six orders of magnitude. Check primal states, x0/p/u
products, several observations, shared parameter dependencies and both
forward and reverse products. Sweep scalar tolerances and scaled directions.
Explain nondimensionalization as the current way to handle different units.

Reduce the existing 1000:1 fast/slow BDF-adjoint failure enough to distinguish
native integrator limitations from Pyomo conversion or composition. Preserve
an executable failing case and a tested alternative configuration without
calling that alternative universally suitable for stiff systems.

Gate: scaled primal and first-derivative errors at the tight setting are
below 3e-7; at least one informative tolerance sweep improves error by ten
or more before reaching the numerical floor. Finite differences across at
least four steps corroborate selected total derivatives with error below
1e-6. Multi-direction products satisfy scaled forward-adjoint identities
within 3e-7. The stiff failure has an explicit classification and bounded
reproducer. All existing DAE tests pass on both CasADi versions.

Failure path: reduce the failing model and inspect native CasADi behavior;
fix only a demonstrated bridge bug. Do not add vector tolerances or a new
solver abstraction to make this gate pass. Commit tests, necessary fixes
and ``results/ode_numerics-<version>.json``.

G2: Public-interface collocation and cost checks (2-4 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connect the independent Radau collocation probe to the actual compiled ODE
map. Use the same model, parameter/IC dependence, output weights and cost.
Compare discrete sensitivities, finite differences of the discrete problem,
continuous analytic sensitivities and compile() JVP/VJP values. Keep ASL
optional and separate from core sensitivity tests.

Measure compilation, primal evaluation, JVP and VJP times as observation
and control interval counts increase. Use warm-up and repeated timings;
report medians, problem sizes and environment. Separate construction from
execution, and compare the controlled case with a comparable uncontrolled
map. Begin with small grids, then at least one grid with 100 or more
intervals; bound individual experiments and stop enlarging failing cases.

Gate: discrete AD/implicit and discrete FD derivatives agree within 1e-7;
continuous sensitivity error decreases under 4/8/16-element refinement and
is below 3e-7 at the finest mesh; the public JVP/VJP satisfies the continuous
analytic check. Performance records explain the observed scaling without
an assertion that timings are portable. No benchmark becomes a timing-based
unit test. If growth is excessive, identify its source before any production
optimization. Commit the comparison, benchmark and
``results/ode_comparison-<version>.json``.

G3: ODE closeout and provisional API freeze (1-3 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Review the entire ODE change against the contribution guide. Build the
relevant Sphinx documentation, execute examples/snippets, and run the full
DAE suite on both versions plus optional-dependency absence checks. Inspect
modified-line coverage; aim for at least 90%, meeting the project's 70%
minimum. Any unreachable defensive refusal is identified explicitly.
Add a short ODE closeout note and a proposed contribution breakdown separating
legacy Simulator fixes, the feature and research-only evidence. Prepare
release-note text in the repository's existing convention if applicable.

Gate: G1 and G2 evidence passes, no known bridge correctness defect remains,
all available local checks pass, and every support claim identifies its
version/model/output limits. The ODE API is provisionally frozen. Record
which supported Python/platform combinations have actually run. Unavailable
remote CI is pending release evidence, not a fabricated pass and not a bar
to local DAE research. Do not push merely to obtain CI.

Commit the ODE closeout. Production DAE changes may begin only after G3.
Native DAE experiments may be prepared earlier but must not divert effort
from the ODE gates.

G4: Native DAE initialization and output contract (3-6 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Investigate xdot=f(t,x,z,p), 0=g(t,x,z,p) with nonsingular g_z on the selected
branch. Begin without controls. Use the linear system xdot=-k*z, z=a*x
and the nonlinear positive branch z*z=x, including x0(theta), an integral
state and explicit parameter terms in the cost.

Distinguish independent differential initial states from algebraic solver
guesses. Establish consistent initialization and its total derivative.
Check that changing a guess within the same branch does not change the
physical answer or derivative. Test branch dependence, invalid guesses,
singular algebraic Jacobians and parameter values that invalidate a branch.
Do not promise global index detection from one local Jacobian check.

Build a native IDAS support matrix: primal x/z, forward x/z, adjoint terminal
x, integral costs, terminal z and mixed observation weights, each with one
and several directions. Sweep tolerances and FD steps for the unresolved
algebraic-terminal discrepancy. Evaluate a general differentiable implicit
algebraic reconstruction if needed, with branch/residual/conditioning
checks. Compare against the original residual equations, not a model-specific
gradient patch. Write ``DAE_DECISION.rst`` before extending compile().

Gate: initialization semantics and input/output ordering are explicit;
every proposed supported mode meets independent analytic and FD checks;
errors in consistent algebraic residuals are measured. A generic reconstruction
must also differentiate correctly at the initial time and include direct
parameter terms. Unsupported dependency behavior has a small isolated
reproducer and a reasoned path forward. Scalar/vector tolerance policy and
version restrictions are explicit.

Failure path: continue native reductions; do not expose a raw algebraic
output whose permitted adjoint seeds cannot be enforced by the returned
Function. Prefer an upstream fix or a general verified reconstruction before
narrowing useful mathematics to accommodate a bug. If no small correct path
is available, document a DAE design checkpoint needing a decision rather
than claiming implementation is complete. Commit the matrix and design.

G5: Minimal autonomous DAE bridge (3-6 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement only the DAE contract justified by G4. Reuse Simulator's analysis
and expression conversion, retain selected parameters in both f and g,
and keep differential ICs independent. Introduce only the input/output or
integrator selection changes required by the written contract. A guess
must never masquerade as an independent physical IC.

Gate: public-interface tests reproduce the G4 analytic/FD results for linear
and nonlinear DAEs, x0(theta), integral and terminal costs, all advertised
algebraic outputs and multiple directions. Invalid/singular cases fail
clearly without changing model state. Parameter selections used only in g
remain symbolic. Existing ODE and simulate tests pass on both versions.
Source complexity remains comparable to the bounded task; reconsider the
contract if it requires a broad rewrite. Commit the bridge and tests.

G6: Segmented DAE controls (2-5 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the verified DAE map to fixed-grid amplitudes. Differential states
remain continuous at a switch; algebraic states may jump and must satisfy
the new interval's input. Specify left/right algebraic observation values,
including initial/final observations, before coding. Propagate consistency
and its derivatives through the phase chain.

Gate: an exactly reducible controlled index-one DAE passes primal, direct
and adjoint checks for x, every advertised z output, p, x0 and amplitudes;
state/integral cost derivatives agree with analytic segment composition and
FD sweeps. Test switches, sparse observations, future inputs, nonlinear
branches and failed consistency. No regression in ODE controls. If G4 does
not support the needed algebraic derivative operation, keep this extension
experimental rather than bypassing that gate. Commit the bounded extension
or a precise dependency limitation record.

G7: Final contribution readiness (2-4 hours)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deliver an executable DAE example, final support matrix, migration-free API
documentation, updated disclosure and release-note/PR-description draft.
Re-run appropriate checks for the final diff and verify evidence hashes.
Keep historical A/B/C and later checkpoint records clearly labeled.
Cross-platform release qualification remains a separate recorded gate if
its execution requires unavailable infrastructure or a user-authorized push.

Gate: every published capability has independent tests through the public
interface; native failures cannot be mistaken for verified support; all
available tests and document builds pass; ODE behavior remains stable.
No unapproved publication is needed to finish the local implementation.
Commit final packaging. Mark this goal complete only when the bounded
implementation and all locally required gates are satisfied. A blocked DAE
gate is an outstanding item, not successful completion.

Checkpoint protocol
-------------------

After each meaningful result, update DEVELOPMENT_LOG.rst with the gate,
status (not started, active, passed, or needs attention), concrete evidence,
commands, source commit/hash, unresolved issues and next action. Each entry
names its checkpoint commit once available, or states that the entry is
included in that commit; do not create commits solely to insert their own
hash. Checkpoint completion is based on evidence, not elapsed time.

When a gate fails, remain at that stage, reduce the case and record the
outcome. Continue independent useful work. Escalate a user decision only
when mathematical scope or a public API choice cannot be resolved within
this goal. Report concise progress during active work. Commit useful
validated units frequently; do not accumulate days of unrelated changes in
one commit or commit a failing capability as if supported.

Deferred work
-------------

High-index DAEs, event/switch-time derivatives, resets, second derivatives,
an adaptive sensitivity integrator, general optimal-control solvers and a
separate explicit Pyomo direct/adjoint equation transformation are outside
this goal. Scalar Var selection, native quadrature outputs and vector
weights need their own bounded justification; they are not prerequisites
for the existing ODE contract or a truthful index-one DAE investigation.

References
----------

* Repository contribution guide: doc/OnlineDocs/contribution_guide.rst.
* Repository development principles: doc/OnlineDocs/principles.rst.
* Original handoff: STANDALONE_GOAL.md in the 2026-10-30 research folder.
* Native integration and sensitivity documentation:
  https://web.casadi.org/docs/#initial-value-problems-and-sensitivity-analysis.
