Differentiable Simulator contribution readiness
================================================

G7, 30 October 2026. Local implementation is complete through G6 commit
17a021b54; the independent DAE collocation comparison is committed in
311630d2f. This note is the final local packaging gate. It is not an upstream
submission or a claim of maintainer approval. Historical A/B/C and G0-G6
records retain their original source snapshots and test counts.

Post-qualification review is active. The user selected one targeted Pyomo.DAE
PR covering the bug fixes, ODE/DAE maps and both sensitivity modes. The current
Markdown draft is PR_DESCRIPTION.md at the repository root. An independent
full-diff audit has reproduced endpoint, algebraic-guard scaling and complex-IC
regressions that the recorded suites missed. These require fixes and regression
tests before submission readiness can be claimed. Development artifacts remain
in the tree during this review.

Public contract and measured support
-------------------------------------

The additive CasADi-only Simulator.compile method returns a native Function.
Its keyword signature, ODE defaults and the existing simulate/initialize_model
interfaces remain stable. Selected mutable scalar ParamData entries remain
symbolic; unselected constants are captured at compile time. Explicit input
order, documented state order and native named seeds provide JVPs/VJPs.
Total derivatives use ordinary symbolic composition of x0, p, u and costs.
The caller retains the fixed observation grid and model-component selection;
there is no metadata/result wrapper or hidden synchronization with model edits.

.. list-table:: Verified public first-derivative paths
   :header-rows: 1
   :widths: 26 37 37

   * - Path
     - Support
     - Independent evidence
   * - ODE, integrator=cvodes
     - x0/p, multiple observations and directions
     - Exponential, coupled matrix exponential, nonlinear, stiff and scaled tests
   * - ODE fixed-grid controls
     - x0/p/u, shared designs and total IC derivatives
     - Exact affine/rational phase formulas, FD sweeps and duality
   * - DAE, explicit integrator=idas
     - x0/p, consistent xf/zf, integral and algebraic terminal costs
     - Linear/nonlinear/coupled residuals, exact partials and total FD sweeps
   * - DAE initial consistency
     - z0 is a branch guess; its local physical derivative is zero
     - Both branches, guess invariance, invalid/singular cases, no IC adjustment
   * - DAE fixed-grid controls
     - Continuous x, right-side z at internal switches, last amplitude at terminal
     - Affine/nonlinear phases, full amplitude products, shared IC/control costs
   * - Explicit time
     - Physical time in f/g and shifted domains
     - Polynomial oracles; DAE internal clock hidden from public states
   * - Integral objectives
     - Ordinary differential accumulator states; explicit cost terms retained
     - Analytic integral/terminal costs and independent collocation
   * - Scalar numerical policy
     - Positive scalar tolerances; required quadrature/algebraic error control
     - Refusal tests, tolerance convergence, native failure reductions

All full DAE suites pass 293 tests and 300 subtests on macOS 27.0.1 ARM64:
Python 3.12.14 with CasADi 3.8.1/3.8.2, and Python 3.14.3 with 3.8.2.
NumPy is 2.5.4 and SciPy is 1.18.1. The lean environment runs 155 Simulator
tests: seven pass and 148 appropriately skip. No optimizer or ASL is needed
for the native interface, core tests, five guide blocks or four examples.
The optional collocation comparisons use the existing ASL binary, with its
path/hash recorded; they do not exercise an optimization solver.

The compile version guard admits CasADi >=3.8.1; only the matrix above has
been executed. Native directions are limited to one per integrator on 3.8.1
because of the measured batching issue; Function composition still provides
multiple directions. Vector tolerances and sensitivity augmentation overrides
are refused. Scale variables/residuals and seed directions explicitly.

Remaining numerical and release limits
----------------------------------------

* Default native stiff BDF adjoints can fail even for scalar exponential
  decay. Case-specific alternatives are recorded in ODE_NUMERICS.rst.
* Raw IDAS algebraic-output adjoints and mixed-observation adjoints showed
  inaccuracies/failures. The public map uses general implicit reconstruction
  and interval composition; explicit-time equations use an internal clock.
  See DAE_DECISION.rst for the independent reductions and qualifications.
* Near-singular DAEs, a tight coupled consistency case and default zero-rate
  feedback adjoints retain native failures. An explicitly configured zero-rate
  public test passes. No solver/tolerance fallback is hidden in production.
  The tested settings are not a guarantee for arbitrary regular DAEs.
* Local residual/Jacobian checks do not prove global index, conditioning or
  branch uniqueness. Events, resets, switch-time derivatives, high-index or
  purely algebraic systems, higher derivatives and vector weights are outside
  this interface. Ordinary scalar duration parameters on normalized time are
  tested separately from moving observation/switch times.
* Linux, Windows, other supported Python combinations and remote CI have not
  run. A release/submission must obtain those checks and human review. They
  are a separate release gate, not fabricated local successes. No push, PR,
  issue, maintainer contact or external message has occurred.

Requirement audit
-----------------

The source handoff's proposed wrapper/result spelling was superseded by the
user-approved minimal native Function decision in API_DECISION.rst. Mutable
Params are supported; selectable scalar Vars and immutable folded parameters
are refused rather than silently returning zero derivatives. Native named
factories supply the requested direct/adjoint operations. The following maps
every required test family to executed public evidence:

.. list-table:: Standalone goal acceptance cases
   :header-rows: 1
   :widths: 33 67

   * - Requirement
     - Executed evidence
   * - Scalar exponential, multiple times, p and x0
     - test_simulator_compile.py: exponential_forward_and_adjoint, multiple_directions_and_duality
   * - Integral and explicit terminal terms
     - compile: integral_terminal_objective_and_total_derivative; DAE each_terminal_cost
   * - Coupled ODE, vector directions
     - compile: coupled_ode; controls: multiple_controls_directions_and_duality
   * - Stiff ODE and state scaling
     - compile: stiff_ode, tolerance_refinement; test_simulator_ode_qualification.py and ODE_NUMERICS.rst
   * - Linear index-one and nonlinear DAE
     - test_simulator_dae_compile.py: analytic_outputs_directions_and_duality, guesses_and_both_branches
   * - Nonlinear ICs and consistent initial z
     - DAE total_initial_condition_and_cost_derivatives; DAE controls total_initial_condition_and_shared_controls
   * - Algebraic terminal functional
     - DAE each_terminal_cost; pure native G4 matrix and generic reconstruction checks
   * - Fixed segmented amplitudes
     - test_simulator_controls.py and test_simulator_dae_controls.py: exact phases and every input partial
   * - Duration parameter
     - controls: duration_parameter_on_normalized_time
   * - Direct/adjoint identity
     - Multiple directions and random scaled weights in ODE, scaled, DAE and control tests
   * - Parameter registry and changed values
     - compile: parameter_order_and_snapshots, stationary_and_nonlinear_parameters, invalid_parameters; DAE indexed ordering
   * - Mixed scales and tolerance policy
     - Scaled rational ODE analytic/FD tests; scalar-option refusal tests; bounded vector-weight dependency probes
   * - Refined ODE and DAE collocation
     - ODE_COMPARISON.rst and DAE_COMPARISON.rst: public maps, analytic, implicit and discrete FD derivatives
   * - Invalid model/IC and bounded errors
     - DAE invalid_and_singular_systems; control failures_and_previous_simulation_preserved; native subprocess reductions
   * - Model/options/cache preservation
     - Interface, compile, ODE-control, DAE and DAE-control success/failure/reuse tests
   * - Optional dependencies and solver-free example
     - Lean run, guarded test imports, four analytic examples; ASL isolated to research comparisons
   * - Explicit equation-transformation track
     - EQUATION_SENSITIVITIES.rst documents the separate, deferred direct/adjoint transformation boundary

Changes and review units
-------------------------

No new numerical integrator, parameter registry class, result wrapper or
initialization callback was added. The DAE extension reuses conversion and
native phase/root functions. G6 changes 26 executable lines, all covered.
The full Simulator diff from upstream includes the independently justified
legacy fixes, input validation and documentation as well as compile; it is
642 added and 70 removed lines. compile spans 309 lines including its docstring
and validation; the DAE helper spans 144. These counts describe the G7 snapshot.
The user chose to submit this as one targeted Pyomo.DAE PR. Organize the review
around the following related parts while keeping the interface additive.

Review outline for the single PR:

1. Existing Simulator fixes and regression tests, including the independently
   verified SciPy reference correction. See INTERFACE_AUDIT.rst.
2. ODE compile and parameter retention, fixed-grid controls, core tests,
   user documentation and ODE examples. See ODE_CLOSEOUT.rst.
3. The gated DAE extension, implicit consistency/output reconstruction,
   explicit-time clock, controls, DAE tests and examples.
4. Keep raw JSON, probes, benchmarks and internal decision/checkpoint notes
   as separate research evidence unless maintainers request them. The local
   branch intentionally retains these artifacts for reproducibility.

Final coverage is 286/290 (98.62%) modified executable Simulator lines since
the upstream baseline and 96.24% module coverage. Four uncovered lines are
availability/version/current-time guards and the post-construction SciPy
availability guard. They are identified rather than covered by mocks.
The instrumented run passes but emits the existing multiprocessing semaphore
cleanup diagnostic; regular runs are clean. The relevant Sphinx guide/API
build passes with warnings as errors, five numerical blocks and four download
hashes checked. Full-manual and external-link checks remain unexecuted.

Release-note text and local PR-description draft
-------------------------------------------------

The repository has released CHANGELOG.md sections and no established
unreleased fragment directory. Published entries are unchanged. Suggested
text for maintainer placement:

* DAE: add Simulator.compile for reusable CasADi ODE and regular index-one
  DAE maps, with native direct/adjoint products for selected mutable
  parameters, initial states and fixed-grid control amplitudes. Compose
  nonlinear ICs and integral/algebraic terminal costs symbolically.
* DAE: fix Simulator equation separation, input switches, physical-time
  handling and algebraic initialization, and preserve caller options and
  prior simulation results after failed calls.

The following draft follows .github/PULL_REQUEST_TEMPLATE.md. The human
author must choose the review-process disclosure, complete the legal
acknowledgement and adjust the issue/scope for the actual review unit before
submission. No human review or legal authorization is asserted by this draft.

.. code-block:: markdown

   ## Fixes # .
   No issue has been filed; replace this line if applicable.

   ## Summary/Motivation:
   Simulator currently discards its integration function and freezes model
   parameters numerically. Add Simulator.compile to retain selected mutable
   parameters and expose reusable native CasADi maps. Users can differentiate
   states and integral/algebraic terminal costs through initial conditions
   and fixed-grid controls using direct or adjoint products.

   ## Changes proposed in this PR:
   - Add the ODE compile path and regular index-one DAE extension with
     independent differential ICs and consistent algebraic outputs.
   - Preserve the existing public simulation workflow, fix demonstrated
     Simulator defects, and add analytic/FD tests, docs and solver-free examples.
   - Require verified scalar numerical policies and document measured native
     limitations rather than silently changing tolerances or solver options.

   Validation: 293 DAE tests and 300 subtests pass in each of three local
   Python/CasADi environments; the lean optional-dependency check passes.
   Focused Sphinx, examples, formatting, typo checks and modified-line
   coverage pass. Separate ODE/DAE Radau comparisons converge to analytic
   sensitivities. Linux/Windows and the remaining supported Python CI matrix
   still need release qualification.

   ## AI-Use Disclosure
   - [ ] **AI tools were NOT used during the preparation of this PR**
   or
   - [x] **AI tools contributed to the development of this PR**
     - [x] AI tools generated documentation (including this draft and Sphinx docs)
     - [x] AI tools generated tests, examples and numerical probes
     - [x] AI tools generated code apart from tests

   Review process (human author must select ONE before submission):
   - [ ] **Rewritten**: All AI-generated content was rewritten by me.
   - [ ] **Reviewed/verified**: I retained AI-generated content and personally
     verified it, including code execution, tests, logic, contribution-guide
     alignment and security implications as applicable.
   - [ ] **As-is**: AI-generated content was committed directly to the repository.

   Notes for reviewers: Codex prepared and locally exercised the changes.
   Human mathematical/API review is still required. Focus review on implicit
   algebraic reconstruction, branch/guess semantics, clock-state equivalence,
   scalar sensitivity tolerances and right-side observations at switches.

   ### Legal Acknowledgement
   The submitting human must retain and complete the exact acknowledgement
   from .github/PULL_REQUEST_TEMPLATE.md, including BSD licensing and authority
   to contribute. This draft does not make those representations for them.

Local completion
------------------

The recorded G0-G7 qualification passed with the listed native limitations
and release exclusions. The subsequent independent full-diff review is active;
its findings must be resolved before submission. The source and validation hashes,
checkpoint history and final checks are in results/contribution_readiness.json.
Historical source claims are checked against retained Git snapshots, not
misrepresented as hashes of today's files. The next user decision is when
and how to pursue human review and cross-platform release qualification;
no such publication step is part of completing this local development goal.

The historical audit recovers 108 of 110 repository source claims through
311630d2f. Two hashes in the original interface_audit.json refer to simulator.py
and the guide before the first combined A/B commit; those exact intermediate
bytes were not retained in Git. Their numerical observations remain historical
and are not advertised as a reproducible source snapshot. Current interface
regressions pass on the final source. No old hash was rewritten to imply
recovery. results/evidence_hash_audit.json reports all_hashes_recovered=false
and lists these two claims explicitly. All current runtime/test/doc source
hashes used for the final qualification are verified separately.
