<!-- ##################################################################### -->
<!-- PLEASE READ BEFORE OPENING THIS PULL REQUEST -->

<!-- All changes must adhere to PEP8 standards as enforced by Black. -->
<!-- If your changes do NOT adhere, the test suite will fail. -->
<!-- Please read our Contributing guide for instructions on how to apply these standards. -->
<!-- Contributing Guide: https://pyomo.readthedocs.io/en/stable/contribution_guide.html -->
<!-- ##################################################################### -->

<!-- DO NOT DELETE OR IGNORE THIS TEMPLATE. Failing to adhere to this template and provide the necessary information may lead to your Pull Request being closed without consideration. -->

<!-- Proposed PR title: DAE: fix Simulator behavior and add differentiable ODE/DAE maps -->
<!-- Local draft. Independent correctness audits completed; current implementation
     includes the missing-input correction in 085a14557.
     Keep the development
     evidence until the author approves cleanup.
     The human submitter must complete the review-process and legal declarations. -->

## Fixes # .

No linked issue yet; replace the issue placeholder before submission.

## Summary/Motivation:

`Simulator.simulate()` returns numerical trajectories and converts ordinary model
parameters to numbers. This prevents callers from reusing the integration map to
compute parameter sensitivities. Existing defects in equation separation, input
switches, physical time, and algebraic initialization also affect the trajectories
that a sensitivity calculation would differentiate.

This PR corrects those behaviors and adds `Simulator.compile()` for reusable,
differentiable ODE and regular index-one DAE maps. It returns a native
`casadi.Function`, with named inputs and outputs, so callers can evaluate the map,
compose initial conditions and objectives, and request direct or adjoint products.
Sensitivity integration uses CasADi/CVODES/IDAS.

This is one targeted Pyomo.DAE contribution. The fixes and features share
Simulator's equation analysis and expression conversion. The existing
`Simulator`, `simulate()`, and `initialize_model()` signatures and the simulation
time/profile return convention remain unchanged. Corrected trajectories and
algebraic switch values intentionally differ from results affected by the bugs.

## Changes proposed in this PR:

### 1. Correct existing Simulator behavior

| Area | Correction and regression evidence |
| --- | --- |
| Equation separation | Preserve non-derivative product terms when rearranging expressions such as `2*x + dx == 0`; test equivalent equation forms with SciPy and CasADi. |
| Constant derivatives | Evaluate constant SciPy right-hand sides correctly, including zero derivatives. |
| Declarative conditionals | Translate `Expr_if` and its parameter-only variant to symbolic CasADi `if_else`, including native boolean leaves. Test nesting, numeric/boolean conditions and individual comparisons, legacy simulation, and ODE/DAE direct/adjoint products. |
| Piecewise inputs | Restart SciPy integration at input switches so cached derivatives and multistep history do not use the previous amplitude. Use a phase-local time origin while retaining physical time and step-size options. Test asynchronous changes, repeated profiles, and one-ULP finite-area pulses. |
| Observation endpoints | Include the final time when `tstep` does not divide the horizon. Protect exact endpoints and switches when reconciling generated output points close to input changes. Test switches adjacent to both endpoints and distinct adjacent interior switches. |
| Shifted domains and explicit time | Read profiles at the actual initial time and retain physical time in converted differential and algebraic equations, including normalized phases. |
| Algebraic initialization | Pass model algebraic values as IDAS guesses in `simulate()`. Verify positive and negative nonlinear branches while preserving differential initial states. |
| Algebraic switch outputs | Return algebraic values consistent with the newly active input at a switch while keeping differential states continuous. |
| Complex SciPy data | Preserve finite complex ICs supplied as lists or arrays and complex control amplitudes with `zvode`; reject complex data for real-only paths. Test analytic decay and constant/switched complex inputs on shifted domains. |
| Invalid input | Validate observation grids, initial-state shapes/finiteness, integrator selection, and used input profiles before integration. SciPy requires every input profile on every call, including after a previous successful simulation; omission raises before cached amplitudes can be reused. An empty suffix remains valid when the model has no time-varying inputs. |
| Caller and Simulator state | Copy option dictionaries and preserve model values/fixity, caller options, ordering, and previous simulation results after failed calls. Test reuse after failure. |

The SciPy piecewise-pendulum fixture is updated because it encoded the input-switch
defect. A separate segmented DOP853 solution verifies the correction: the old
fixture's maximum error was approximately `1.95e-2`; the corrected Simulator's
error was approximately `6.51e-5`. The permanent regression checks the example
against that independent solve with absolute tolerance `1e-3`.

### 2. Add a reusable ODE/DAE compile interface

```python
flow = sim.compile(
    parameters=[model.k, model.a],
    times=observation_times,
    controls=[model.u],
    control_times=interval_boundaries,
    integrator='idas',
    integrator_options={'reltol': 1e-10, 'abstol': 1e-12},
)
```

ODEs use the default `integrator='cvodes'`; DAEs require explicit
`integrator='idas'`. Controls and their boundaries are optional. Parameter
selection is ordered and accepts mutable scalar `ParamData`, including individual
entries of indexed Params. Duplicate, unused, foreign-model, time-indexed,
immutable/folded, and scalar-Var parameter selections are rejected. Unselected
scalar data are captured at compilation; selected values are supplied on each
evaluation. Model equation changes require a new Simulator.

| Map | Named inputs | Named outputs |
| --- | --- | --- |
| ODE | `x0`, `p`, optionally `u` | `xf` |
| DAE | `x0`, `p`, `z0`, optionally `u` | `xf`, `zf` |

Output rows follow the documented Simulator variable order; columns follow the
requested observation times. Parameter entries follow `parameters`. The control
matrix has one row per selected control and one column per fixed interval.
Compilation does not evaluate a trajectory, read independent initial conditions
from model values, call `initialize_model()`, or alter prior simulation results.
There is no result wrapper, compilation cache, or hidden synchronization with
later model edits.

### 3. Support direct and adjoint sensitivities

The returned Function supports CasADi's named `factory()` interface and
`forward(n)`/`reverse(n)` methods:

- **Direct sensitivities:** Jacobian-vector products `J*v` for differential
  initial states, selected parameters, and fixed-grid control amplitudes. DAE
  products include consistent algebraic observations.
- **Adjoint sensitivities:** vector-Jacobian products `J.T*w`, including weights
  on several states/observations, integral states, and algebraic terminal costs.
  A scalar cost gradient does not require forming the full trajectory Jacobian.
- **Total derivatives:** compose `x0(theta)`, `p(theta)`, `u(theta)`, and the cost
  symbolically before differentiation. This retains nonlinear IC derivatives,
  shared-design dependencies, explicit parameter terms, and control-energy terms.
- **Integrated costs:** represent an integral as an ordinary additional
  differential state and combine it with terminal/observation costs.

Both modes support multiple directions. Inputs `x0`, `p`, and `u` are independent
until the caller supplies their symbolic dependencies. Declarative `Expr_if`
expressions retain those dependencies on the selected branch. Numeric conditions,
native booleans, individual comparisons and nested conditionals are supported;
compound logical expressions and three-term ranged predicates still require
additional converter support. Branch-boundary differentiability and arbitrary
short-circuit evaluation are not guaranteed. This PR does not add a
Pyomo transformation that constructs explicit sensitivity or adjoint components;
CasADi constructs the native sensitivity systems. A hand-written ODE tangent
system is integrated separately as one independent test oracle.

### 4. Handle DAE consistency and fixed-grid controls

Supported DAEs have the semi-explicit form `xdot=f(t,x,z,p,u)`,
`0=g(t,x,z,p,u)`, with square, locally nonsingular `g_z` on the selected branch.
`x0` is an independent differential initial state. `z0` is a consistency-solver
guess and branch seed; its local physical derivative is zero on a fixed branch.
The implementation does not adjust `x0` to satisfy the algebraic residual.

IDAS integrates the original implicit equations. A generic differentiable root
solve of the original residual supplies consistent initial/observed algebraic
states and their implicit derivatives, including direct parameter/control terms.
Normalized interval composition and this reconstruction address measured native
mixed-observation/algebraic-output adjoint failures. Explicit-time DAE equations
use an internal physical-clock state, hidden from public state layouts. No
model-specific elimination or numerical gradient correction is used.

Consistency checks use per-entry finite predicates and native QR of the
row/column-equilibrated algebraic Jacobian. Every absolute QR diagonal must
exceed the algebraic dimension times double-precision machine epsilon. This
avoids determinant overflow/underflow and refuses numerically dependent roots
even at an already-consistent guess. The checks leave the original residual and
its tolerance unchanged; they do not provide global conditioning/index analysis.

For fixed-grid controls:

- Selected controls are whole time-indexed `Var` or `Param` components and may
  appear in differential equations, algebraic equations, or both.
- An amplitude applies on `[s_j, s_(j+1))`. Differential and integral states are
  continuous at switches. Internal-switch algebraic observations use the new,
  right-held amplitude; the terminal boundary uses the last interval's amplitude.
- Observations are independent of control boundaries, and the map returns exactly
  the requested columns. Distinct adjacent floating-point times and narrow
  finite-area intervals are retained.
- With independent initial states, intervals strictly after the last observation
  have zero partial derivatives. An interval starting at that observation can
  immediately affect its algebraic value.

### 5. Document numerical policies and limits

Compilation requires CasADi `>=3.8.1` and positive scalar tolerances. It requires
`quad_err_con=True`, including for adjoint parameter quadratures. DAE maps also
require `calc_ic=True`, `calc_icB=True`, and `suppress_algebraic=False`.
The DAE scalar `abstol` is also the consistency residual target. Vector absolute
tolerances, time-grid overrides, and derivative-augmentation overrides are
rejected. On CasADi 3.8.1, each native integration is limited to one direction to
avoid a measured batching defect; Function composition still supplies multiple
directions. Native batching is permitted on 3.8.2.

These policies do not guarantee convergence for every regular model. Native
stiff BDF adjoints, a tight coupled DAE consistency case, near-singular DAEs, and
zero-feedback adjoints retain recorded failures. The zero-feedback public test
passes with explicitly supplied tighter tolerances and step-size control.
Failures propagate without silent solver changes, tolerance relaxation, or
disabling error control. Users should scale states/residuals and verify convergence
of the complete map and its derivatives.

Purely algebraic/high-index systems, global branch/index analysis, higher-order
sensitivities, resets, and event/switch-time derivatives are outside this PR's
scope. A scalar duration parameter on a fixed normalized time domain is tested.

### 6. Add user documentation and executable examples

The [DAE guide](doc/OnlineDocs/explanation/modeling/dae.rst) documents ordering,
named products, independent versus total derivatives, consistency, switch
semantics, numerical policies, and known limits. Four runnable examples verify
analytic results without ASL or an optimization solver:

- [ODE sensitivities](examples/dae/simulator_sensitivity_example.py).
- [ODE controls and total derivatives](examples/dae/simulator_control_sensitivity_example.py).
- [Nonlinear DAE sensitivities](examples/dae/simulator_dae_sensitivity_example.py).
- [Nonlinear DAE controls and total derivatives](examples/dae/simulator_dae_control_sensitivity_example.py).

### Validation

The complete `pyomo/dae/tests` runs pass **311 tests and 353 subtests in each
environment** below on the implementation committed in `085a14557`.
A separate worker reviewed the combined production diff, fixture, tests/oracles,
guide/examples and draft, including a follow-up review of conditional conversion.
Independent verification includes a full CasADi 3.8.2 run on `669cd15b1`, fifteen
additional regression checks per version, targeted expression/boolean checks,
and a follow-up of missing-input refusal and repeated-call state preservation. The final full matrix below was run on the updated
implementation. No actionable production findings remain in the reviewed scope;
human author review and cross-platform qualification remain pending.

| Platform | Python | CasADi | NumPy | SciPy | Result |
| --- | --- | --- | --- | --- | --- |
| macOS 27.0.1 ARM64 | 3.12.14 | 3.8.1 | 2.5.4 | 1.18.1 | 311 tests + 353 subtests pass |
| macOS 27.0.1 ARM64 | 3.12.14 | 3.8.2 | 2.5.4 | 1.18.1 | 311 tests + 353 subtests pass |
| macOS 27.0.1 ARM64 | 3.14.3 | 3.8.2 | 2.5.4 | 1.18.1 | 311 tests + 353 subtests pass |

Tests use `pyomo.common.unittest`, guarded optional dependencies, actual numerical
integration, and independent references without mocks.

| Check | Method and result |
| --- | --- |
| Existing Simulator behavior | Equivalent equations, constant/coupled/stiff ODEs, indexed states, explicit/shifted time, asynchronous controls, endpoints, initialization, repeated calls, and forced-failure recovery. Missing SciPy inputs are checked on fresh and reused simulators with omitted/empty suffixes; prior results/order and `initialize_model()` remain usable, and explicit-profile reuse still succeeds. |
| ODE derivatives | Exact exponential, affine and nonlinear solutions; a coupled matrix-exponential reference; initial/parameter/control derivatives; integral/terminal costs; two directions and scaled random weights. |
| DAE derivatives | Exact linear, nonlinear and coupled algebraic models; both regular branches; guess invariance; parameters used only in algebraic residuals; initial-only and multiple observations; algebraic/integral terminal costs; full initial/parameter partials. |
| Audit-derived edge cases | One-ULP finite-area pulses at either endpoint and inside the interval; complex `zvode` ICs/controls; 160-variable scaled algebraic blocks outside determinant range with analytic JVP/VJP; mixed units, finite roots of size `1e308`, and dependent-row refusal at consistent guesses with nonsingular neighbors. |
| Controlled DAE derivatives | Exact affine/nonlinear phase composition; controls in `f` and `g`; reordered controls and control types; sparse/dense/switch/adjacent-time observations; narrow pulses; all amplitude partials; failed switch consistency and recovery. |
| Conditional conversion | Eight generic methods cover nested expressions, symbolic parameter-only and native boolean nodes, selected branch values and local derivatives, ODE/DAE initial/parameter/control products, and legacy CasADi simulation. Polynomial, exponential, log and sqrt references are analytic. Local second expression derivatives are tested separately; integrated-map support remains first-order. Conversion failures were reproduced before the correction. |
| Total derivatives | Nonlinear ICs, shared parameter/control dependencies and explicit cost terms; analytic chain rules and centered finite differences across four step sizes. |
| Numerical convergence | Scalar tolerance sweeps, state scales separated by six orders of magnitude, explicitly scaled directions/weights, and unchanged analytic comparison targets, generally `rtol=atol=3e-7`. Selected total FD checks require best scaled error below `1e-6`. |
| Direct/adjoint agreement | Independent analytic products plus forward-adjoint identities; agreement between the two modes alone is not the correctness criterion. |
| Independent collocation | ODE and DAE Radau3 meshes with 4/8/16 elements. Discrete implicit derivatives agree with discrete FD, and continuous errors decrease under refinement. DAE total-cost gradient error decreases from `9.74e-7` to `3.12e-8` to `9.89e-10`; the public DAE analytic checks are below `5.6e-11`. These optional checks use PyNumero/ASL sparse solves, with no NLP solver. |
| Dependency absence | Without NumPy, SciPy or CasADi, 173 Simulator tests run: seven pass and 166 appropriately skip. Core sensitivities and all four examples do not require ASL. |
| Coverage | `309/313` modified executable Simulator lines covered (`98.72%`); module coverage `96.47%`; DAE helper `66/66` statements. All `24/24` executable audit-fix lines, `4/4` conditional-change statements and `2/2` moved input-validation statements are covered. Four uncovered statements are identified defensive guards. |
| Documentation and style | Focused Sphinx 9.1.1 guide/API build with warnings as errors; five executed guide blocks; four executed analytic examples and verified download hashes; Black, repository-configured typos, and whitespace checks pass. |

The coverage-instrumented run passes with an existing multiprocessing semaphore
shutdown diagnostic; regular test runs are clean. The focused documentation
build uses a temporary source copy with the installed theme's deprecated hook
omitted and external intersphinx downloads disabled. It does not establish a
whole-manual or external-link check.

To rerun the core suite from the repository root in an environment with the
selected optional dependencies:

```sh
PYTHONPATH="$PWD" python -m pytest pyomo/dae/tests -q
python -m black --check pyomo/dae/simulator.py pyomo/dae/tests/test_simulator*.py
typos --config .github/workflows/typos.toml pyomo/dae/simulator.py \
    pyomo/dae/tests/test_simulator*.py doc/OnlineDocs/explanation/modeling/dae.rst
```

Run parallel full suites from separate working directories: an existing
ContinuousSet I/O test uses a shared relative filename. The sensitivity examples
are executable directly with the checkout on `PYTHONPATH`.

### Remaining before submission

- Validate the final diff against current upstream `main`; obtain Linux/Windows
  and the remaining supported-Python CI coverage, checking that numerical tests
  execute with supported CasADi versions.
- Run the full documentation build and external-link checks, or explicitly record
  their status in the final submission.
- Once the author is satisfied with correctness, remove development JSON reports,
  performance probes, and development logs from the submitted tree. Retain the
  core regression tests, user documentation, and four executable examples. The
  committed development history preserves the investigation evidence.
- Replace the issue placeholder, complete the human review/AI disclosure and
  legal acknowledgement, and finalize release-note placement.

## AI-Use Disclosure
<!-- Contributors must disclose whether and how AI tools were used and highlight any areas of uncertainty or where they want focused reviewer feedback -->

- [ ] **AI tools were NOT used during the preparation of this PR**

or

- [x] **AI tools contributed to the development of this PR**
    - [x] AI tools generated documentation (including the PR description/comments, code comments, and/or Sphinx documentation)
    - [x] AI tools generated tests (baselines, examples, and/or code)
    - [x] AI tools generated code (apart from tests)

    *Review process (select ONE)*:
    - [ ] **Rewritten**: All AI-generated content was rewritten by me before being committed.
    - [ ] **Reviewed/verified**: I retained AI-generated content and verified it before committing. Verification included (as applicable):
        - [ ] Ran the code and fixed issues
        - [ ] Added and ran tests
        - [ ] Checked correctness/logic of code and tests
        - [ ] Checked for alignment with the contribution guide
        - [ ] Considered security implications
    - [ ] **As-is**: AI-generated content was committed directly to the repository

 **Notes for reviewers (optional):** Codex contributed production code, tests, numerical
probes, documentation, and this draft. Automated checks and an independent agent
audit supplement the submitting author's personal review. Review-process boxes
are intentionally unselected until the human author completes that declaration.
Please focus mathematical review on consistency/guess semantics, implicit
algebraic-output derivatives, the internal clock, control-switch observations,
and the numerical restrictions required by the tested dependency versions.

### Legal Acknowledgement

By contributing to this software project, I have read the [contribution guide](https://pyomo.readthedocs.io/en/stable/contribution_guide.html) and agree to the following terms and conditions for my contribution:

1. I agree my contributions are submitted under the BSD license.
2. I represent I am authorized to make the contributions and grant the license. If my employer has rights to intellectual property that includes these contributions, I represent that I have received permission to make contributions and grant the required license on behalf of that employer.
