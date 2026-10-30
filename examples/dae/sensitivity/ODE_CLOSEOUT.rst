ODE closeout and contribution packaging
=======================================

G3, 30 October 2026. ODE implementation parent: 5adc03eac. This checkpoint
provisionally freezes the ODE API and permits G4 DAE experiments. It does
not establish DAE support or upstream maintainer acceptance.

Local acceptance evidence
-------------------------

G1 numerical qualification and G2 collocation/cost gates pass. No production
Simulator changes were needed after work package C. The public interface
remains a native CasADi Function with x0/p and optional u inputs, fixed grids,
first derivative products, scalar tolerances and required quadrature error
control. Existing simulate signatures and return layout remain unchanged.

The full pyomo/dae/tests directory passes 267 tests and 236 subtests in each
of three local environments: Python 3.12.14/CasADi 3.8.1, Python
3.12.14/CasADi 3.8.2, and Python 3.14.3/CasADi 3.8.2. All use macOS ARM64,
NumPy 2.5.4 and SciPy 1.18.1. Five Simulator modules run 129 tests without
NumPy/SciPy/CasADi: seven pass and 122 skip. These are actual local checks;
Linux, Windows, other Python combinations and remote CI remain unexecuted.
Their release qualification is pending and does not prevent local research.

Fresh coverage over the full DAE suite measures 208/212 (98.1%) modified
executable Simulator lines since the upstream baseline, 101/104 (97.1%)
compile lines, and 95.6% module coverage. Four modified uncovered lines
are defensive availability/version/current-time refusals. No mock or
unsupported numerical bypass was added merely to cover those lines.
The instrumented run passes but emits existing multiprocessing semaphore
cleanup diagnostics, including a shutdown FileNotFoundError; the regular
runs are clean. The report preserves this distinction.

The focused Sphinx 9.1.1 build uses the real project configuration and
unchanged DAE guide/API sources in a temporary copy, with warnings treated
as errors. It builds the guide, compile API and the guide's bibliography
entry, verifies both downloaded examples by hash, and executes all four
map snippets and both analytic examples. The numerical checks also pass
in the other two environments. This is a relevant-guide build, not a
whole-manual build or an external-link check.

The build fixture selects only the referenced bibliography entry to avoid
warnings about unrelated uncited entries. In the temporary configuration
it omits the legacy get_html_theme_path call, which is deprecated in the
installed theme, and disables external intersphinx inventory downloads.
Those changes do not alter the repository configuration or suppress guide
warnings. Run check_ode_docs.py to reproduce the exact build setup.

Known numerical and cost limits
-------------------------------

The stiff-adjoint scalar reduction remains a native failure, with explicit
case-specific alternatives; see ODE_NUMERICS.rst. Controlled maps restart
CVODES on observation/switch intervals and can accumulate local errors.
The retained coarse-tolerance pilot misses its accuracy target; explicit
tolerance refinement passes. Runtime grows with interval work, and the
small uncontrolled map is much cheaper. See ODE_COMPARISON.rst. Neither
result justifies silently changing integrator settings or a broad rewrite.

No known Pyomo bridge correctness defect remains in the verified ODE
contract. This is a bounded statement supported by the listed tests, not
proof of numerical correctness for all ODEs. Event-time derivatives,
high-index systems, second derivatives and vector weights are deferred.

Proposed contribution breakdown
-------------------------------

Prepare related review units without rewriting the existing local history:

1. Existing Simulator correctness fixes and regression tests: equation
   parsing, explicit time and shifted input domains, switch restarts,
   algebraic initialization, endpoint/argument validation and preservation
   of caller/model/simulation state. Explain the independently justified
   SciPy reference update. See INTERFACE_AUDIT.rst.
2. Additive ODE compile bridge and parameter retention, followed by
   fixed-grid control composition. Include core optional-dependency tests,
   documentation and the two runnable examples. Explain the numerical
   policies and independent oracles in the description.
3. Keep dependency probes, raw research JSON, checkpoint logs, benchmarks
   and internal API decisions separate from a proposed upstream production
   diff unless maintainers request them. They remain available in this
   branch as reproducibility evidence. Do not mix a future DAE extension
   into the ODE review without a clear reason and completed DAE gates.

The repository CHANGELOG.md contains released-version sections, without
an established unreleased fragment directory. Preserve those published
entries. The following text is ready for the eventual release-note or
PR-description draft, subject to maintainer placement:

* DAE: add Simulator.compile for reusable CasADi ODE maps and native direct
  and adjoint products with respect to selected mutable parameters,
  independent initial states and fixed-grid piecewise control amplitudes.
  Total derivatives through initial conditions and integral/terminal costs
  use ordinary symbolic composition. CVODES with CasADi 3.8.1 or later is
  required; scalar tolerances and quadrature error control are enforced.
* DAE: correct Simulator equation parsing, switching/time handling and
  algebraic initialization, and preserve caller options and prior results
  across successful and failed simulations.

These implementation, test and documentation changes were prepared with
Codex assistance. A human contributor remains responsible for reviewing
the mathematics, scope and final submission under the Pyomo contribution
policy. No PR has been published and no maintainer has been contacted.

Reproduction
------------

Run each full suite from its own working directory because the existing
ContinuousSet I/O tests use a shared relative filename. Use an absolute
path to the tests and PYTHONPATH pointing at the checkout. See
results/ode_closeout.json for commands, environment versions, source hashes
and coverage. From the checkout root, the documentation check is:

.. code-block:: sh

   PYTHONPATH="$PWD" <sphinx-and-casadi-env-python> \
       examples/dae/sensitivity/check_ode_docs.py

Use --skip-build to execute the numerical blocks/examples in a CasADi
environment without Sphinx. Next checkpoint: native DAE initialization and
output/derivative contract, with a decision note before production changes.
