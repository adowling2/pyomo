Independent audit of the combined Pyomo.DAE contribution
========================================================

30 October 2026. The user requested one targeted PR covering the Simulator
fixes, ODE/DAE maps and direct/adjoint products, plus an independent worker
audit of the entire diff. The review baseline is a6c7d20b27b379e9e54aa25986fcd50202fbf10c.
PR_DESCRIPTION.md was drafted and committed in 35c6d4884. Development
artifacts remain until the author is satisfied and approves cleanup.

Scope and findings
-------------------

The worker reviewed the combined production changes, SciPy fixture correction,
new core tests and oracle independence, user guide, four examples and PR draft.
It independently reproduced three correctness categories that the earlier
passing suites missed. The parent implemented fixes; the worker reviewed
and tested the corrected source without editing tracked files.

* Endpoint/switch handling: snapping near a switch could remove the exact
  terminal observation. A one-ULP interval with amplitude 1e16 should add
  1.1102230246251565 to a state initially equal to one. Endpoints and exact
  switches are now protected. SciPy uses a local time origin at its existing
  switch restarts, retaining physical time and physical step-size options,
  so narrow intervals do not fail merely from absolute-time roundoff.
* Algebraic regularity: determinant products overflowed/underflowed on regular,
  well-conditioned blocks, and roundoff could admit exactly dependent rows
  at an already-consistent guess. Native QR now checks a row/column-equilibrated
  Jacobian with a dimension-times-machine-epsilon numerical rank threshold.
  Per-entry finite predicates avoid overflowing sums of finite roots. Original
  residual equations, root tolerances and IDAS integration are unchanged.
* Complex SciPy data: float coercion could discard the imaginary part of
  ndarray ICs or reject complex lists; finite complex control amplitudes were
  also rejected. zvode retains both complex ICs and amplitudes. Other backends
  require real data, and all paths retain shape/finiteness checks.

Permanent regressions
----------------------

Nine additional collected test methods exercise these cases through the real
public interface: two shared methods run on SciPy and CasADi, two SciPy-only
complex-data methods, and three DAE-compile methods. The new cases cover both
endpoints and interior one-ULP pulses, list/array complex ICs and shifted
complex input switches, 160-variable diagonal algebraic blocks with scale
1e-3 or 100 and analytic JVP/VJP values, already-consistent dependent rows and
nonsingular neighbors, mixed row/column units and finite roots of size 1e308.
No mocks, solver fallback or changed derivative oracle were introduced.

All three full DAE suites pass 302 tests and 320 subtests. The lean run has
seven passes and 157 skips out of 164 Simulator tests. Fresh modified coverage
is 24/24 executable audit-fix lines, 304/308 (98.70%) since the upstream
baseline, and 680/706 (96.32%) overall. The DAE helper is 66/66 statements.
The existing instrumented semaphore shutdown diagnostic remains separately
recorded; regular suites are clean.

The worker independently reran the stable-source full suite on CasADi 3.8.2:
302 tests/320 subtests pass, exit zero, with source hashes unchanged during
execution. Fifteen additional scratch checks pass on each of 3.8.1 and 3.8.2,
including 180-variable 0.01*I and 160-variable 100*I primal/JVP/VJP cases.
Permanent tests cover the same failure families with independent exact checks.

Five guide blocks, four analytic examples in all three environments, the
focused warning-as-error Sphinx build and four download hashes pass. The DAE
Radau comparison was rerun on both versions: all prior analytic/FD/refinement
checks still pass on the corrected source. Black, typo and whitespace checks
pass. results/pr_audit_validation.json records exact source hashes, commands,
versions, numerical evidence and the independent review outcome.

Outcome and limits
-------------------

The worker reports no outstanding production correctness findings on the
corrected source. This is a scoped independent agent audit, not a proof for
arbitrary DAEs or a substitute for the human author's review. The previously
documented native integration limits, local numerical-rank/branch limits,
unexecuted cross-platform/full-manual checks and human submission declarations
remain. Historical G0-G7 JSON records retain their earlier source snapshots;
this audit's validation supersedes their test counts for the current tree.
