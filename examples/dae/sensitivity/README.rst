Differentiable simulation: dependency baseline
==============================================

30 October 2026. Work package A on branch ``pyomo-dae-sensitivity``.
This folder contains investigation scripts and evidence. The dependency
baseline below predates the existing interface bug fixes recorded in
``INTERFACE_AUDIT.rst``. Work packages B and C implement the ODE-first compile
API described in ``API_DECISION.rst``, including fixed-grid controls and total
derivatives through initial conditions. Executable examples are
``examples/dae/simulator_sensitivity_example.py`` and
``examples/dae/simulator_control_sensitivity_example.py``. Current validation
for the C snapshot is recorded in ``results/controls_validation.json``;
the prior B snapshot is ``results/compile_validation.json`` (commit
``74e516a11``). The ODE closeout and provisional API freeze are described
in ``ODE_CLOSEOUT.rst`` and ``results/ode_closeout.json``. See
``DEVELOPMENT_GOAL.rst`` and ``DEVELOPMENT_LOG.rst`` for the gated DAE work.
G5 now implements the index-one DAE bridge described in ``DAE_DECISION.rst``;
its public-interface evidence is ``results/dae_compile_validation.json``
and its runnable example is
``examples/dae/simulator_dae_sensitivity_example.py``. G6 enables fixed-grid
DAE controls with right-side algebraic observations at switches. Its contract,
public validation and runnable example are ``DAE_CONTROLS.rst``,
``results/dae_controls_validation.json`` and
``examples/dae/simulator_dae_control_sensitivity_example.py``.
Final local support, requirement coverage and proposed contribution packaging
are in ``CONTRIBUTION_READINESS.rst`` and
``results/contribution_readiness.json``. ``DAE_COMPARISON.rst`` closes the
independent refined DAE collocation check. ``EQUATION_SENSITIVITIES.rst``
keeps a future explicit Pyomo equation transformation separate from the
implemented native direct/adjoint interface.
The user has selected one targeted Pyomo.DAE PR. PR_DESCRIPTION.md at the
repository root is the current Markdown draft, including validation details
and the status of the independent full-diff audit. Development artifacts are
retained until cleanup is approved. The independent full-diff review is complete;
its three finding categories are fixed and verified in 8cbe3070d. PR_AUDIT.rst
and results/pr_audit_validation.json contain the latest evidence and supersede
earlier test counts for the current implementation.

The supplied probes reproduce: eight of twelve native cases meet their
analytic checks with both CasADi versions. Four cases remain outside the
verified capability. Upgrading to 3.8.2 improves the algebraic adjoint but
does not make that test pass. The collocation probe passes independently.

Environment and provenance
--------------------------

The runs used fresh virtual environments with Python 3.12.14, NumPy 2.5.4,
and SciPy 1.18.1 on macOS 27.0.1, ARM64. The CasADi and released Pyomo
packages came from PyPI wheels. The four combinations were:

.. list-table:: Executed dependency matrix
   :header-rows: 1
   :widths: 30 15 20 20

   * - Pyomo source
     - CasADi
     - Native analytic checks
     - Collocation meshes
   * - Released 6.10.1
     - 3.8.1
     - 8 pass / 4 fail
     - 4, 8, 16: pass
   * - Released 6.10.1
     - 3.8.2
     - 8 pass / 4 fail
     - 4, 8, 16: pass
   * - Checkout 6.10.2.dev0
     - 3.8.1
     - 8 pass / 4 fail
     - 4, 8, 16: pass
   * - Checkout 6.10.2.dev0
     - 3.8.2
     - 8 pass / 4 fail
     - 4, 8, 16: pass

The baseline checkout was ``a6c7d20b27b379e9e54aa25986fcd50202fbf10c``. Its
unmodified ``pyomo/dae/simulator.py`` was byte-identical to the installed 6.10.1 module:
SHA256 ``834896495d3d3d65a19be2c57bd16234c555335fe637d66cad8ea78646ecbede``.
Native CasADi results do not exercise the Pyomo bridge; listing both Pyomo
sources establishes provenance, not additional sensitivity coverage.

Collocation used the existing, externally discovered PyNumero ASL library at
``~/.idaes/bin/libpynumero_ASL.dylib``. Its hash is recorded in the results.
This was not a fresh ASL build. No optimization solver was called. Native
probes need neither ASL nor an optimization solver.

``results/supplied_probes.json`` retains the four complete result records,
Python/package versions, module paths, and source/library hashes. The initial
runs used byte-for-byte copies of the supplied scripts, whose hashes are
included. The copies here have Black formatting, docstrings, and an
``--output`` argument added to the native probe. Their numerical checks are
unchanged. ``results/provenance.json`` records the portable script hashes and
subsequent validation, including the later interface audit.

Observed capability
-------------------

Pass means the supplied analytic check meets its scaled ``3e-7`` threshold
for this particular model and configuration. It is not a guarantee for
arbitrary ODEs or index-one DAEs.

.. list-table:: Supplied native probe results, identical classifications for both versions
   :header-rows: 1
   :widths: 52 16 32

   * - Case
     - Result
     - Qualification
   * - Exponential ODE: forward/adjoint, quadrature, initial state
     - Pass
     - Scalar absolute tolerance
   * - Linear index-one DAE: x, z, quadrature forward derivatives
     - Pass
     - Three directions; scalar tolerance
   * - DAE: unexpanded vector absolute tolerance
     - Fail
     - Zero error weights, IDA_ILL_INPUT
   * - DAE: grouped augmented vector tolerance, one direction
     - Pass
     - Forward only
   * - DAE: grouped augmented vector tolerance, two directions
     - Pass
     - Forward only
   * - DAE: adjoint objective using final x and quadrature
     - Pass
     - No terminal z seed
   * - DAE: adjoint objective using final z only
     - Fail
     - Analytic discrepancy
   * - DAE: adjoint objective using final x, z and quadrature
     - Fail
     - Analytic discrepancy
   * - DAE: reconstruct terminal z as gain times final x
     - Pass
     - Model-specific explicit elimination
   * - Native CVODES control grid with discontinuous amplitudes
     - Fail
     - Primal CV_ERR_FAILURE at first switch
   * - Explicitly composed fixed control segments
     - Pass
     - Primal, forward, adjoint
   * - Nonlinear DAE with parameter-dependent consistent IC
     - Pass
     - Positive branch; total forward derivative

Reduced cases and conclusions
-----------------------------

``dependency_reproducers.py`` imports only CasADi and NumPy. Every case runs
in its own subprocess, with a 30 second timeout and core dumps disabled on
POSIX systems. Full observations are in ``results/triage-3.8.1.json`` and
``results/triage-3.8.2.json``.

* Algebraic terminal objective: the discrepancy persists after removing the
  quadrature and differentiating only with respect to the two physical
  parameters. Four tolerance pairs and four finite-difference steps are
  retained. At relative tolerance ``1e-10``, maximum absolute reverse error
  is about ``0.401`` with 3.8.1 and ``6.34e-4`` with 3.8.2. Explicitly
  reconstructing z gives about ``4.97e-10`` for both. At ``1e-12``, the native
  3.8.2 error drops to ``9.86e-5``; this is an improvement, not evidence of
  sufficient accuracy. Forward and finite-difference results support the
  analytic oracle, and the native forward/adjoint dot products disagree.

  The `3.8.2 release notes <https://web.casadi.org/get/>`_ report a fix for
  missing direct parameter/control dependence in IDAS adjoints, referencing
  `CasADi issue 4063 <https://github.com/casadi/casadi/issues/4063>`_. The
  observed change is consistent with that fix. The cause of the remaining
  error is unresolved. No general correction or upstream bug claim is
  justified by these probes.

* Vector weights: the failure persists without a quadrature. Grouped
  ``augmented_options['abstolv']`` entries pass for one and two forward
  directions. This does not establish a safe reverse-mode or arbitrary
  direction policy. Reject vector tolerances in the first sensitivity API;
  do not silently expand them, suppress algebraic error control, or reuse
  forward weights for reverse calculations. The original handoff mentions
  a native crash for a mixed mode configuration without supplying an exact
  crash fixture; that historical crash has not been independently reproduced.

* Controls: one native control-grid call fails at the first change for
  relative tolerances ``1e-7``, ``1e-9``, and ``1e-11`` on both versions.
  Explicit segment composition converges to the analytic solution. At the
  tightest pair, forward and reverse errors are below ``6e-11``. This does
  not diagnose all CasADi control-grid configurations; keep explicit fixed
  segments as the candidate for later control support.

* ODE without quadratures: three observations, initial-state and parameter
  directions, explicit terminal parameter dependence, parameter-dependent
  IC composition, and a forward/adjoint identity pass on both versions.

The collocation probe differentiates discrete Radau equations using PyNumero
Jacobian blocks and SciPy linear solves. Its finite-difference error is at
most ``3.4e-11``. Continuous derivative error decreases from ``1.86e-6`` to
``6.00e-8`` to ``1.91e-9`` over 4, 8, and 16 elements. This is a discretization
comparison, not an adaptive IDAS sensitivity test.

Reproduction
------------

From the repository root, create two environments with a Python 3.12
interpreter. For example, repeat with ``VERSION=3.8.1`` and ``VERSION=3.8.2``:

.. code-block:: sh

   VERSION=3.8.1
   python3.12 -m venv /tmp/dae-casadi-$VERSION
   /tmp/dae-casadi-$VERSION/bin/python -m pip install \
       pyomo==6.10.1 casadi==$VERSION numpy==2.5.4 scipy==1.18.1
   PYTHONPATH="$PWD" /tmp/dae-casadi-$VERSION/bin/python \
       examples/dae/sensitivity/probe_sensitivities.py \
       --output /tmp/native-$VERSION.json
   /tmp/dae-casadi-$VERSION/bin/python \
       examples/dae/sensitivity/dependency_reproducers.py \
       --output /tmp/triage-$VERSION.json
   PYTHONPATH="$PWD" /tmp/dae-casadi-$VERSION/bin/python \
       examples/dae/sensitivity/pynumero_collocation_probe.py \
       --output /tmp/collocation-$VERSION.json

Omit ``PYTHONPATH`` and run from outside the checkout to test the released
Pyomo package. The collocation command additionally needs a discoverable
PyNumero ASL extension; record which library it loads. These scripts are
explicit diagnostics with dependency imports, not automatically collected
Pyomo unit tests. Native and triage runners record expected failures and
can exit zero despite individual failures: inspect the JSON records.

Validation and limits
---------------------

The original ``pyomo/dae/tests/test_simulator.py`` suite passed all 39 tests
with each CasADi version. It emitted four existing NumPy 2.5 shape-assignment
deprecation warnings per run. The portable probes and formatting checks
are recorded in ``results/provenance.json``. These are the original work
package A checks. The later audit extends production test coverage and fixes
existing defects; see ``INTERFACE_AUDIT.rst``. These earlier test counts and
hashes are historical snapshots; later controls and ODE-closeout records
identify their own validated source snapshots. The work package B compile
record retains its original results and source hashes.

This matrix covers one Python version, operating system, architecture, and
ASL binary. Older CasADi releases, Windows/Linux, singular or high-index
DAE support remain unqualified. G5 adds
public-interface checks for parameter-dependent differential ICs and
consistent algebraic initialization on tested index-one branches, including
reverse products. The versioned native and public records distinguish
accurate derivative paths from numerical failures; they do not establish
arbitrary stiff-system or branch robustness. The G6 public control checks
extend this historical work package A matrix; they do not change its original
native probe results. See DAE_CONTROLS.rst for the additional zero-rate IDAS
adjoint limitation and an explicitly configured, verified alternative.
