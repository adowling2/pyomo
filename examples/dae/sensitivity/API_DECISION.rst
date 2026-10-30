Simulator sensitivity API decision
==================================

30 October 2026. Work packages B/C and the G5/G6 index-one DAE bridge are
implemented locally; upstream maintainer review is pending. The ODE API
was provisionally frozen after G3, recorded in ``results/ode_closeout.json``
and ``ODE_CLOSEOUT.rst``. Current DAE validation is in
``results/dae_controls_validation.json``; G5's autonomous snapshot is
``results/dae_compile_validation.json``. The contract and native qualifications
are in ``DAE_DECISION.rst`` and ``DAE_CONTROLS.rst``. The work package C snapshot is
``results/controls_validation.json``. The work package B snapshot is
``results/compile_validation.json``, committed in ``74e516a11``.

Implemented interface
---------------------

The CasADi-only ``Simulator.compile(...)`` method returns an ordinary
``casadi.Function``. Start with ODEs, explicitly selected mutable parameters,
fixed observation times, fixed-grid piecewise controls, and scalar tolerances.
Keep ``Simulator.__init__``,
``simulate()`` and ``initialize_model()`` signatures, return conventions,
and initialization workflow. Correct demonstrated bugs in those methods.

A native Function already supports evaluation, symbolic composition,
statistics, and named forward/reverse factories. Reusing it avoids a second
result type, a parameter registry class, and Pyomo wrappers for every CasADi
derivative operation. The caller retains the function. No compilation cache
or hidden model synchronization is needed.

G5 extends the same method to index-one DAEs with explicit integrator='idas'.
The additional native z0 input is a branch/solver guess; zf is reconstructed
from the original algebraic residual using an implicit root solve. Native
IDAS intervals provide derivative integration. Explicit time uses an internal
clock. No new compile keyword or result wrapper is introduced. G6 extends
the existing fixed-grid controls to DAEs: differential states are continuous,
internal-switch algebraic observations use the new amplitude, and terminal
observations use the last interval's amplitude. The native layout is
x0, p, z0, u -> xf, zf. See DAE_CONTROLS.rst for numerical qualifications.

The supplied handoff proposed a compiler/result wrapper with ``evaluate``,
``jvp`` and ``vjp`` methods. Defer those conveniences until a demonstrated
use case requires them. A separate compiler class would also need to reuse
or repeat Simulator's model analysis. One additive method is the smaller
starting point for the requested contribution.

Compatibility boundary
----------------------

Compatibility protects intended mathematical behavior, not implementation
defects or inaccurate reference trajectories. The accompanying
``INTERFACE_AUDIT.rst`` records fixes to the existing Simulator, including
equation parsing, input switches, time handling, algebraic initialization,
and caller/state preservation. The sensitivity implementation should reuse
those corrected paths where useful. It must not create parallel code merely
to preserve a bug in ``simulate``.

The ODE-first boundary below is a staging decision supported by the measured
CasADi derivative limitations. It is not a requirement to retain current
Simulator deficiencies. Revisit it when dependency accuracy and a small
implementation support a broader contract.

Public contract
---------------

Usage:

.. code-block:: python

   import casadi as ca

   sim = Simulator(model, package='casadi')
   flow = sim.compile(
       parameters=[model.k],
       times=[0.2, 0.7, 1.2],
       integrator='cvodes',
       integrator_options={'reltol': 1e-10, 'abstol': 1e-12},
   )
   result = flow(x0=[1.3], p=[-0.7])
   tangent = flow.factory(
       'tangent', ['x0', 'p', 'fwd:x0', 'fwd:p'], ['fwd:xf']
   )
   direction = tangent(x0=[1.3], p=[-0.7], fwd_x0=[0], fwd_p=[1])
   adjoint = flow.factory(
       'adjoint', ['x0', 'p', 'adj:xf'], ['adj:x0', 'adj:p']
   )
   gradient = adjoint(x0=[1.3], p=[-0.7], adj_xf=ca.DM([[0, 0, 1]]))

The input names are ``x0`` and ``p``, plus ``u`` when controls are selected;
the output is ``xf``. Returning a
small ordinary Function around the native integrator can remove unused
native inputs/outputs without a Python wrapper class. Rows follow
``sim.get_variable_order()``; columns follow the supplied times. Parameters
follow the explicitly supplied component-data list. This layout follows
CasADi's matrix convention; the existing ``simulate()`` profile layout
remains unchanged. Publish both conventions with a concrete example.

The implemented contract is:

* Require CasADi mode and use CVODES for the initial ODE contribution.
  Retain the documented first-order separable, single-ContinuousSet scope.
  Treat other existing restrictions as decisions to assess, not invariants.
  Require a nonempty, finite,
  strictly increasing observation grid within the ContinuousSet bounds;
  define inclusion of the initial time explicitly and test it.
* Accept individual mutable scalar Param data, including explicitly selected
  entries of indexed Params. Reject duplicates, foreign components, immutable
  or unused selections, and time-indexed parameter selections. An empty list
  can support initial-state derivatives. Defer scalar Var selection until it
  has its own validation and tests; do not change fixity to expose a symbol.
* Substitute selected leaves by identity, before numerical conversion.
  Supply selected values on every Function evaluation. Other scalar data
  are snapshots at compilation; changing them requires recompilation.
  A parameter at a stationary point can have a legitimate zero derivative;
  validate structural dependence, not the numerical gradient at one point.
* Treat x0 and p as independent inputs. Derivatives are partial derivatives
  unless the caller explicitly composes x0(p) into a CasADi Function. Show
  the x0=p**2 chain rule. Do not infer dependencies from numeric model values
  or introduce a Pyomo IC mapping API in the first patch.
* Copy caller options, including nested dictionaries. Refuse ``abstolv`` and
  derivative augmentation overrides in the first implementation. A map with
  preconfigured ``nfwd``/``nadj`` has different shapes from the promised
  primal map. Do not silently adjust tolerances or change error control.
* Leave model values, fixity, activation, scaling, Simulator simulation
  results, and caller dictionaries unchanged on both success and failure.
  Compiling must not initialize the model or run a numerical trajectory.

Fixed-grid controls and initial conditions
------------------------------------------

Work package C adds optional ``controls=()`` and ``control_times=None``
arguments. Controls are ordered whole Var or Param components indexed solely
by the ContinuousSet, not individual time-indexed data entries. Mutable and
immutable time-indexed Params retain symbolic amplitudes. All time-varying
inputs must be selected; reject unused, duplicate, foreign or multi-index
selections. Constant parameters still use the existing ``parameters`` list.

The additional native input ``u`` has one row per selected control and one
column per interval. A common, finite, strictly increasing ``control_times``
grid includes both model bounds. Column j applies on [s_j, s_(j+1)); the
terminal state is reached with the last interval's amplitude. States are
continuous across switches. Observation times are independent of the control
grid, and only requested observation columns are returned. Amplitudes after
the last observation have zero partial derivatives for independent x0.
Supplying control_times without controls is rejected. No Suffix, model value
or fixity determines the numerical control values of this map.

Use the existing normalized-interval/mapaccum approach from Simulator's
varying-input path. Pack each interval's constant amplitudes into the native
integrator's parameter vector. The new implementation reuses the same small
ODE assembly helper; legacy simulate behavior is unchanged. Compose over
the union of observations and switches, normalizing each nonzero interval
to [0, 1] and retaining physical time as start+duration*tau. A per-phase
observation grid exposed CV_TOO_CLOSE in native adjoints for observations
immediately beside switches. Per-interval normalization passes both sides
without snapping observations or deleting narrow intervals. A finite-area
pulse between adjacent floating-point boundaries is explicitly tested.
The native integrator is constructed once and reused through mapaccum;
no numerical integrator or derivative algorithm is implemented in Pyomo.

Keep x0, p and u independent at the map boundary. For a composed objective
J(theta)=Phi(flow(h(theta), p(theta), u(theta)), theta), both native derivative
modes include the initial-state, parameter, control and explicit objective
terms. In reverse mode this includes h_theta.T*adj_x0. Repeated or shared
control amplitudes use ordinary CasADi expressions. No IC mapping class,
callback or inference from numeric Pyomo initial values is introduced.

Integral objectives remain ordinary differential states, carried across
intervals. A horizon parameter works in an explicitly time-normalized model
with fixed grids; it is tested in both derivative modes without adding a
new time API. Numeric switch-time derivatives, events and resets remain
outside the contract. Scalar tolerance and version policies below apply
equally to controlled and uncontrolled maps.

Direct and adjoint equations
----------------------------

Both modes are supported through native CasADi derivative functions. For
x_dot=f(t,x,p), a direct direction satisfies S_dot=f_x*S+f_p*dp with
S(t0)=dx0. An adjoint for a terminal/integral cost satisfies
-lambda_dot=f_x.T*lambda+L_x, with lambda(T)=Phi_x. Its independent-input
gradients are lambda(t0) and Phi_p+integral(L_p+f_p.T*lambda)dt.
Parameter-dependent ICs contribute x0_p.T*lambda(t0) to the total gradient.

CasADi constructs and integrates these sensitivity systems. The returned
Function exposes named JVP/VJP factories; a separate transformation that
writes sensitivity equations as Pyomo components is not part of this patch.
Tests compare native products with analytic formulas and independently
written direct/adjoint ODEs, including a reverse-time adjoint integration.

Numerical policy and version qualifications
-------------------------------------------

Compilation requires ``quad_err_con=True``. Adjoint parameter gradients use
internal quadratures
even if the user ODE has no quadrature output. With that control disabled,
the simple x_dot=p*t example gives about 8% gradient error at tight primal
tolerances; enabling it reduces error below 5e-11. A false override raises
ValueError. Requested scalar tolerances are preserved.

The new method requires CasADi 3.8.1 or later, matching the verified matrix
and the upstream fix for empty-quadrature adjoints in that release. CasADi
3.8.1 gives incorrect
multi-direction adjoints in a two-state/three-parameter example. Setting its
native ``max_num_dir=1`` correctly composes single-direction calls, retaining
the public multi-direction API. Compilation sets and enforces that limit on
3.8.1. CasADi 3.8.2 uses normal native batching. No Python derivative wrapper
or correction to numerical gradients is introduced. Existing ``simulate``
dependency support is unchanged.

These native fixes are described in the
`CasADi release notes <https://web.casadi.org/get/>`_. The independent probes
retain version-specific observations instead of assuming that the notes
establish accuracy for every configuration.

Native BDF adjoints fail with CV_TOO_MUCH_WORK on the tested 1000:1
fast/slow decay example. Explicit Adams configuration passes that case;
default BDF passes the 100:1 case. The compiler does not switch algorithms
after a failure or claim universal stiff-adjoint robustness. Pure-CasADi
records are in ``results/compile-dependencies-3.8.1.json`` and its 3.8.2
counterpart; the permanent public-interface tests use the verified settings.

First patch boundary
--------------------

ODE primal maps and first derivative products, including fixed-grid control
amplitudes, form the initial contribution. G5 adds the separately qualified
index-one DAE path described in DAE_DECISION.rst.
The two tested CasADi versions are validation targets, not justification for
restricting the existing Simulator's dependency support.

An integral cost can be an additional differential state q with q_dot=L.
A terminal objective with explicit parameter dependence can be composed
symbolically outside the map. This retains the direct parameter term and
avoids new output-expression and quadrature APIs. Native quadrature
extraction and custom outputs can follow separately. Duration parameters in
explicitly normalized models already use ordinary parameter selection.

The G4 dependency matrix established a correct algebraic-output derivative
path through generic implicit reconstruction and normalized IDAS interval
composition. G5 implements this path with local consistency/regularity
checks, instead of exposing an inaccurate native zf and attempting to limit
its reverse seeds. Independent differential ICs are held fixed; algebraic
guesses select a branch. The implementation does not insert model-specific
gradient corrections. Numerical convergence limitations remain explicit.

Control boundaries are fixed and only amplitudes are Function inputs.
Event times, resets, second derivatives, and a new adaptive
sensitivity integrator remain outside this contribution.

Why this can stay small
-----------------------

Source inspection and a tiny read-only check establish the relevant seams:

* ``Substitute_Pyomo2Casadi_Visitor`` replaces time-indexed state templates,
  while retaining mutable scalar Params and scalar Vars in the intermediate
  RHS expressions. A check with a mutable scalar Param, an indexed mutable
  Param entry, and a scalar Var confirmed that their identities survive
  Simulator construction. Generic Pyomo leaf walkers do not accept the
  mixed Pyomo/CasADi tree; extend the existing conversion walker instead.
* ``Convert_Pyomo2Casadi_Visitor.visiting_potential_leaf`` calls ``value`` on
  those scalar leaves. A small optional identity-to-symbol map in this
  visitor and ``convert_pyomo2casadi`` can retain selected Params. The
  default remains numerical conversion. Parameter selection can therefore
  occur at ``compile`` time without modifying the constructor.
* ``_simulate_with_casadi_no_inputs`` already assembles the differential
  state vector and RHS. Share only the small graph-assembly portion if that
  shortens the implementation. Leave legacy trajectory evaluation, profile
  assembly, and Simulator state updates in their current paths.
* The audit corrected mutation of caller options in existing simulation.
  CasADi simulation still uses a deprecated integrator overload. The new
  compile method should use the explicit time-grid overload. Any migration
  of the existing overload needs dependency compatibility checks, rather
  than preservation solely because it is current behavior.

Avoid unrelated rewrites of constraint parsing, the SciPy path, and
varying-input simulation. Small bug fixes in these shared paths are part of
making the contribution correct. If retaining parameters requires a large
rewrite, first look for a smaller supported expression slice and a simpler
implementation.

Validation
----------

The public API tests cover multiple observation times/directions,
initial-state derivatives, parameter ordering and changed parameter values,
integral/explicit terminal
terms, parameter-dependent initial conditions, nonlinear parameter-only
expressions, coupled/stiff ODEs, and tolerance refinement. Finite-difference
step sweeps and forward/adjoint identities supplement the analytic checks.

Tests cover selected subsets and indexed mutable entries, legitimate zero
derivatives, unused/folded selections, option validation, and state/options
preservation after forced failures. The existing Simulator tests remain in
the full DAE suite; optional-dependency checks run with CasADi absent.
New unit tests use ``pyomo.common.unittest`` and guarded optional imports,
without mocks. The executable example is
``examples/dae/simulator_sensitivity_example.py``.

Work package B added twenty compile test methods; its historical full-suite
result is 250 tests and 196 subtests per version. Work package C adds fifteen
methods for control ordering/types, dense and sparse observations, switch
endpoints, narrow pulses, empty parameters, nonlinear amplitudes, physical
time, normalized duration parameters, option/state preservation and invalid
selections. Exact affine phase formulas check states, integral states and
derivatives with respect to x0, p and every amplitude. Two-direction JVPs
and VJPs pass analytic checks and forward-adjoint identities. A composed cost
with nonlinear ICs, shared controls, an integral state and explicit parameter
terms passes analytic checks and a four-step finite-difference sweep.

The full DAE suite now passes 265 tests and 232 subtests on both CasADi 3.8.1
and 3.8.2. Without NumPy, SciPy or CasADi, the four Simulator test modules run
127 tests: seven pass and 120 skip. Current coverage and executable guide and
example checks are recorded in ``results/controls_validation.json``. The new
example is ``examples/dae/simulator_control_sensitivity_example.py`` and
requires no optimizer or ASL extension.

Fresh line coverage is 95.6% for Simulator and 97.1% for compile. The three
uncovered compile lines are defensive refusals for missing NumPy, older
CasADi versions and control expressions not indexed by the current time.

The `Pyomo contribution guide
<https://pyomo.readthedocs.io/en/stable/contribution_guide.html>`_ and
`development principles <https://pyomo.readthedocs.io/en/stable/principles.html>`_
call for focused changes, backward compatibility, tests, documentation,
Black formatting and typo checks. Their AI policy requires disclosure and
human responsibility for contributions. These probes, reductions and notes
were prepared with Codex assistance; human review of the mathematics and
API boundary is still needed before submission. This is a local
implementation; no maintainer agreement is implied.

The native Function derivative API is documented in the
`CasADi sensitivity guide
<https://web.casadi.org/docs/#initial-value-problems-and-sensitivity-analysis>`_.
The measured dependency qualifications and raw evidence are in ``README.rst``
and ``results/``. Work package A supports proceeding with a small ODE patch;
it does not establish a public DAE sensitivity contract.
