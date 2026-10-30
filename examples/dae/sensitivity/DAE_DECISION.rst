Index-one DAE API and numerical decision
========================================

G4, 30 October 2026. ODE closeout is committed in f2fcaa0cc. This note
specifies the next bounded implementation; Simulator.compile still rejects
DAEs at this checkpoint. Native evidence is in results/dae_native-3.8.1.json
and results/dae_native-3.8.2.json, with an aggregate in
results/dae_decision.json. No production DAE change has been made yet.

Decision
--------

Extend the existing compile method to semi-explicit index-one DAEs using
explicit integrator='idas'. Keep the ODE defaults and contract. Return the
same native CasADi Function type, with additional DAE input z0 and output
zf. Here x0 is an independent differential initial state, including any
ordinary integral states. z0 is an algebraic solver guess and branch seed,
not an independent physical initial condition.

Retain IDAS for the implicit dynamics and native direct/adjoint integration.
Compose normalized interval maps over observations, reusing the existing
fixed-grid composition approach. At initialization and observations, solve
the original algebraic residual with CasADi's differentiable rootfinder.
Expose the consistent root as zf. This is a general implicit reconstruction,
not a model-specific elimination formula or correction to gradient numbers.
The differential equations retain their algebraic variables inside IDAS.

Why a raw IDAS map is insufficient
----------------------------------

Three native models exercise different algebraic structures:

* Linear: xdot=-k*z, qdot=x, z=a*x.
* Nonlinear branch: xdot=k*z, qdot=x, z*z=a*x, on either regular branch.
* Coupled algebraic rows: xdot=-k*z1, qdot=z2,
  z1=a*x, z2=z1*z1+b. The parameter b occurs only in the residual equations.

Each has a closed-form solution and hand-derived independent x0/q0/parameter
partials. The matrix tests all observation rows, one and several directions,
terminal x, integral q, terminal z and mixed observation costs. It also
checks a total cost with nonlinear parameter-dependent initial states,
explicit parameter terms, four FD steps and forward-adjoint identities.
There are 58 bounded subprocess experiments per CasADi version.

Native multi-observation IDAS gives inaccurate algebraic-terminal adjoints.
For example, on the linear model at reltol=1e-11, the raw z-terminal scaled
error is about 1.69e-3 with 3.8.2. Its projected output gives about 2.05e-11.
The nonlinear raw algebraic-terminal error is about 1.54e-2. Tightening
primal tolerances does not establish a trustworthy raw algebraic-output
adjoint contract. Returning raw zf and asking callers to avoid certain
reverse seeds would be unenforceable on an ordinary Function.

A second failure occurs with mixed weights at several observations: native
backward IDAS integration can fail at a change in observation weights, even
when algebraic terminal outputs are reconstructed. Explicit calc_icB=True
alone does not resolve it. Composing one normalized IDAS map per observation
interval resolves the recorded mixed-weight cases without weakening error
control, changing seeds or introducing a numerical integrator in Pyomo.
This same construction will later accommodate fixed-grid controls.

Verified native contract
------------------------

At reltol=1e-10 and abstol=1e-12, every advertised primal/direct/adjoint mode
of the composed, reconstructed maps passes the scaled 3e-7 analytic target
on all three models and both versions. Maximum analytic/product error is
below 1e-9. Independent-input FD error is below 3.7e-9; the total IC/cost FD
sweep has minimum error below 3.8e-7. At reltol=1e-11, all modes also pass;
total FD errors are below 2.1e-8. The larger coupled independent-input FD
error there is about 1.4e-7, consistent with adaptive-solve noise amplified
by small FD steps. Analytic direct and adjoint errors remain much smaller.

Initial-only maps give correct algebraic initial derivatives without any
positive-duration integration. Changing guesses within a branch leaves
physical initial states, outputs and derivatives unchanged to the recorded
accuracy. Positive and negative branch seeds yield their distinct expected
solutions. The local derivative with respect to the guess is zero in both
modes, while derivatives through x0 and parameters include the algebraic
initial dependence.

These observations qualify a bounded initial implementation. They do not
promise convergence for arbitrary index-one systems or all option values.
The coarse tolerance sweep deliberately misses the accuracy target. At
reltol=1e-12 one coupled total-cost adjoint fails native IDACalcICB with
IDA_LINESEARCH_FAIL on both versions, although other products at that point
are accurate when they converge. A near-singular branch example also fails
native integration. Retain these records and propagate backend failure;
do not silently relax tolerances, retry another algorithm, change the
initial state or label a failed solve as a verified derivative.

Initialization and output mathematics
-------------------------------------

Let g(t,x,z,p)=0 and assume g_z is nonsingular on the selected branch. The
consistent root Z(t,x,p) has local derivatives Z_x=-solve(g_z,g_x) and
Z_p=-solve(g_z,g_p). A terminal cost involving z therefore contributes
Z_x.T*Phi_z to the differential output weight and Z_p.T*Phi_z to the direct
parameter term. CasADi differentiates the implicit root using these equations,
not by differentiating a fixed count of Newton iterations.

Use that root at t0 with fixed differential x0, and again at each observation
with its computed differential state and physical time. Seed later root
solves using IDAS's algebraic values to follow the local branch; their guess
derivatives are zero. The incorrect native algebraic-output adjoint is not
used as an algebraic output derivative path. Mixed observation weights pass
through the composed phase functions and their root projections.

Total derivatives remain ordinary symbolic composition: x0=h(theta),
p=p(theta), followed by a symbolic cost. No new IC mapping class or callback
is needed. Derivatives are local to a branch. Moving a guess across branches
can change the solution and is not a smooth sensitivity operation.

Proposed public interface
-------------------------

No additional compile keyword is required for this initial extension. The
existing integrator selector must be 'idas' for DAEs; 'cvodes' remains the
ODE default. Illustrative G5 usage, not an already implemented method:

.. code-block:: python

   flow = sim.compile(
       parameters=[model.k, model.a],
       times=[0, 0.2, 0.7, 1.2],
       integrator='idas',
       integrator_options={'reltol': 1e-10, 'abstol': 1e-12},
   )
   result = flow(x0=[1.3, 0], p=[0.7, 1.4], z0=[1])
   adjoint = flow.factory(
       'adjoint', ['x0', 'p', 'z0', 'adj:xf', 'adj:zf'],
       ['adj:x0', 'adj:p'],
   )

Inputs are x0, p, z0, plus u when controls are selected. Outputs xf and zf
have one column per requested observation. Differential rows follow
get_variable_order(); algebraic rows follow the remaining entries of
get_variable_order('time-varying') after selected controls are removed.
Parameter and control orders retain the ODE contract. Algebraic guesses
must be supplied in that algebraic order. A native omitted z0 defaults to
zero, which can be an invalid nonlinear guess; examples should supply a
meaningful guess explicitly. Do not infer it from model values or fixity.

Direct/adjoint factories support both outputs and multiple directions.
A factory may request the guess derivative, which is locally zero. This
does not make z0 an independent physical IC. Explicit selected parameters
must remain symbolic in both f and g, including parameters used only in g.
Reject nonsquare algebraic systems and structurally deficient g_z clearly.
Purely algebraic systems and higher-index systems remain outside scope.

Numerical policy
----------------

* Retain CasADi >=3.8.1 and the 3.8.1 single-direction native batching limit.
  Multi-direction public products still use native composition.
* Require quad_err_con=True, calc_ic=True, calc_icB=True and
  suppress_algebraic=False. The matrix retains algebraic error control.
  Disabling calc_ic fails a tested DAE even after a consistent algebraic
  root is supplied, because consistent initial derivatives are also needed.
  Do not expose a separate initial-derivative guessing interface now.
* Keep scalar positive reltol/abstol. Refuse vector and derivative-augmentation
  overrides as for ODEs. Preserve user-requested supported tolerances.
* Use native Newton on the original residual. Set its residual target to
  the scalar abstol, disable step-size-only termination (abstolStep=0), and
  bound iterations at 100. Require error_on_fail and regularity_check.
  Step-only termination otherwise permits stopping before the requested
  residual target. This is a solver configuration, not a new Newton method.
* Check finite roots, residuals and local Jacobian determinant; require the
  determinant to be nonzero and the residual to meet its target. Native
  Newton can otherwise return a nonfinite value from a singular guess.
  The postcondition also refuses an exact singular root even when the
  initial residual is already zero. Initial-only observations must use the
  same checks. Do not set an arbitrary dimensionful conditioning threshold.
* Scale state and residual equations appropriately. One scalar tolerance
  is not a unit-aware policy. The determinant check is a local numerical
  refusal, not global index analysis or a guarantee of good conditioning.
  The caller must stay on an index-one branch throughout integration;
  checks at initial/observation points do not detect every interior event.

The initial implementation should use documented scalar defaults when an
absolute tolerance is omitted and state those defaults explicitly. A
separate algebraic-solver configuration API is deferred until a concrete
case needs independent settings; do not add a speculative solver registry.
All solver options must be copied and no model, Simulator cache or caller
array may be mutated by graph construction or failed evaluation.

G5 implementation boundary
--------------------------

Reuse the existing equation analysis, identity-based parameter substitution
and normalized fixed-grid graph assembly. Add the small consistent-root
helper and the IDAS phase path. Preserve the uncontrolled ODE path exactly.
Keep graph construction separate from numerical evaluation. Do not expose
raw zf through an alternate public path with different derivative semantics.

First implement autonomous DAEs and public-interface tests mirroring this
native evidence. Test both nonlinear branches, x0(theta), integral and
explicit costs, initial-only observations, parameters occurring only in g,
multiple directions, ordering, invalid/singular guesses and state/options
preservation. Root failure must not overwrite a previously simulated model.
Then G6 adds controls and specifies right-side algebraic observations at
switches, with consistency and its derivatives propagated through phases.

G4 passes as a native design gate once the aggregate checks and source
hashes are verified; it does not complete G5 or qualify DAE controls. The
1e-12 consistency failure and near-singular failure are outstanding native
limitations, not reasons to preserve an inaccurate algebraic-output API.

Sources and provenance
----------------------

The executable probe contains uneliminated f/g equations and generic root
reconstruction; analytic elimination appears only in the independent oracle.
Each process has a 30-second limit and core dumps disabled. Native and
projected accuracy flags, failures and process exits are distinct. A zero
script exit code does not imply that every numerical case passed.

CasADi describes the implicit root derivative contract in its
`rootfinding guide <https://web.casadi.org/docs/#nonlinear-root-finding-problems>`_
and the rootfinder `API <https://web.casadi.org/api/html/d3/d65/group__rootfinder.html>`_.
IDAS options are also queried from the installed plugin documentation in
this investigation. Those sources explain the mechanism; the measured
support claims above come from the versioned experiment records.

This investigation and decision note were prepared with Codex assistance.
Human review is needed before upstream submission. No upstream issue, PR
or external message has been published.
