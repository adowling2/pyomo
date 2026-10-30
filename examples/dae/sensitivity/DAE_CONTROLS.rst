Fixed-grid DAE controls and switch derivatives
==============================================

G6, 30 October 2026. Parent implementation: 78818eef2 (G5). This note fixes
the control convention and native qualification before enabling production
DAE controls. The public interface keeps the existing controls/control_times
keywords and native u matrix layout; no new compile argument is proposed.

Switch convention
-----------------

* Boundaries include both ContinuousSet bounds and define fixed intervals.
  Control amplitudes are right-held: column j applies on [s_j, s_(j+1)).
* Differential states, including integral states, are continuous at a switch.
  Algebraic states can jump. Observed zf at an internal switch satisfies g
  with the new amplitude and the continuous differential state.
* The terminal observation uses the final interval's amplitude. There is
  no additional control column at the terminal boundary.
* Observation times are independent of control boundaries. Return exactly
  the requested columns. Distinct neighboring floating-point observations
  must not be snapped together, and narrow intervals must not be deleted.
* Amplitudes in intervals starting strictly after the last observation have
  zero partial derivatives for independent x0. An interval starting at the
  last observation can already influence its algebraic value, even though
  its dynamics have not yet contributed.
* Switch-time derivatives, resets and branch-changing event sensitivities
  remain out of scope. Derivatives are local to regular algebraic branches.

Consistency and derivative construction
---------------------------------------

Pack selected scalar parameters and the current interval's controls into
native IDAS parameters. Reuse the normalized unit-interval DAE map and its
consistent-root helper. Carry differential states and algebraic branch
seeds across the union of observations and switches. The initial root uses
the first amplitude; each phase start solves g using that phase's amplitude.
A failed root may not change the differential IC or prior model/cache state.

The accumulated phase's algebraic output is consistent with the interval
just integrated. At requested observations, apply the generic implicit root
again with the right-held amplitude, using the accumulated value only as a
branch seed. This separates continuous state propagation from the algebraic
jump convention without a Python derivative wrapper. Rootfinding supplies
both the direct amplitude term and derivatives through the continuous state.
No zero-duration IDAS integration or explicit model-specific elimination is
needed to implement a jump.

A control may enter f, g or both. Structural-use checks must examine both
sets of equations. Algebraic rows follow the time-varying registry after
selected controls are removed; u rows remain in the explicit control order.
Support the same whole time-only Var/Param components as ODE controls,
including mutable and immutable time-indexed Params. The map still has
x0, p, z0, u inputs and xf/zf outputs. z0 is a branch/solver guess with zero
local derivative on a fixed branch.

Independent native models
-------------------------

The native qualification uses no Pyomo or optimizer. It tests:

* Affine feedback: xdot=k*z+v, qdot=x, z=a*x+u. The control u appears only
  in the algebraic residual, and v only in the differential equations.
  Exact affine phase formulas give states and all independent initial,
  parameter and amplitude partials.
* Nonlinear feedback: xdot=k*z, qdot=x, z*z=a*x+u. On a fixed interval,
  zdot=a*k/2. Integrating this identity gives exact x/q phase formulas;
  the original residual determines the right-side z at observations.
  Positive and negative branches are checked separately.

The probe compares full direct Jacobians, two adjoint directions, forward-
adjoint identities, guess derivatives and four-step finite differences.
Grids include initial-only, exact switches, sparse observations, terminal-
only, dense observations and points immediately on both sides of a switch.
Versioned observations are in results/dae_controls_native-<version>.json;
process completion and accuracy are separate flags. The coarse tolerance
sweep is calibration evidence and is not expected to meet a tight target.

Public-interface exit evidence
------------------------------

Before marking G6 complete, add public tests for the native models and
conventions, multiple control selection order, total IC/control composition,
physical time, nonlinear branches, failed consistency, unused future inputs,
narrow pulses, state/options preservation and both derivative modes. Check
that right-side algebraic derivatives are present even when the observation
is the final time of a truncated trajectory. All ODE/G5 tests must remain
green. Keep native numerical failures visible rather than relaxing error
control or changing the endpoint convention to obtain agreement.

Native qualification result
---------------------------

Seventeen bounded experiments complete on each of CasADi 3.8.1 and 3.8.2.
At reltol=1e-10, abstol=1e-12, every analytic primal/direct/adjoint and
forward-adjoint check passes the scaled 3e-7 target; the largest error across
these grids is below 4.7e-9. The minimum four-step FD errors are below
4.6e-7. Guess adjoints are zero on the fixed branches. The nonlinear
1e-11 refinement has analytic/product error below 3.4e-11. Coarse 1e-7
settings have larger errors and an FD minimum above 1e-6; they remain
calibration evidence rather than proof of the tighter FD criterion.

This qualifies the native phase/observation construction for the bounded
G6 public extension. It does not by itself complete the public-interface
exit evidence or alter the G5 implementation. No solver setting, seed or
switch convention was changed to obtain agreement.
