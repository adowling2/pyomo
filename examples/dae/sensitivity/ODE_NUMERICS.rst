ODE numerical qualification
===========================

G1, 30 October 2026. These checks extend the ODE evidence after f9a93159b;
no production Simulator changes were required. Results are recorded in
results/ode_numerics-3.8.1.json and results/ode_numerics-3.8.2.json.
The source hashes and actual environment versions are in each report.

Nonlinear controlled states and scaling
---------------------------------------

The model has two independent rational-decay states with a shared rate and
piecewise control: xdot_i = -(k+u)*x_i*x_i/scale_i, with scale = (1, 1e6).
Its exact solution uses the integrated piecewise exposure a(t):
x_i(t) = x_i(0)/(1 + x_i(0)*a(t)/scale_i). The tests differentiate this
formula independently for x0, k and every control amplitude. Two directions
and two output weights exercise the full input/output block layout.

Errors are measured in dimensionless state units. Tangent seeds are scaled
with state units; adjoint weights are inversely scaled. Scalar reltol/abstol
pairs are (1e-3, 1e-5), (1e-7, 1e-9), and (1e-11, 1e-13). On both tested
CasADi versions, tight primal error is below 2.2e-10, JVP/VJP errors below
7e-11, and the largest forward-adjoint discrepancy below 1.4e-11. All four
quantities improve by much more than ten relative to the coarse setting.

A shared symbolic composition supplies nonlinear initial conditions,
repeated control amplitudes and an explicit parameter/control cost term.
All three total derivatives match the analytic gradient within 1.1e-10.
A centered finite-difference sweep from 1e-2 to 1e-5 reduces maximum error
from about 6.9e-5 to below 2.0e-10. The tests depend only on NumPy and CasADi;
no optimizer or ASL library is needed.

These scale factors test a particular model; scalar tolerances do not create
a general unit-aware sensitivity policy. For states in different units,
write the model in scaled variables y_i=x_i/scale_i and convert outputs and
seeds explicitly. The resulting equations here are ydot_i=-(k+u)*y_i*y_i.
This is ordinary model nondimensionalization, without changing integrator
options silently. Vector weights remain outside the compile contract.

Reduced stiff-adjoint limitation
--------------------------------

The failure reduces to one scalar equation xdot=-p*x with p=1000, x0=1 and
observations [0, 0.0001, 0.01, 0.1, 1]. Seed every observation with one and
differentiate their sum with respect to p. This removes the slow mode,
state scaling and Pyomo expression conversion from the native reproducer.
Primal and direct derivatives pass; default BDF reverse integration fails
with CV_TOO_MUCH_WORK. Native and public compile maps give matching outcomes
on both CasADi versions.

The bounded experiment matrix also checks the original two-state fast/slow
model. Changing adjoint interpolation, increasing the step allowance to
50000 or limiting the primal step to 0.001 does not resolve these cases.
Checkpoint spacing of one resolves the two-state case, with derivative error
below 2.5e-15, but still fails for the scalar reduction. An explicitly chosen
Adams configuration passes both models, with errors below 9e-14. This is a
case-specific alternative, not a general recommendation for stiff dynamics.
No fallback or altered default has been added to Simulator.

The observations localize the problem to native sensitivity integration,
not the Pyomo bridge. They do not identify the exact cause within
CasADi/SUNDIALS or establish that all BDF adjoints fail. Each of 24 native
and public experiments per version has a separate process and 30-second
limit; failures and accurate derivatives have distinct report fields.

The options are described in the `CasADi Python API
<https://web.casadi.org/python-api/index.html>`_. SUNDIALS describes adjoint
checkpoint interpolation in its `CVODES mathematics documentation
<https://sundials.readthedocs.io/en/latest/cvodes/Mathematics_link.html>`_.
These references describe mechanisms; the numerical conclusions above
come from the recorded experiments.

Reproduction
------------

From the checkout root, for each supported CasADi environment:

.. code-block:: sh

   PYTHONPATH="$PWD" <env-python> examples/dae/sensitivity/ode_qualification.py \
       --output /tmp/ode-numerics.json
   PYTHONPATH="$PWD" <env-python> -m pytest \
       pyomo/dae/tests/test_simulator_ode_qualification.py -q

For full-suite runs in parallel environments, use separate working
directories and an absolute path to pyomo/dae/tests. An existing
ContinuousSet I/O test writes diffset.dat in the current working directory;
parallel runs sharing that directory can delete each other's fixture.
This harness collision is separate from integration behavior.
