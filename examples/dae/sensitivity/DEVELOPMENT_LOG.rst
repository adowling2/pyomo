Development checkpoint log
==========================

30 October 2026: G0 baseline and execution record
-------------------------------------------------

Status: passed; this entry and DEVELOPMENT_GOAL.rst are included in the G0
plan commit. Next checkpoint: G1 ODE numerical qualification.

The starting tree is clean at f9a93159b on pyomo-dae-sensitivity. Work package
C supports fixed-grid control amplitudes and total derivatives through
symbolically composed initial conditions. Work package B and the earlier
Simulator fixes are committed in 74e516a11.

Existing evidence: results/controls_validation.json records 265 tests and
232 subtests passing with each of CasADi 3.8.1 and 3.8.2 on Python 3.12.14,
macOS ARM64. Seven tests pass and 120 skip in the lean optional-dependency
check. Four guide blocks and both ODE examples pass on both versions.
These are historical baseline results, not proof of future changes.

Known issues to resolve or qualify: controlled large-grid cost is unmeasured;
the collocation probe does not yet call compile(); a recorded native BDF
adjoint failure occurs for the 1000:1 decay case; native algebraic-terminal
DAE adjoints remain inaccurate in supplied probes. Current compile() refuses
DAEs, vector absolute weights and switch-time derivatives.

The goal has explicit gates G0-G7, acceptance thresholds, failure paths,
commit boundaries and a local-versus-remote validation distinction. Production
DAE changes are gated on ODE closeout and a native initialization/output
contract. This plan does not authorize pushing or external messages.
