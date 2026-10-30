# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tiny, standalone CasADi sensitivity probes; no adsorption dependency."""

from pathlib import Path
import argparse
import json
import math
import resource
import subprocess
import sys
import time

import casadi as ca
import numpy as np
import pyomo

ROOT = Path(__file__).resolve().parent


def array(value):
    """Convert a native CasADi value to a NumPy array."""
    return np.asarray(value, dtype=float)


def require_close(actual, expected, tolerance=3e-7):
    """Check finite values and maximum scaled analytic error."""
    actual, expected = array(actual), array(expected)
    absolute = float(np.max(np.abs(actual - expected)))
    scaled = float(
        np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected)))
    )
    if not np.isfinite(actual).all() or scaled > tolerance:
        raise AssertionError((actual.tolist(), expected.tolist(), scaled))
    return {"max_abs_error": absolute, "max_scaled_error": scaled}


def ode_probe():
    """Check exponential ODE states, quadrature, and derivative products."""
    x, p = ca.MX.sym("x"), ca.MX.sym("p")
    T, a, b = 1.2, 1.3, -0.7
    integrator = ca.integrator(
        "ode",
        "cvodes",
        {"x": x, "p": p, "ode": p * x, "quad": x},
        0.0,
        T,
        {"reltol": 1e-11, "abstol": 1e-12},
    )
    fwd = integrator.factory("ode_fwd", ["x0", "p", "fwd:p"], ["fwd:xf", "fwd:qf"])
    adj = integrator.factory(
        "ode_adj", ["x0", "p", "adj:xf", "adj:qf"], ["adj:x0", "adj:p"]
    )
    value = integrator(x0=a, p=b)
    fv = fwd(x0=a, p=b, fwd_p=1.0)
    av = adj(x0=a, p=b, adj_xf=2.0, adj_qf=3.0)
    e = math.exp(b * T)
    dx, dq = T * a * e, a * ((b * T - 1) * e + 1) / b**2
    return {
        "primal_x": require_close(value["xf"], a * e),
        "primal_q": require_close(value["qf"], a * (e - 1) / b),
        "forward_x_parameter": require_close(fv["fwd_xf"], dx),
        "forward_q_parameter": require_close(fv["fwd_qf"], dq),
        "adjoint_parameter": require_close(av["adj_p"], 2 * dx + 3 * dq),
        "adjoint_initial_state": require_close(av["adj_x0"], 2 * e + 3 * (e - 1) / b),
    }


def dae_probe(
    *,
    vector_weights=False,
    expanded_weights=False,
    directions=3,
    reverse_weights=None,
    recover_algebraic_output=False,
):
    """Check an index-one DAE against its analytic reduced solution."""
    x, z, p = ca.MX.sym("x"), ca.MX.sym("z"), ca.MX.sym("p", 2)
    T, a, k, gain = 1.2, 1.3, 0.7, 1.4
    options = {
        "reltol": 1e-10,
        "abstol": 1e-12,
        "calc_ic": True,
        "suppress_algebraic": False,
    }
    if vector_weights:
        options["abstolv"] = [1e-12, 1e-11]
    if expanded_weights:
        options["augmented_options"] = {
            "abstolv": [1e-12] * (directions + 1) + [1e-11] * (directions + 1),
            "suppress_algebraic": False,
        }
    I = ca.integrator(
        "dae",
        "idas",
        {"x": x, "z": z, "p": p, "ode": -p[0] * z, "alg": z - p[1] * x, "quad": x},
        0.0,
        T,
        options,
    )
    w = ca.MX.sym("w", 3)
    result = I(x0=w[0], z0=1.0, p=w[1:])  # deliberately only a guess for z0
    algebraic_output = w[2] * result["xf"] if recover_algebraic_output else result["zf"]
    F = ca.Function(
        "dae_map", [w], [ca.vertcat(result["xf"], algebraic_output, result["qf"])]
    )
    point = np.array([a, k, gain])
    value = F(point)
    seed = np.eye(3)[:, :directions]
    derivative = F.forward(directions)(point, value, seed)
    r, e = -k * gain, math.exp(-k * gain * T)
    exp_integral = (e - 1) / r
    integral_r = ((r * T - 1) * e + 1) / r**2
    expected = np.array(
        [
            [e, -gain * T * a * e, -k * T * a * e],
            [gain * e, -(gain**2) * T * a * e, a * e - gain * k * T * a * e],
            [exp_integral, -gain * a * integral_r, -k * a * integral_r],
        ]
    )
    records = {
        "primal": require_close(
            value, np.array([a * e, gain * a * e, a * exp_integral]).reshape(3, 1)
        ),
        "forward": require_close(derivative, expected[:, :directions]),
    }
    if reverse_weights is not None:
        weight = np.array(reverse_weights)
        rev = F.reverse(1)(point, value, weight)
        records["reverse_all_outputs"] = require_close(
            rev, (weight @ expected).reshape(3, 1)
        )
    return records


def control_probe():
    """Check control amplitudes in one native CVODES control grid."""
    x, u = ca.MX.sym("x"), ca.MX.sym("u")
    times = np.array([0.2, 0.7, 1.2])
    a, p = 0.3, -0.7
    I = ca.integrator(
        "controls",
        "cvodes",
        {"x": x, "u": u, "ode": p * x + u},
        0.0,
        times.tolist(),
        {"reltol": 1e-11, "abstol": 1e-12},
    )
    U = ca.MX.sym("U", 1, 3)
    trajectory = I(x0=a, u=U)["xf"]
    F = ca.Function("controlled_map", [U], [trajectory[0, -1]])
    values = np.array([[0.1, 0.4, -0.2]])
    out = F(values)
    jac = ca.Function("controlled_jac", [U], [ca.jacobian(trajectory[0, -1], U)])(
        values
    )
    previous, expected = 0.0, a
    derivative = []
    for end, val in zip(times, values.ravel()):
        delta = end - previous
        expected = math.exp(p * delta) * expected + val * (math.exp(p * delta) - 1) / p
        derivative.append(
            math.exp(p * (times[-1] - end)) * (math.exp(p * delta) - 1) / p
        )
        previous = end
    return {
        "primal": require_close(out, expected),
        "control_amplitudes": require_close(jac, [derivative]),
    }


def parameter_ic_probe():
    """Check the chain rule through a parameter-dependent initial state."""
    x, z, p = ca.MX.sym("x"), ca.MX.sym("z"), ca.MX.sym("p")
    T = 0.8
    I = ca.integrator(
        "parameter_ic",
        "idas",
        {"x": x, "z": z, "p": p, "ode": p * z, "alg": z * z - x},
        0.0,
        T,
        {"reltol": 1e-11, "abstol": 1e-12, "calc_ic": True},
    )
    theta = ca.MX.sym("theta")
    result = I(x0=theta**2, z0=theta + 0.1, p=theta)
    final = ca.vertcat(result["xf"], result["zf"])
    F = ca.Function("dependent_ic", [theta], [final, ca.jacobian(final, theta)])
    point = 1.3
    value, derivative = F(point)
    return {
        "primal": require_close(
            value, [[point**2 * (1 + T / 2) ** 2], [point * (1 + T / 2)]]
        ),
        "total_derivative_including_IC": require_close(
            derivative, [[2 * point * (1 + T / 2) ** 2], [1 + T / 2]]
        ),
    }


def segmented_control_probe():
    """Check derivatives through explicit fixed control segments."""
    x, p = ca.MX.sym("x"), ca.MX.sym("p")
    U = ca.MX.sym("U", 3)
    endpoint = 0.3
    previous = 0.0
    T, a, rate = 1.2, 0.3, -0.7
    expected = a
    derivative = []
    for i, end in enumerate([0.2, 0.7, T]):
        delta = end - previous
        I = ca.integrator(
            "segment" + str(i),
            "cvodes",
            {"x": x, "p": p, "ode": rate * x + p},
            0.0,
            delta,
            {"reltol": 1e-11, "abstol": 1e-12},
        )
        endpoint = I(x0=endpoint, p=U[i])["xf"]
        expected = (
            math.exp(rate * delta) * expected
            + [0.1, 0.4, -0.2][i] * (math.exp(rate * delta) - 1) / rate
        )
        derivative.append(
            math.exp(rate * (T - end)) * (math.exp(rate * delta) - 1) / rate
        )
        previous = end
    F = ca.Function("segmented_controls", [U], [endpoint])
    value = F([0.1, 0.4, -0.2])
    forward = F.forward(3)([0.1, 0.4, -0.2], value, np.eye(3))
    reverse = F.reverse(1)([0.1, 0.4, -0.2], value, 1.0)
    return {
        "primal": require_close(value, expected),
        "forward": require_close(forward, [derivative]),
        "adjoint": require_close(reverse, np.array(derivative).reshape(3, 1)),
    }


def cases():
    """Return the supplied independent native CasADi probe cases."""
    return {
        "analytic_ODE_forward_and_adjoint": ode_probe,
        "analytic_index1_DAE_scalar_weights": dae_probe,
        "analytic_index1_DAE_vector_weights_no_expansion": lambda: dae_probe(
            vector_weights=True, directions=1
        ),
        "analytic_index1_DAE_vector_weights_expanded_one": lambda: dae_probe(
            vector_weights=True, expanded_weights=True, directions=1
        ),
        "analytic_index1_DAE_vector_weights_expanded_two": lambda: dae_probe(
            vector_weights=True, expanded_weights=True, directions=2
        ),
        "analytic_index1_DAE_adjoint_x_and_q": lambda: dae_probe(
            reverse_weights=[2.0, 0.0, 4.0]
        ),
        "analytic_index1_DAE_adjoint_z_only": lambda: dae_probe(
            reverse_weights=[0.0, 1.0, 0.0]
        ),
        "analytic_index1_DAE_adjoint_x_z_q": lambda: dae_probe(
            reverse_weights=[2.0, 3.0, 4.0]
        ),
        "analytic_index1_DAE_adjoint_recovered_z": lambda: dae_probe(
            reverse_weights=[0.0, 1.0, 0.0], recover_algebraic_output=True
        ),
        "piecewise_control_amplitudes": control_probe,
        "segmented_control_amplitudes": segmented_control_probe,
        "parameter_dependent_consistent_IC": parameter_ic_probe,
    }


def main():
    """Run probes and record numerical errors and dependency failures."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=cases())
    parser.add_argument('--output', type=Path, default=ROOT / 'PROBE_RESULTS.json')
    args = parser.parse_args()
    if args.case:
        name = args.case
        try:
            result = {"execution_passed": True, "checks": cases()[name]()}
        except Exception as error:
            result = {"execution_passed": False, "error": str(error)[-2200:]}
        print("RESULT " + json.dumps(result))
        return
    started = time.monotonic()
    report = {
        "pyomo_version": pyomo.__version__,
        "casadi_version": ca.__version__,
        "scope": "Standalone tiny native CasADi probes; no Pyomo interface extension or PSA solve",
        "native_failure_containment": "Each case runs in an isolated subprocess with 30-second timeout; core dumps disabled",
        "cases": {},
    }
    for name in cases():
        tic = time.monotonic()
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            results = [
                line[len("RESULT ") :]
                for line in run.stdout.splitlines()
                if line.startswith("RESULT ")
            ]
            result = json.loads(results[-1]) if results else {"execution_passed": False}
            result.update(returncode=run.returncode, stderr_tail=run.stderr[-3000:])
            if run.returncode != 0:
                result["execution_passed"] = False
        except subprocess.TimeoutExpired as error:
            result = {"execution_passed": False, "timeout_s": 30, "error": str(error)}
        result["elapsed_s"] = time.monotonic() - tic
        report["cases"][name] = result
    report["elapsed_s"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "execution_passed": item["execution_passed"],
                    "returncode": item.get("returncode"),
                }
                for name, item in report["cases"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
