# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Reduce CasADi sensitivity discrepancies without importing Pyomo.

Each case runs in a subprocess with a 30 second timeout and core dumps disabled.
Results describe measured behavior; completing this script does not mean that
all derivative modes passed their analytic checks.
"""

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import casadi as ca
import numpy as np


def error(actual, expected):
    """Return the largest absolute error against an analytic value."""
    return float(np.max(np.abs(np.asarray(actual, dtype=float) - expected)))


def ode_observations():
    """Check an ODE without quadratures at several observation times."""
    x, p, w = ca.MX.sym('x'), ca.MX.sym('p'), ca.MX.sym('w', 2)
    times = np.array([0.2, 0.7, 1.2])
    integrate = ca.integrator(
        'flow',
        'cvodes',
        {'x': x, 'p': p, 'ode': p * x},
        0,
        times.tolist(),
        {'reltol': 1e-11, 'abstol': 1e-12},
    )
    trajectory = integrate(x0=w[0], p=w[1])['xf'].T
    flow = ca.Function('observations', [w], [trajectory])
    point = [1.3, -0.7]
    value = flow(point)
    expected_value = point[0] * np.exp(point[1] * times)
    expected_jacobian = np.column_stack(
        [np.exp(point[1] * times), times * expected_value]
    )
    weight, direction = np.array([0.3, -0.2, 0.7]), np.array([0.2, -0.4])
    forward = np.asarray(flow.forward(2)(point, value, np.eye(2)))
    reverse = np.asarray(flow.reverse(1)(point, value, weight)).ravel()
    objective = 2 * trajectory[-1] + w[1] ** 2
    gradient = ca.Function('gradient', [w], [ca.gradient(objective, w)])(point)
    theta = ca.MX.sym('theta')
    dependent_ic = flow(ca.vertcat(theta**2, theta))[-1]
    total = ca.Function('total', [theta], [ca.jacobian(dependent_ic, theta)])(point[1])
    expected_total = math.exp(point[1] * times[-1]) * (
        2 * point[1] + times[-1] * point[1] ** 2
    )
    observations = {
        'primal_error': error(value, expected_value.reshape(3, 1)),
        'forward_error': error(forward, expected_jacobian),
        'reverse_error': error(reverse, weight @ expected_jacobian),
        'duality_error': abs(float(weight @ forward @ direction - direction @ reverse)),
        'explicit_objective_gradient_error': error(
            gradient,
            [
                [2 * expected_jacobian[-1, 0]],
                [2 * expected_jacobian[-1, 1] + 2 * point[1]],
            ],
        ),
        'parameter_dependent_ic_error': error(total, expected_total),
    }
    if max(observations.values()) > 3e-7:
        raise AssertionError(observations)
    return observations


def ode_adjoint_quadrature_control():
    """Measure parameter-adjoint accuracy with and without quadrature error control."""
    rows = []
    for enabled in (False, True):
        x, p, time = ca.SX.sym('x'), ca.SX.sym('p'), ca.SX.sym('t')
        options = {'reltol': 1e-11, 'abstol': 1e-13, 'quad_err_con': enabled}
        integrate = ca.integrator(
            'flow',
            'cvodes',
            {'x': x, 'p': p, 't': time, 'ode': p * time},
            2,
            [2, 2.2, 3],
            options,
        )
        gradient = integrate.factory('gradient', ['x0', 'p', 'adj:xf'], ['adj:p'])(
            x0=1, p=0.3, adj_xf=ca.DM.ones(1, 3)
        )['adj_p']
        expected = sum((t * t - 4) / 2 for t in (2, 2.2, 3))
        rows.append(
            {
                'options': options,
                'actual': float(gradient),
                'analytic': expected,
                'absolute_error': abs(float(gradient) - expected),
            }
        )
    return rows


def ode_adjoint_batching():
    """Compare batched ODE adjoints with an analytic Jacobian and native serial calls."""
    times = np.array([0.2, 0.7, 1.2])
    point, initial = [-0.7, -1.2, 0.3], [1.3, -0.2]
    weights = np.array(
        [[0.2, -0.4, 0.7, 0.3, 0.1, -0.2], [-0.1, 0.5, 0.2, 0.6, -0.3, 0.4]]
    )
    e0, e1 = np.exp(point[0] * times), np.exp(point[1] * times)
    jacobian = np.zeros((2, 3, 3))
    jacobian[0, :, 0] = (
        initial[0] * times * e0
        + point[2] * ((point[0] * times - 1) * e0 + 1) / point[0] ** 2
    )
    jacobian[1, :, 1] = initial[1] * times * e1
    jacobian[0, :, 2] = np.expm1(point[0] * times) / point[0]
    expected = np.column_stack(
        [
            np.einsum('it,itp->p', weights[:, 3 * i : 3 * (i + 1)], jacobian)
            for i in range(2)
        ]
    )
    rows = []
    for limit in (None, 1):
        x, p = ca.SX.sym('x', 2), ca.SX.sym('p', 3)
        options = {'reltol': 1e-11, 'abstol': 1e-13, 'quad_err_con': True}
        if limit is not None:
            options['max_num_dir'] = limit
        integrate = ca.integrator(
            'flow',
            'cvodes',
            {'x': x, 'p': p, 'ode': ca.vertcat(p[0] * x[0] + p[2], p[1] * x[1])},
            0,
            times.tolist(),
            options,
        )
        x0, parameters = ca.MX.sym('x0', 2), ca.MX.sym('p', 3)
        flow = ca.Function(
            'observations',
            [x0, parameters],
            [integrate(x0=x0, p=parameters)['xf']],
            ['x0', 'p'],
            ['xf'],
        )
        value = flow(x0=initial, p=point)['xf']
        batch = flow.reverse(2)(x0=initial, p=point, out_xf=value, adj_xf=weights)[
            'adj_p'
        ]
        single = np.hstack(
            [
                np.asarray(
                    flow.reverse(1)(
                        x0=initial,
                        p=point,
                        out_xf=value,
                        adj_xf=weights[:, 3 * i : 3 * (i + 1)],
                    )['adj_p']
                )
                for i in range(2)
            ]
        )
        rows.append(
            {
                'options': options,
                'batch': np.asarray(batch).tolist(),
                'single': single.tolist(),
                'analytic': expected.tolist(),
                'batch_error': error(batch, expected),
                'single_error': error(single, expected),
            }
        )
    return rows


def stiff_adjoint_methods():
    """Compare native BDF and Adams adjoints on fast/slow decay modes."""
    rows = []
    times = [0, 1e-4, 0.01, 0.1, 1]
    for rate in (100, 1000):
        for method in ('bdf', 'adams'):
            x, p = ca.SX.sym('x', 2), ca.SX.sym('p')
            options = {
                'reltol': 1e-11,
                'abstol': 1e-13,
                'quad_err_con': True,
                'linear_multistep_method': method,
            }
            row = {'rate': rate, 'options': options, 'times': times}
            try:
                integrate = ca.integrator(
                    'flow',
                    'cvodes',
                    {'x': x, 'p': p, 'ode': ca.vertcat(-x[0], -p * x[1])},
                    0,
                    times,
                    options,
                )
                weights = ca.DM.zeros(2, len(times))
                weights[1, :] = 1
                gradient = integrate.factory(
                    'gradient', ['x0', 'p', 'adj:xf'], ['adj:p']
                )(x0=[1, 1], p=rate, adj_xf=weights)['adj_p']
                expected = -sum(t * math.exp(-rate * t) for t in times)
                row.update(
                    actual=float(gradient),
                    analytic=expected,
                    absolute_error=abs(float(gradient) - expected),
                )
            except RuntimeError as exception:
                row['error'] = str(exception)
            rows.append(row)
    return rows


def algebraic_terminal():
    """Compare native and explicitly reconstructed algebraic terminal outputs."""
    point = np.array([0.7, 1.4])
    horizon, initial = 1.2, 1.3
    k, gain = point
    endpoint = initial * math.exp(-k * gain * horizon)
    expected = np.array(
        [
            [-gain * horizon * endpoint, -k * horizon * endpoint],
            [-(gain**2) * horizon * endpoint, endpoint * (1 - gain * k * horizon)],
        ]
    )
    records = []
    for tolerance in (1e-6, 1e-8, 1e-10, 1e-12):
        x, z, p = ca.MX.sym('x'), ca.MX.sym('z'), ca.MX.sym('p', 2)
        integrate = ca.integrator(
            'flow',
            'idas',
            {'x': x, 'z': z, 'p': p, 'ode': -p[0] * z, 'alg': z - p[1] * x},
            0,
            horizon,
            {'reltol': tolerance, 'abstol': tolerance / 100, 'calc_ic': True},
        )
        result = integrate(x0=initial, z0=1, p=p)
        for reconstructed in (False, True):
            terminal = p[1] * result['xf'] if reconstructed else result['zf']
            flow = ca.Function('outputs', [p], [ca.vertcat(result['xf'], terminal)])
            value = flow(point)
            forward = flow.forward(2)(point, value, np.eye(2))
            reverse = flow.reverse(1)(point, value, [0, 1])
            differences = []
            for step in (1e-3, 1e-4, 1e-5, 1e-6):
                fd = np.column_stack(
                    [
                        np.asarray(
                            (flow(point + step * seed) - flow(point - step * seed))
                            / (2 * step)
                        ).ravel()
                        for seed in np.eye(2)
                    ]
                )
                differences.append({'step': step, 'max_abs_error': error(fd, expected)})
            direction = np.array([0.3, -0.7])
            records.append(
                {
                    'reltol': tolerance,
                    'abstol': tolerance / 100,
                    'reconstructed': reconstructed,
                    'analytic_z_gradient': expected[1].tolist(),
                    'forward_z_gradient': np.asarray(forward)[1].tolist(),
                    'reverse_z_gradient': np.asarray(reverse).ravel().tolist(),
                    'primal_error': error(value, [[endpoint], [gain * endpoint]]),
                    'forward_error': error(forward, expected),
                    'reverse_error': error(reverse, expected[1].reshape(2, 1)),
                    'duality_error': abs(
                        float(np.asarray(forward)[1] @ direction)
                        - float(direction @ np.asarray(reverse).ravel())
                    ),
                    'finite_differences': differences,
                }
            )
    return records


def vector_weights(directions, expanded):
    """Check grouped forward weights on a DAE without quadratures."""
    x, z, p = ca.MX.sym('x'), ca.MX.sym('z'), ca.MX.sym('p', 2)
    options = {'reltol': 1e-10, 'abstolv': [1e-12, 1e-11], 'calc_ic': True}
    if expanded:
        options['augmented_options'] = {
            'abstolv': [1e-12] * (directions + 1) + [1e-11] * (directions + 1)
        }
    integrate = ca.integrator(
        'flow',
        'idas',
        {'x': x, 'z': z, 'p': p, 'ode': -p[0] * z, 'alg': z - p[1] * x},
        0,
        1.2,
        options,
    )
    w = ca.MX.sym('w', 3)
    result = integrate(x0=w[0], z0=1, p=w[1:])
    flow = ca.Function('outputs', [w], [ca.vertcat(result['xf'], result['zf'])])
    point = [1.3, 0.7, 1.4]
    derivative = flow.forward(directions)(point, flow(point), np.eye(3)[:, :directions])
    e = math.exp(-0.7 * 1.4 * 1.2)
    expected = np.array(
        [[e, -1.4 * 1.2 * 1.3 * e], [1.4 * e, -(1.4**2) * 1.2 * 1.3 * e]]
    )
    return {
        'directions': directions,
        'options': options,
        'forward_error': error(derivative, expected[:, :directions]),
    }


def controls(segmented):
    """Compare one native control grid with explicit interval composition."""
    times, values = [0.2, 0.7, 1.2], np.array([0.1, 0.4, -0.2])
    rate, initial = -0.7, 0.3
    expected, previous, gradient = initial, 0, []
    for end, amplitude in zip(times, values):
        duration = end - previous
        expected = (
            math.exp(rate * duration) * expected
            + amplitude * math.expm1(rate * duration) / rate
        )
        gradient.append(
            math.exp(rate * (times[-1] - end)) * math.expm1(rate * duration) / rate
        )
        previous = end
    records = []
    for tolerance in (1e-7, 1e-9, 1e-11):
        options = {'reltol': tolerance, 'abstol': tolerance / 10}
        x, u, amplitudes = ca.MX.sym('x'), ca.MX.sym('u'), ca.MX.sym('amplitudes', 3)
        try:
            if segmented:
                endpoint, previous = initial, 0
                for index, end in enumerate(times):
                    integrate = ca.integrator(
                        'segment',
                        'cvodes',
                        {'x': x, 'p': u, 'ode': rate * x + u},
                        0,
                        end - previous,
                        options,
                    )
                    endpoint = integrate(x0=endpoint, p=amplitudes[index])['xf']
                    previous = end
            else:
                integrate = ca.integrator(
                    'grid',
                    'cvodes',
                    {'x': x, 'u': u, 'ode': rate * x + u},
                    0,
                    times,
                    options,
                )
                endpoint = integrate(x0=initial, u=amplitudes.T)['xf'][0, -1]
            flow = ca.Function('terminal', [amplitudes], [endpoint])
            value = flow(values)
            forward = flow.forward(3)(values, value, np.eye(3))
            reverse = flow.reverse(1)(values, value, 1)
            records.append(
                {
                    'options': options,
                    'primal_error': error(value, expected),
                    'forward_error': error(forward, [gradient]),
                    'reverse_error': error(reverse, np.array(gradient).reshape(3, 1)),
                }
            )
        except RuntimeError as exception:
            records.append({'options': options, 'error': str(exception)})
    return records


def main():
    """Run isolated dependency cases and retain failures as observations."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    cases = {
        'ode_observations': ode_observations,
        'ode_adjoint_quadrature_control': ode_adjoint_quadrature_control,
        'ode_adjoint_batching': ode_adjoint_batching,
        'stiff_adjoint_methods': stiff_adjoint_methods,
        'algebraic_terminal': algebraic_terminal,
        'vector_unexpanded': lambda: vector_weights(1, False),
        'vector_expanded_one': lambda: vector_weights(1, True),
        'vector_expanded_two': lambda: vector_weights(2, True),
        'controls_native': lambda: controls(False),
        'controls_segmented': lambda: controls(True),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=cases)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        try:
            record = {'completed': True, 'observations': cases[args.case]()}
        except Exception as exception:
            record = {'completed': False, 'error': str(exception)}
        print(json.dumps(record, allow_nan=False))
        return
    report = {
        'casadi': ca.__version__,
        'numpy': np.__version__,
        'python': sys.version,
        'platform': platform.platform(),
        'cases': {},
    }
    for name in cases:
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--case', name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            record = (
                json.loads(run.stdout) if run.returncode == 0 else {'completed': False}
            )
            record.update(returncode=run.returncode, stderr_tail=run.stderr[-3000:])
        except subprocess.TimeoutExpired:
            record = {'completed': False, 'timeout_s': 30}
        report['cases'][name] = record
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
