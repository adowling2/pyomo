# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Native G4 index-one DAE and implicit algebraic reconstruction experiments.

No Pyomo import, optimization solver or ASL extension is used. Each case is
isolated with a 30-second timeout and core dumps disabled. Process success
and mathematical accuracy are recorded separately. This is research evidence,
not a public Simulator interface or a promise that every native case passes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import casadi as ca
import numpy as np

TIMES = np.array([0, 0.2, 0.7, 1.2])


def equations(kind):
    """Construct uneliminated residual equations, including ordinary integral state."""
    x, p, time = (
        ca.SX.sym('x', 2),
        ca.SX.sym('p', 3 if kind == 'coupled' else 2),
        ca.SX.sym('t'),
    )
    z = ca.SX.sym('z', 2 if kind == 'coupled' else 1)
    if kind == 'nonlinear':
        ode, alg = ca.vertcat(p[0] * z, x[0]), z**2 - p[1] * x[0]
    elif kind == 'coupled':
        ode = ca.vertcat(-p[0] * z[0], z[1])
        alg = ca.vertcat(z[0] - p[1] * x[0], z[1] - z[0] ** 2 - p[2])
    else:
        ode, alg = ca.vertcat(-p[0] * z, x[0]), z - p[1] * x[0]
    return x, z, p, time, ode, alg


def consistent_solver(x, z, p, time, alg, tolerance):
    """Use the original residual to solve and locally validate a regular root."""
    packed = ca.vertcat(time, x, p)
    root = ca.rootfinder(
        'root',
        'newton',
        {'x': z, 'p': packed, 'g': alg},
        {
            'abstol': tolerance,
            'abstolStep': 0,
            'max_iter': 100,
            'regularity_check': True,
            'error_on_fail': True,
        },
    )
    guess, data = ca.MX.sym('guess', z.numel()), ca.MX.sym('data', packed.numel())
    solution = root(guess, data)
    check = ca.Function('check', [z, packed], [ca.det(ca.jacobian(alg, z)), alg])
    determinant, residual = check(solution, data)
    valid = ca.logic_and(
        ca.logic_and(
            ca.sum1(ca.fabs(solution)) < ca.inf,
            ca.logic_and(ca.fabs(determinant) > 0, ca.fabs(determinant) < ca.inf),
        ),
        ca.logic_and(
            ca.sum1(ca.fabs(residual)) < ca.inf, ca.mmax(ca.fabs(residual)) <= tolerance
        ),
    )
    solution = solution.attachAssert(
        valid, 'Algebraic root must be finite, locally regular and satisfy its residual'
    )
    return ca.Function('consistent', [guess, data], [solution])


def build_map(kind, project, tolerance=1e-11, times=TIMES, segmented=False):
    """Integrate an implicit DAE, optionally reconstructing every algebraic output."""
    x, z, p, time, ode, alg = equations(kind)
    options = {
        'reltol': tolerance,
        'abstol': tolerance / 100,
        'quad_err_con': True,
        'calc_ic': True,
        'calc_icB': True,
        'suppress_algebraic': False,
    }
    if ca.__version__ == '3.8.1':
        options['max_num_dir'] = 1
    if segmented:
        return segmented_map(kind, tolerance, times, options), options
    integrate = ca.integrator(
        'F',
        'idas',
        {'x': x, 'z': z, 'p': p, 't': time, 'ode': ode, 'alg': alg},
        0,
        np.asarray(times).tolist(),
        options,
    )
    w, guess = ca.MX.sym('w', 2 + p.numel()), ca.MX.sym('guess', z.numel())
    root = consistent_solver(x, z, p, time, alg, tolerance / 100) if project else None
    z0 = root(guess, ca.vertcat(0, w)) if project else guess
    result = integrate(x0=w[:2], z0=z0, p=w[2:])
    zout = result['zf']
    if project:
        zout = ca.horzcat(
            *(
                root(zout[:, i], ca.vertcat(t, result['xf'][:, i], w[2:]))
                for i, t in enumerate(times)
            )
        )
    flow = ca.Function(
        'flow', [w, guess], [ca.vertcat(result['xf'], zout)], ['w', 'guess'], ['yf']
    )
    return flow, options


def segmented_map(kind, tolerance, times, options):
    """Compose native unit-interval IDAS maps with implicit algebraic reconstruction."""
    x, z, p, time, ode, alg = equations(kind)
    root = consistent_solver(x, z, p, time, alg, tolerance / 100)
    start, duration = ca.SX.sym('start'), ca.SX.sym('duration')
    integrate = ca.integrator(
        'F',
        'idas',
        {
            'x': x,
            'z': z,
            'p': ca.vertcat(p, start, duration),
            't': time,
            'ode': duration * ca.substitute(ode, time, start + duration * time),
            'alg': ca.substitute(alg, time, start + duration * time),
        },
        0,
        1,
        options,
    )
    x0, z0, parameters = (
        ca.MX.sym('x0', x.numel()),
        ca.MX.sym('z0', z.numel()),
        ca.MX.sym('p', p.numel() + 2),
    )
    initial_z = root(z0, ca.vertcat(parameters[-2], x0, parameters[:-2]))
    out = integrate(x0=x0, z0=initial_z, p=parameters)
    final_z = root(
        out['zf'],
        ca.vertcat(parameters[-2] + parameters[-1], out['xf'], parameters[:-2]),
    )
    phase = ca.Function(
        'phase',
        [x0, z0, parameters],
        [out['xf'], final_z],
        ['x0', 'z0', 'p'],
        ['xf', 'zf'],
    )
    w, guess = ca.MX.sym('w', x.numel() + p.numel()), ca.MX.sym('guess', z.numel())
    grid = np.union1d([0], times)
    count = len(grid) - 1
    initial_z = root(guess, ca.vertcat(0, w))
    if count:
        inputs = ca.vertcat(
            ca.repmat(w[2:], 1, count), ca.DM(grid[:-1]).T, ca.DM(np.diff(grid)).T
        )
        accumulated = phase.mapaccum('phases', count, ['x0', 'z0'], ['xf', 'zf'])(
            x0=w[:2], z0=initial_z, p=inputs
        )
        values = ca.horzcat(
            ca.vertcat(w[:2], initial_z),
            ca.vertcat(accumulated['xf'], accumulated['zf']),
        )
    else:
        values = ca.vertcat(w[:2], initial_z)
    return ca.Function(
        'flow',
        [w, guess],
        [values[:, np.searchsorted(grid, times).tolist()]],
        ['w', 'guess'],
        ['yf'],
    )


def reference(kind, point, branch=1, times=TIMES):
    """Evaluate exact reduced solutions and hand-derived independent partials."""
    times = np.asarray(times)
    initial, q0, k, a = point[:4]
    parameters = len(point)
    jac = np.zeros((4 if kind == 'coupled' else 3, len(times), parameters))
    if kind == 'nonlinear':
        r = np.sqrt(a * initial)
        z = branch * r + a * k * times / 2
        x = z * z / a
        q = (
            q0
            + initial * times
            + branch * r * k * times**2 / 2
            + a * k * k * times**3 / 12
        )
        dz_x, dz_k, dz_a = (
            branch * a / (2 * r),
            a * times / 2,
            branch * initial / (2 * r) + k * times / 2,
        )
        jac[0, :, 0], jac[0, :, 2], jac[0, :, 3] = (
            2 * z / a * dz_x,
            2 * z / a * dz_k,
            2 * z / a * dz_a - z * z / a**2,
        )
        jac[1, :, 0], jac[1, :, 1] = times + branch * a * k * times**2 / (4 * r), 1
        jac[1, :, 2], jac[1, :, 3] = (
            branch * r * times**2 / 2 + a * k * times**3 / 6,
            branch * initial * k * times**2 / (4 * r) + k * k * times**3 / 12,
        )
        jac[2, :, 0], jac[2, :, 2], jac[2, :, 3] = dz_x, dz_k, dz_a
        return np.vstack([x, q, z]), jac
    rate = -k * a
    e = np.exp(rate * times)
    g = np.expm1(rate * times) / rate
    gp = ((rate * times - 1) * e + 1) / rate**2
    x = initial * e
    jac[0, :, 0], jac[0, :, 2], jac[0, :, 3] = e, -a * times * x, -k * times * x
    jac[2, :, 0], jac[2, :, 2], jac[2, :, 3] = (
        a * e,
        -a * a * times * x,
        x * (1 - k * a * times),
    )
    if kind == 'linear':
        q = q0 + initial * g
        jac[1, :, 0], jac[1, :, 1], jac[1, :, 2], jac[1, :, 3] = (
            g,
            1,
            -a * initial * gp,
            -k * initial * gp,
        )
        return np.vstack([x, q, a * x]), jac
    b = point[4]
    e2 = np.exp(2 * rate * times)
    g2 = np.expm1(2 * rate * times) / (2 * rate)
    gp2 = ((2 * rate * times - 1) * e2 + 1) / (2 * rate) ** 2
    q = q0 + a * a * initial * initial * g2 + b * times
    jac[1, :, 0], jac[1, :, 1] = 2 * a * a * initial * g2, 1
    jac[1, :, 2] = -2 * a**3 * initial**2 * gp2
    jac[1, :, 3] = 2 * a * initial**2 * g2 - 2 * k * a * a * initial**2 * gp2
    jac[1, :, 4] = times
    jac[3] = 2 * a * x[:, None] * jac[2]
    jac[3, :, 4] += 1
    return np.vstack([x, q, a * x, a * a * x * x + b]), jac


def scaled_error(actual, expected):
    """Measure finite, dimensionally matched error relative to a reference scale."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(
        expected, dtype=float
    )
    if actual.shape != expected.shape:
        raise AssertionError((actual.shape, expected.shape))
    if not np.isfinite(actual).all():
        raise AssertionError('Nonfinite numerical result')
    return float(np.max(abs(actual - expected) / np.maximum(1, abs(expected))))


def accuracy_case(kind, project, tolerance, segmented=False):
    """Test every output/mode, total IC composition and independent FD refinement."""
    point = np.array(
        [1.3, 0, 0.3 if kind == 'nonlinear' else 0.7, 1.4]
        + ([0.2] if kind == 'coupled' else [])
    )
    guess = [0.3, 0.5] if kind == 'coupled' else [0.3]
    flow, options = build_map(kind, project, tolerance, segmented=segmented)
    expected, jac = reference(kind, point)
    value = flow(w=point, guess=guess)['yf']
    n = point.size
    full = flow.forward(n)(
        w=point,
        guess=guess,
        out_yf=value,
        fwd_w=np.eye(n),
        fwd_guess=ca.DM.zeros(len(guess), n),
    )['fwd_yf']
    one = flow.forward(1)(
        w=point,
        guess=guess,
        out_yf=value,
        fwd_w=np.arange(1, n + 1) / 10,
        fwd_guess=np.zeros(len(guess)),
    )['fwd_yf']
    errors = {
        'primal': scaled_error(value, expected),
        'forward_multiple': scaled_error(
            full, np.hstack([jac[:, :, i] for i in range(n)])
        ),
        'forward_one': scaled_error(
            one, np.einsum('itp,p->it', jac, np.arange(1, n + 1) / 10)
        ),
    }
    failures = {}
    weights = []
    for index in (0, 1, 2):
        seed = np.zeros_like(expected)
        seed[index, -1] = 1
        weights.append(seed)
    rng = np.random.default_rng(604)
    weights.append(rng.normal(size=expected.shape) / 3)
    for name, seed in zip(
        ('terminal_x', 'integral', 'terminal_z', 'mixed_observations'), weights
    ):
        try:
            adj = flow.reverse(1)(w=point, guess=guess, out_yf=value, adj_yf=seed)
            errors['adjoint_' + name] = scaled_error(
                adj['adj_w'], np.einsum('it,itp->p', seed, jac).reshape(n, 1)
            )
            errors['guess_adjoint_' + name] = scaled_error(
                adj['adj_guess'], np.zeros((len(guess), 1))
            )
        except RuntimeError as exception:
            failures['adjoint_' + name] = str(exception)
    try:
        batch = flow.reverse(2)(
            w=point,
            guess=guess,
            out_yf=value,
            adj_yf=np.hstack([weights[2], weights[3]]),
        )
        errors['adjoint_multiple'] = scaled_error(
            batch['adj_w'],
            np.column_stack([np.einsum('it,itp->p', s, jac) for s in weights[2:]]),
        )
    except RuntimeError as exception:
        batch = None
        failures['adjoint_multiple'] = str(exception)
    directions = np.column_stack([np.arange(1, n + 1) / 10, np.arange(n, 0, -1) / 10])
    products = flow.forward(2)(
        w=point,
        guess=guess,
        out_yf=value,
        fwd_w=directions,
        fwd_guess=ca.DM.zeros(len(guess), 2),
    )['fwd_yf']
    if batch is not None:
        errors['duality'] = max(
            abs(
                float(np.sum(s * np.asarray(products)[:, 4 * j : 4 * (j + 1)]))
                - float(directions[:, j] @ np.asarray(batch['adj_w'])[:, i])
            )
            for i, s in enumerate(weights[2:])
            for j in range(2)
        )
    differences = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5):
        fd = np.stack(
            [
                np.asarray(
                    (
                        flow(w=point + step * d, guess=guess)['yf']
                        - flow(w=point - step * d, guess=guess)['yf']
                    )
                    / (2 * step)
                )
                for d in np.eye(n)
            ],
            axis=2,
        )
        differences.append({'step': step, 'scaled_error': scaled_error(fd, jac)})
    x, z, p, time, ode, alg = equations(kind)
    residual = ca.Function('residual', [x, z, p], [alg])
    errors['algebraic_residual'] = max(
        float(np.max(abs(np.asarray(residual(value[:2, i], value[2:, i], point[2:])))))
        for i in range(4)
    )
    theta = ca.MX.sym('theta', n - 1)
    # Shared theta[0] sets the differential IC and physical rate.
    total_w = ca.vertcat(theta[0] ** 2, theta[1] ** 2, theta[0], theta[2:])
    trajectory = flow(w=total_w, guess=guess)['yf']
    terminal_weights = np.arange(2, 2 + expected.shape[0])
    cost = (
        ca.dot(ca.DM(terminal_weights), trajectory[:, -1])
        + theta[0] ** 2
        + 0.2 * theta[2] ** 2
    )
    total = ca.Function('total', [theta], [cost])
    total_point = np.array([point[2], 0.2, *point[3:]])
    total_input = np.array(
        [total_point[0] ** 2, total_point[1] ** 2, total_point[0], *total_point[2:]]
    )
    _, total_jac = reference(kind, total_input)
    partial = terminal_weights @ total_jac[:, -1, :]
    total_expected = np.array(
        [
            2 * total_point[0] * partial[0] + partial[2] + 2 * total_point[0],
            2 * total_point[1] * partial[1],
            partial[3] + 0.4 * total_point[2],
            *partial[4:],
        ]
    )
    total_value = total(total_point)
    errors['total_IC_direct'] = scaled_error(
        total.forward(n - 1)(total_point, total_value, np.eye(n - 1)),
        total_expected.reshape(1, n - 1),
    )
    try:
        errors['total_IC_adjoint'] = scaled_error(
            total.reverse(1)(total_point, total_value, 1),
            total_expected.reshape(n - 1, 1),
        )
    except RuntimeError as exception:
        failures['total_IC_adjoint'] = str(exception)
    total_differences = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5):
        fd = np.array(
            [
                float(
                    (total(total_point + step * d) - total(total_point - step * d))
                    / (2 * step)
                )
                for d in np.eye(n - 1)
            ]
        )
        total_differences.append(
            {'step': step, 'scaled_error': scaled_error(fd, total_expected)}
        )
    return {
        'kind': kind,
        'projected': project,
        'segmented': segmented,
        'options': options,
        'errors': errors,
        'accurate': {k: v < 3e-7 for k, v in errors.items()},
        'finite_differences': differences,
        'minimum_fd_error': min(d['scaled_error'] for d in differences),
        'total_finite_differences': total_differences,
        'minimum_total_fd_error': min(d['scaled_error'] for d in total_differences),
        'failures': failures,
        'all_modes_accurate': not failures and all(v < 3e-7 for v in errors.values()),
    }


def initialization_case(kind, guess, initial=1.3, gain=1.4):
    """Check branch/guess semantics, local regularity and failed real roots."""
    point = np.array(
        [initial, 0, 0.3 if kind == 'nonlinear' else 0.7, gain]
        + ([0.2] if kind == 'coupled' else [])
    )
    flow, options = build_map(kind, True, segmented=True)
    row = {'kind': kind, 'point': point.tolist(), 'guess': guess, 'options': options}
    if kind == 'nonlinear' and initial * gain > 0:
        row['analytic_initial_gz_norm'] = float(2 * np.sqrt(initial * gain))
        row['analytic_initial_inverse_gz_norm'] = float(
            1 / (2 * np.sqrt(initial * gain))
        )
    try:
        value = flow(w=point, guess=guess)['yf']
        branch = -1 if kind == 'nonlinear' and guess[0] < 0 else 1
        expected, jac = reference(kind, point, branch)
        adj = flow.reverse(1)(
            w=point, guess=guess, out_yf=value, adj_yf=np.ones(expected.shape)
        )
        row.update(
            primal_error=scaled_error(value, expected),
            adjoint_error=scaled_error(
                adj['adj_w'], np.sum(jac, axis=(0, 1)).reshape(point.size, 1)
            ),
            guess_adjoint_error=scaled_error(
                adj['adj_guess'], np.zeros((len(guess), 1))
            ),
            branch=branch,
            initial_x=float(value[0, 0]),
            initial_z=np.asarray(value[2:, 0]).ravel().tolist(),
        )
        x, z, p, time, ode, alg = equations(kind)
        jacobian = ca.Function('g_z', [x, z, p], [ca.jacobian(alg, z)])
        row['initial_gz_condition'] = float(
            np.linalg.cond(np.asarray(jacobian(point[:2], value[2:, 0], point[2:])))
        )
        row['initial_gz_norm'] = float(
            np.linalg.norm(
                np.asarray(jacobian(point[:2], value[2:, 0], point[2:])), ord=2
            )
        )
        row['completed'] = True
    except RuntimeError as exception:
        row.update(completed=False, error=str(exception))
    return row


def invalid_residual_case(kind):
    """Reject structurally or locally singular equations, including initial-only maps."""
    x, z, p, time = ca.SX.sym('x'), ca.SX.sym('z'), ca.SX.sym('p'), ca.SX.sym('t')
    alg = x if kind == 'structural' else z * z
    row = {'kind': kind}
    try:
        root = consistent_solver(x, z, p, time, alg, 1e-13)
        value = root(0, [0, 0, 0.7])
        row.update(rejected=False, value=np.asarray(value).ravel().tolist())
    except RuntimeError as exception:
        row.update(rejected=True, error=str(exception))
    return row


def initial_only_case(kind):
    """Check algebraic IC derivatives without making any positive-duration call."""
    point = np.array([1.3, 0, 0.3 if kind == 'nonlinear' else 0.7, 1.4])
    flow, options = build_map(kind, True, times=[0], segmented=True)
    expected, jac = reference(kind, point, times=[0])
    value = flow(w=point, guess=[0.3])['yf']
    forward = flow.forward(4)(
        w=point, guess=[0.3], out_yf=value, fwd_w=np.eye(4), fwd_guess=ca.DM.zeros(1, 4)
    )['fwd_yf']
    adjoint = flow.reverse(1)(
        w=point, guess=[0.3], out_yf=value, adj_yf=np.ones(expected.shape)
    )
    return {
        'kind': kind,
        'primal_error': scaled_error(value, expected),
        'forward_error': scaled_error(forward, jac[:, 0, :]),
        'adjoint_error': scaled_error(
            adjoint['adj_w'], np.sum(jac, axis=(0, 1)).reshape(4, 1)
        ),
        'guess_adjoint_error': scaled_error(adjoint['adj_guess'], np.zeros((1, 1))),
    }


def nonfinite_case(where):
    """Verify failure instead of returning a native nonfinite Newton result."""
    flow, _ = build_map('nonlinear', True, segmented=True)
    point, guess = [1.3, 0, 0.3, 1.4], [0.3]
    if where == 'guess':
        guess = [float('nan')]
    else:
        point[3] = float('inf')
    try:
        value = flow(w=point, guess=guess)['yf']
        return {
            'input_kind': where,
            'rejected': False,
            'finite_result': bool(np.isfinite(np.asarray(value)).all()),
        }
    except RuntimeError as exception:
        return {'input_kind': where, 'rejected': True, 'error': str(exception)}


def initialization_options_case(calc_ic):
    """Record IDAS initialization behavior even when an exact algebraic root is given."""
    tolerance = 1e-11
    options = {
        'reltol': tolerance,
        'abstol': tolerance / 100,
        'calc_ic': calc_ic,
        'calc_icB': True,
        'quad_err_con': True,
        'suppress_algebraic': False,
    }
    if ca.__version__ == '3.8.1':
        options['max_num_dir'] = 1
    flow = segmented_map('linear', tolerance, TIMES, options)
    point = [1.3, 0, 0.7, 1.4]
    try:
        value = flow(w=point, guess=[0.3])['yf']
        expected, _ = reference('linear', point)
        return {
            'options': options,
            'completed': True,
            'primal_error': scaled_error(value, expected),
        }
    except RuntimeError as exception:
        return {'options': options, 'completed': False, 'error': str(exception)}


def main():
    """Run each native experiment in isolation and retain qualified outcomes."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    cases = {}
    for kind in ('linear', 'nonlinear', 'coupled'):
        for tolerance in (1e-7, 1e-10, 1e-11, 1e-12):
            cases[f'{kind}-segmented-{tolerance}'] = (
                lambda k=kind, t=tolerance: accuracy_case(k, True, t, segmented=True)
            )
        for project in (False, True):
            for tolerance in (1e-7, 1e-10, 1e-11, 1e-12):
                cases[f'{kind}-{project}-{tolerance}'] = (
                    lambda k=kind, p=project, t=tolerance: accuracy_case(k, p, t)
                )
    for kind, guesses in [
        ('linear', [[0.3], [3]]),
        ('nonlinear', [[0.2], [3], [-0.2], [-3]]),
        ('coupled', [[0.3, 0.5], [3, 5]]),
    ]:
        for index, guess in enumerate(guesses):
            cases[f'initial-{kind}-{index}'] = (
                lambda k=kind, g=guess: initialization_case(k, g)
            )
    cases.update(
        {
            'invalid-guess': lambda: initialization_case('nonlinear', [0]),
            'singular-initial': lambda: initialization_case(
                'nonlinear', [0], initial=0
            ),
            'invalid-gain': lambda: initialization_case('nonlinear', [1], gain=-1),
            'invalid-differential-IC': lambda: initialization_case(
                'nonlinear', [1], initial=-1
            ),
            'singular-gain': lambda: initialization_case('nonlinear', [0], gain=0),
            'near-singular': lambda: initialization_case(
                'nonlinear', [1e-6], initial=1e-12
            ),
            'structural-singularity': lambda: invalid_residual_case('structural'),
            'local-singularity': lambda: invalid_residual_case('local'),
            'initial-only-linear': lambda: initial_only_case('linear'),
            'initial-only-nonlinear': lambda: initial_only_case('nonlinear'),
            'nonfinite-guess': lambda: nonfinite_case('guess'),
            'nonfinite-parameter': lambda: nonfinite_case('parameter'),
            'calc_ic-disabled': lambda: initialization_options_case(False),
            'calc_ic-enabled': lambda: initialization_options_case(True),
        }
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=cases)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        try:
            record = {'executed': True, 'observations': cases[args.case]()}
        except Exception as exception:
            record = {'executed': False, 'error': str(exception)}
        print(json.dumps(record, allow_nan=False))
        return
    report = {
        'checkpoint': 'G4',
        'casadi': ca.__version__,
        'numpy': np.__version__,
        'python': sys.version,
        'platform': platform.platform(),
        'cases': {},
        'matrix_policy': 'Scalar tolerances, required quadrature error control, calc_ic=True, calc_icB=True, algebraic error control retained; serial derivative batching on 3.8.1 only.',
        'root_policy': 'Native Newton on original residuals; max_iter=100; residual tolerance equals scalar abstol; abstolStep=0; error_on_fail and regularity_check enabled. Assert finite root/residual/determinant, nonzero local determinant and residual within target.',
    }
    for name in cases:
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--case', name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            row = json.loads(run.stdout) if run.returncode == 0 else {'executed': False}
            row.update(
                process_exit_code=run.returncode,
                stderr_first_line=run.stderr.splitlines()[0] if run.stderr else '',
            )
        except subprocess.TimeoutExpired:
            row = {'executed': False, 'timeout_s': 30}
        report['cases'][name] = row
    source = Path(__file__).read_bytes()
    report['source_sha256'] = hashlib.sha256(source).hexdigest()
    report['command'] = (
        ' <env-python> examples/dae/sensitivity/dae_qualification.py --output <report.json>'
    )
    report['references'] = [
        'https://web.casadi.org/docs/#nonlinear-root-finding-problems',
        'https://web.casadi.org/api/html/d3/d65/group__rootfinder.html',
    ]
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
