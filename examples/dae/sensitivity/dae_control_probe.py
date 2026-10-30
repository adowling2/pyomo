# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Qualify native fixed-grid DAE controls before extending Simulator.compile.

Pure CasADi/NumPy, with no Pyomo, ASL or optimizer import. Branch-safe
consistency comes from the original residual. Native experiments are run
in separate processes with a 30-second timeout and core dumps disabled.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import casadi as ca
import numpy as np
from dae_qualification import consistent_solver

BOUNDS = np.array([0, 0.2, 0.7, 1.2])


def control_reference(kind, times, bounds, point, branch=1):
    """Propagate exact phase formulas and independent x0/q0/k/a/amplitude partials."""
    times, bounds, point = np.asarray(times), np.asarray(bounds), np.asarray(point)
    phases = len(bounds) - 1
    controls = point[4:].reshape(-1, phases)
    n = len(point)
    basis = np.eye(n)
    k, a = point[2:4]
    values = []
    jac = []
    for time in times:
        x, q = point[:2]
        dx, dq = basis[0].copy(), basis[1].copy()
        for j, (start, end) in enumerate(zip(bounds, bounds[1:])):
            if time <= start:
                break
            d = min(time, end) - start
            u = controls[0, j]
            du = basis[4 + j]
            if kind == 'nonlinear':
                z = branch * math.sqrt(a * x + u)
                dz = (a * dx + x * basis[3] + du) / (2 * z)
                q, dq = (
                    q + x * d + k * z * d * d / 2 + a * k * k * d**3 / 12,
                    dq
                    + d * dx
                    + (z * d * d / 2 + a * k * d**3 / 6) * basis[2]
                    + k * d * d / 2 * dz
                    + k * k * d**3 / 12 * basis[3],
                )
                x, dx = (
                    x + k * z * d + a * k * k * d * d / 4,
                    dx
                    + (z * d + a * k * d * d / 2) * basis[2]
                    + k * d * dz
                    + k * k * d * d / 4 * basis[3],
                )
            else:
                v = controls[1, j]
                dv = basis[4 + phases + j]
                rate = k * a
                dr = a * basis[2] + k * basis[3]
                forcing = k * u + v
                df = u * basis[2] + k * du + dv
                e = math.exp(rate * d)
                if rate:
                    g = math.expm1(rate * d) / rate
                    gp = ((rate * d - 1) * e + 1) / rate**2
                    h = (g - d) / rate
                    hp = (gp - h) / rate
                else:
                    g, gp, h, hp = d, d * d / 2, d * d / 2, d**3 / 6
                q, dq = (
                    q + g * x + forcing * h,
                    dq + g * dx + (gp * x + forcing * hp) * dr + h * df,
                )
                x, dx = (
                    e * x + forcing * g,
                    e * dx + (d * e * x + forcing * gp) * dr + g * df,
                )
        right = min(np.searchsorted(bounds, time, side='right') - 1, phases - 1)
        u = controls[0, right]
        du = basis[4 + right]
        if kind == 'nonlinear':
            z = branch * math.sqrt(a * x + u)
            dz = (a * dx + x * basis[3] + du) / (2 * z)
        else:
            z = a * x + u
            dz = a * dx + x * basis[3] + du
        values.append([x, q, z])
        jac.append([dx, dq, dz])
    return np.asarray(values).T, np.asarray(jac).transpose(1, 0, 2)


def build_native(kind, times, bounds, tolerance=1e-10):
    """Compose normalized IDAS phases and reproject right-side observed algebraics."""
    times, bounds = np.asarray(times), np.asarray(bounds)
    nctrl = 1 if kind == 'nonlinear' else 2
    x, z, t = ca.SX.sym('x', 2), ca.SX.sym('z'), ca.SX.sym('t')
    p = ca.SX.sym('p', 2 + nctrl)
    ode = ca.vertcat(p[0] * z + (p[3] if nctrl == 2 else 0), x[0])
    alg = (z * z if kind == 'nonlinear' else z) - p[1] * x[0] - p[2]
    root = consistent_solver(x, z, p, t, alg, tolerance / 100)
    start, duration = ca.SX.sym('start'), ca.SX.sym('duration')
    opts = {
        'reltol': tolerance,
        'abstol': tolerance / 100,
        'quad_err_con': True,
        'calc_ic': True,
        'calc_icB': True,
        'suppress_algebraic': False,
    }
    if ca.__version__ == '3.8.1':
        opts['max_num_dir'] = 1
    integrate = ca.integrator(
        'F',
        'idas',
        {
            'x': x,
            'z': z,
            't': t,
            'p': ca.vertcat(p, start, duration),
            'ode': duration * ode,
            'alg': alg,
        },
        0,
        1,
        opts,
    )
    x0, z0, data = ca.MX.sym('x0', 2), ca.MX.sym('z0'), ca.MX.sym('data', p.numel() + 2)
    initial_z = root(z0, ca.vertcat(data[-2], x0, data[:-2]))
    out = integrate(x0=x0, z0=initial_z, p=data)
    final_z = root(out['zf'], ca.vertcat(data[-2] + data[-1], out['xf'], data[:-2]))
    phase = ca.Function(
        'phase', [x0, z0, data], [out['xf'], final_z], ['x0', 'z0', 'p'], ['xf', 'zf']
    )
    w, guess = ca.MX.sym('w', 4 + nctrl * (len(bounds) - 1)), ca.MX.sym('guess')
    u = ca.reshape(w[4:], len(bounds) - 1, nctrl).T
    grid = np.union1d(times, bounds[bounds <= times[-1]])
    count = len(grid) - 1
    initial = root(guess, ca.vertcat(bounds[0], w[:4], u[:, 0]))
    xf, zf = w[:2], initial
    if count:
        phases = np.searchsorted(bounds, grid[:-1], side='right') - 1
        inputs = ca.vertcat(
            ca.repmat(w[2:4], 1, count),
            u[:, phases.tolist()],
            ca.DM(grid[:-1]).T,
            ca.DM(np.diff(grid)).T,
        )
        out = phase.mapaccum('intervals', count, ['x0', 'z0'], ['xf', 'zf'])(
            x0=w[:2], z0=initial, p=inputs
        )
        xf, zf = ca.horzcat(w[:2], out['xf']), ca.horzcat(initial, out['zf'])
    columns = np.searchsorted(grid, times).tolist()
    xf, zf = xf[:, columns], zf[:, columns]
    right = np.minimum(
        np.searchsorted(bounds, times, side='right') - 1, len(bounds) - 2
    )
    zf = root.map(len(times))(
        zf,
        ca.vertcat(
            ca.DM(times).T, xf, ca.repmat(w[2:4], 1, len(times)), u[:, right.tolist()]
        ),
    )
    return (
        ca.Function('flow', [w, guess], [ca.vertcat(xf, zf)], ['w', 'guess'], ['yf']),
        opts,
    )


def scaled_error(actual, expected):
    """Compare finite, explicitly shaped values with a scaled exact reference."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(
        expected, dtype=float
    )
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError((actual.shape, expected.shape))
    return float(np.max(abs(actual - expected) / np.maximum(1, abs(expected))))


def experiment(kind, times, branch=1, tolerance=1e-10):
    """Check amplitude Jacobians, several weighted adjoints and FD phase composition."""
    point = np.array(
        [1.3, 0, 0.3 if kind == 'nonlinear' else -0.7, 1.4, 0.1, 0.4, -0.2]
        + ([] if kind == 'nonlinear' else [0.2, -0.1, 0.3])
    )
    flow, options = build_native(kind, times, BOUNDS, tolerance)
    guess = 0.3 * branch
    expected, jac = control_reference(kind, times, BOUNDS, point, branch)
    value = flow(w=point, guess=guess)['yf']
    n = len(point)
    nt = len(times)
    direct = flow.forward(n)(
        w=point, guess=guess, out_yf=value, fwd_w=np.eye(n), fwd_guess=ca.DM.zeros(1, n)
    )['fwd_yf']
    rng = np.random.default_rng(606)
    weights = np.hstack(
        [rng.normal(size=expected.shape) / 3, rng.normal(size=expected.shape) / 3]
    )
    adjoint = flow.reverse(2)(w=point, guess=guess, out_yf=value, adj_yf=weights)
    errors = {
        'primal': scaled_error(value, expected),
        'direct': scaled_error(direct, np.hstack([jac[:, :, i] for i in range(n)])),
        'adjoint': scaled_error(
            adjoint['adj_w'],
            np.column_stack(
                [
                    np.einsum('it,itp->p', weights[:, nt * i : nt * (i + 1)], jac)
                    for i in range(2)
                ]
            ),
        ),
        'guess_adjoint': scaled_error(adjoint['adj_guess'], np.zeros((1, 2))),
    }
    errors['duality'] = max(
        abs(
            float(
                np.sum(
                    weights[:, nt * i : nt * (i + 1)]
                    * np.asarray(direct)[:, nt * j : nt * (j + 1)]
                )
            )
            - float(adjoint['adj_w'][j, i])
        )
        for i in range(2)
        for j in range(n)
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
    return {
        'kind': kind,
        'branch': branch,
        'times': np.asarray(times).tolist(),
        'control_times': BOUNDS.tolist(),
        'options': options,
        'errors': errors,
        'accurate': all(v < 3e-7 for v in errors.values()),
        'finite_differences': differences,
        'minimum_fd_error': min(v['scaled_error'] for v in differences),
    }


def main():
    """Run bounded native cases and retain exact endpoint and branch qualifications."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    cases = {}
    grids = {
        'switches': [0, 0.2, 0.5, 0.7, 1.2],
        'initial': [0],
        'at_switch': [0.2],
        'sparse': [0.2, 0.7],
        'terminal': [1.2],
        'adjacent': [np.nextafter(0.2, 0), 0.2, np.nextafter(0.2, 1), 1.2],
        'dense': np.linspace(0, 1.2, 21),
    }
    for kind in ('affine', 'nonlinear'):
        for name, times in grids.items():
            cases[f'{kind}-{name}'] = lambda k=kind, t=times: experiment(k, t)
    cases['nonlinear-negative'] = lambda: experiment(
        'nonlinear', [0, 0.2, 0.5, 0.7, 1.2], -1
    )
    for tolerance in (1e-7, 1e-11):
        cases[f'nonlinear-tolerance-{tolerance}'] = lambda t=tolerance: experiment(
            'nonlinear', [0, 0.2, 0.7, 1.2], tolerance=t
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=cases)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        try:
            row = {'executed': True, 'observations': cases[args.case]()}
        except Exception as exception:
            row = {'executed': False, 'error': str(exception)}
        print(json.dumps(row, allow_nan=False))
        return
    report = {
        'checkpoint': 'G6 native control qualification',
        'casadi': ca.__version__,
        'numpy': np.__version__,
        'python': sys.version,
        'platform': platform.platform(),
        'cases': {},
        'convention': 'Continuous differential states; right-side algebraics at internal switches; last interval at terminal time.',
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
    report['source_files'] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('dae_control_probe.py', 'dae_qualification.py')
    }
    report['command'] = (
        ' <env-python> examples/dae/sensitivity/dae_control_probe.py --output <report.json>'
    )
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
