# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Reduce native IDAS explicit-time adjoint failures and test an internal clock.

The original DAE is xdot=k*t, qdot=x, z=a*x+t. The equivalent autonomous
form adds cdot=1 and uses c in place of t. All cases are pure CasADi/NumPy,
isolated with a 30-second limit and core dumps disabled. Disabled quadrature
error control is diagnostic only and is not permitted by Simulator.compile.
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


def experiment(clocked, normalized, tolerance, quadrature=True):
    """Check the same physical interval and terminal cost in equivalent formulations."""
    start, end, initial, k, a = 2.7, 3.2, [0.8065, 0.84585], -0.3, 1.4
    duration = end - start
    x, z, time = ca.SX.sym('x', 3 if clocked else 2), ca.SX.sym('z'), ca.SX.sym('t')
    p = ca.SX.sym('p', 4 if normalized else 2)
    clock = x[2] if clocked else (p[2] + p[3] * time if normalized else time)
    ode = (
        ca.vertcat(p[0] * clock, x[0], 1) if clocked else ca.vertcat(p[0] * clock, x[0])
    )
    if normalized:
        ode = p[3] * ode
    options = {
        'reltol': tolerance,
        'abstol': tolerance / 100,
        'quad_err_con': quadrature,
        'calc_ic': True,
        'calc_icB': True,
        'suppress_algebraic': False,
    }
    if ca.__version__ == '3.8.1':
        options['max_num_dir'] = 1
    row = {
        'clocked': clocked,
        'normalized': normalized,
        'options': options,
        'physical_bounds': [start, end],
        'initial_xq': initial,
        'parameters': [k, a],
    }
    try:
        integrate = ca.integrator(
            'F',
            'idas',
            {
                'x': x,
                'z': z,
                't': time,
                'p': p,
                'ode': ode,
                'alg': z - p[1] * x[0] - clock,
            },
            0 if normalized else start,
            1 if normalized else end,
            options,
        )
        point = {
            'x0': initial + ([start] if clocked else []),
            'z0': a * initial[0] + start,
            'p': [k, a] + ([start, duration] if normalized else []),
        }
        dx = (end**2 - start**2) / 2
        dq = (end**3 - start**3) / 6 - start**2 * duration / 2
        expected = np.array(
            [initial[0] + k * dx, initial[1] + initial[0] * duration + k * dq]
        )
        weight = [2.4, 1] + ([0] if clocked else [])
        value = integrate(**point)
        row['primal_error'] = float(
            np.max(abs(np.asarray(value['xf'])[:2, 0] - expected))
        )
        direct = integrate.factory('direct', ['x0', 'z0', 'p', 'fwd:p'], ['fwd:xf'])(
            **point, fwd_p=[1, 0] + ([0, 0] if normalized else [])
        )['fwd_xf']
        row['direct_error'] = float(np.max(abs(np.asarray(direct)[:2, 0] - [dx, dq])))
        gradient = integrate.factory(
            'gradient', ['x0', 'z0', 'p', 'adj:xf'], ['adj:p']
        )(**point, adj_xf=weight)['adj_p']
        row.update(
            analytic_rate_gradient=2.4 * dx + dq,
            rate_gradient=float(gradient[0]),
            adjoint_error=abs(float(gradient[0]) - (2.4 * dx + dq)),
        )
        row['accurate'] = (
            max(row['primal_error'], row['direct_error'], row['adjoint_error']) < 3e-7
        )
    except RuntimeError as exception:
        row.update(accurate=False, error=str(exception))
    return row


def main():
    """Run the diagnostic matrix without mistaking successful processes for accuracy."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    cases = {}
    for clocked in (False, True):
        for normalized in (False, True):
            for tolerance in (1e-7, 1e-10, 1e-12):
                cases[f'{clocked}-{normalized}-{tolerance}'] = (
                    lambda c=clocked, n=normalized, t=tolerance: experiment(c, n, t)
                )
    for normalized in (False, True):
        cases[f'unsafe-quad-disabled-{normalized}'] = lambda n=normalized: experiment(
            False, n, 1e-10, False
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=cases)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(cases[args.case](), allow_nan=False))
        return
    report = {
        'purpose': 'G5 native explicit-time dependency reduction and internal clock qualification',
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
            row = json.loads(run.stdout) if run.returncode == 0 else {'accurate': False}
            row.update(
                process_exit_code=run.returncode,
                stderr_first_line=run.stderr.splitlines()[0] if run.stderr else '',
            )
        except subprocess.TimeoutExpired:
            row = {'accurate': False, 'timeout_s': 30}
        report['cases'][name] = row
    report['clock_cases_accurate'] = all(
        report['cases'][n].get('accurate') for n in cases if n.startswith('True-')
    )
    report['clock_tight_cases_accurate'] = all(
        row.get('accurate')
        for name, row in report['cases'].items()
        if name.startswith('True-') and row.get('options', {}).get('reltol', 1) <= 1e-10
    )
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['command'] = (
        ' <env-python> examples/dae/sensitivity/dae_time_reproducer.py --output <report.json>'
    )
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
