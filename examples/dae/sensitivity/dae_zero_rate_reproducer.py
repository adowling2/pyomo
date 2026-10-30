# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Bounded native IDAS zero-feedback adjoint reduction; no Pyomo import.

A successful process is not necessarily an accurate derivative. Preserve
both numerical errors and runtime failures, with mandatory quadrature control.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys

import casadi as ca
import numpy as np


def experiment(case):
    """Differentiate one normalized interval of xdot=k*z+v, qdot=x, z=a*x+u."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    options = dict(
        reltol=1e-10,
        abstol=1e-12,
        quad_err_con=True,
        calc_ic=True,
        calc_icB=True,
        suppress_algebraic=False,
    )
    if case != 'default':
        options.update(max_step_size=0.001, max_num_steps=50000)
    if case == 'refined':
        options.update(reltol=1e-11, abstol=1e-13)
    if ca.__version__ == '3.8.1':
        options['max_num_dir'] = 1
    row = {'options': options, 'accurate': False}
    try:
        x, z, p = ca.SX.sym('x', 2), ca.SX.sym('z'), ca.SX.sym('p', 4)
        d, initial, a, u, v = 0.1, 1.3, 1.4, 0.1, 0.2
        native = ca.integrator(
            'F',
            'idas',
            {
                'x': x,
                'z': z,
                'p': p,
                'ode': d * ca.vertcat(p[0] * z + p[3], x[0]),
                'alg': z - p[1] * x[0] - p[2],
            },
            0,
            1,
            options,
        )
        xx, pp, zz = ca.MX.sym('x0', 2), ca.MX.sym('p', 4), ca.MX.sym('z0')
        flow = ca.Function(
            'flow',
            [xx, pp, zz],
            [native(x0=xx, p=pp, z0=zz)['xf']],
            ['x0', 'p', 'z0'],
            ['xf'],
        )
        args = {'x0': [initial, 0], 'p': [0, a, u, v], 'z0': a * initial + u}
        weights = np.array([[0.2, -0.3], [0.4, 0.1]])
        result = flow(**args)
        reverse = flow.reverse(2)(**args, out_xf=result['xf'], adj_xf=weights)
        zstart = a * initial + u
        jac = np.array(
            [
                [zstart * d + a * v * d * d / 2, 0, 0, d],
                [zstart * d * d / 2 + a * v * d**3 / 6, 0, 0, d * d / 2],
            ]
        )
        expected = jac.T @ weights
        actual = np.asarray(reverse['adj_p'])
        row['parameter_adjoint'] = actual.tolist()
        row['analytic_adjoint'] = expected.tolist()
        row['scaled_error'] = float(
            np.max(np.abs(actual - expected) / (1 + np.abs(expected)))
        )
        row['accurate'] = row['scaled_error'] < 3e-7
    except RuntimeError as exc:
        row['error'] = str(exc)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('default', 'step-limited', 'refined'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(experiment(args.case), allow_nan=False))
        return
    report = {
        'purpose': 'G6 zero-feedback native IDAS adjoint reduction',
        'python': sys.version,
        'casadi': ca.__version__,
        'numpy': np.__version__,
        'platform': platform.platform(),
        'cases': {},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for case in ('default', 'step-limited', 'refined'):
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--case', case],
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
        report['cases'][case] = row
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
