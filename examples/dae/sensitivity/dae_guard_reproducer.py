# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Measure dense symbolic determinant construction versus native symbolic QR.

This bounded diagnostic explains the DAE local-regularity implementation;
its timings are not unit-test thresholds or portable performance claims.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import casadi as ca


def case(size, strategy):
    """Construct one determinant and verify regular and singular matrix evaluations."""
    x, p = ca.SX.sym('x', size), ca.SX.sym('p', size, size)
    matrix = p + ca.diag(x)
    start = perf_counter()
    determinant = ca.det(matrix, 'symbolicqr') if strategy == 'qr' else ca.det(matrix)
    function = ca.Function('determinant', [x, p], [determinant])
    return {
        'size': size,
        'strategy': strategy,
        'construction_seconds': perf_counter() - start,
        'nodes': function.n_nodes(),
        'identity_value': float(function(ca.DM.zeros(size), ca.DM.eye(size))),
        'singular_value': float(function(ca.DM.zeros(size), ca.DM.zeros(size, size))),
    }


def main():
    """Contain expensive expansion in child processes with a five-second limit."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='store_true')
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--strategy', choices=('default', 'qr'), default='default')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(case(args.size, args.strategy), allow_nan=False))
        return
    report = {
        'casadi': ca.__version__,
        'python': sys.version,
        'platform': platform.platform(),
        'cases': [],
    }
    for strategy, sizes in (('default', (4, 6, 8, 10)), ('qr', (4, 6, 8, 10, 20))):
        for size in sizes:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                '--case',
                '--size',
                str(size),
                '--strategy',
                strategy,
            ]
            try:
                run = subprocess.run(command, capture_output=True, text=True, timeout=5)
                row = (
                    json.loads(run.stdout)
                    if run.returncode == 0
                    else {
                        'size': size,
                        'strategy': strategy,
                        'error': run.stderr[-1000:],
                    }
                )
                row['process_exit_code'] = run.returncode
            except subprocess.TimeoutExpired:
                row = {'size': size, 'strategy': strategy, 'timeout_s': 5}
            report['cases'].append(row)
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['command'] = (
        ' <env-python> examples/dae/sensitivity/dae_guard_reproducer.py --output <report.json>'
    )
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
