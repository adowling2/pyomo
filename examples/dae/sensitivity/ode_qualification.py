# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Record G1 ODE numerical qualification and bounded stiff-adjoint reductions.

Run with PYTHONPATH pointing at this checkout, once per CasADi environment.
Native failure cases run in separate processes with a 30-second timeout.
The report distinguishes an executed experiment from an accurate derivative.
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
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.dae.tests.test_simulator_ode_qualification import (
    check_scaled_controls,
    check_scaled_total_derivative,
)

CONFIGURATIONS = {
    'default_bdf': {},
    'adams': {'linear_multistep_method': 'adams'},
    'polynomial': {'interpolation_type': 'polynomial'},
    'checkpoint_one': {'steps_per_checkpoint': 1},
    'more_steps': {'max_num_steps': 50000},
    'small_step': {'max_step_size': 0.001},
}


def stiff_case(formulation, coupled, configuration):
    """Compare one identical fast decay problem through native or public graphs."""
    times = [0, 0.0001, 0.01, 0.1, 1]
    rate = 1000
    options = {'reltol': 1e-11, 'abstol': 1e-13, 'quad_err_con': True}
    options.update(CONFIGURATIONS[configuration])
    n = 2 if coupled else 1
    record = {
        'formulation': formulation,
        'coupled': coupled,
        'rate': rate,
        'times': times,
        'options': options,
        'analytic': -sum(t * math.exp(-rate * t) for t in times),
    }
    try:
        if formulation == 'native':
            x, p = ca.SX.sym('x', n), ca.SX.sym('p')
            ode = ca.vertcat(-x[0], -p * x[1]) if coupled else -p * x
            integrate = ca.integrator(
                'F', 'cvodes', {'x': x, 'p': p, 'ode': ode}, 0, times, options
            )
            x0, parameter = ca.MX.sym('x0', n), ca.MX.sym('p')
            flow = ca.Function(
                'flow',
                [x0, parameter],
                [integrate(x0=x0, p=parameter)['xf']],
                ['x0', 'p'],
                ['xf'],
            )
        else:
            model = pyo.ConcreteModel()
            model.t = ContinuousSet(bounds=(0, 1))
            model.k = pyo.Param(mutable=True, initialize=rate)
            model.x = pyo.Var(model.t, range(n))
            model.dx = DerivativeVar(model.x, wrt=model.t)
            model.ode = pyo.Constraint(
                model.t,
                range(n),
                rule=lambda m, t, i: m.dx[t, i]
                == (-m.x[t, i] if coupled and i == 0 else -m.k * m.x[t, i]),
            )
            flow = Simulator(model, package='casadi').compile(
                parameters=[model.k], times=times, integrator_options=options
            )
        weight = np.zeros((n, len(times)))
        weight[-1] = 1
        primal = flow(x0=[1] * n, p=rate)['xf']
        expected = np.exp(-rate * np.asarray(times))
        record['primal_error'] = float(np.max(abs(np.asarray(primal)[-1] - expected)))
        direct = flow.factory('direct', ['x0', 'p', 'fwd:p'], ['fwd:xf'])(
            x0=[1] * n, p=rate, fwd_p=1
        )['fwd_xf']
        record['direct_error'] = float(
            np.max(abs(np.asarray(direct)[-1] + np.asarray(times) * expected))
        )
        actual = float(
            flow.factory('adjoint', ['x0', 'p', 'adj:xf'], ['adj:p'])(
                x0=[1] * n, p=rate, adj_xf=weight
            )['adj_p']
        )
        record.update(actual=actual, adjoint_error=abs(actual - record['analytic']))
        record['adjoint_accurate'] = record['adjoint_error'] < 3e-7
    except RuntimeError as exception:
        record.update(adjoint_accurate=False, error=str(exception))
    return record


def fingerprint(path):
    """Hash a source file to tie observations to a checkout snapshot."""
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    """Run the numerical matrix, retaining native failures as classified evidence."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='store_true')
    parser.add_argument('--formulation', choices=('native', 'public'), default='native')
    parser.add_argument('--coupled', action='store_true')
    parser.add_argument(
        '--configuration', choices=CONFIGURATIONS, default='default_bdf'
    )
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        print(
            json.dumps(
                stiff_case(args.formulation, args.coupled, args.configuration),
                allow_nan=False,
            )
        )
        return
    report = {
        'checkpoint': 'G1',
        'casadi': ca.__version__,
        'numpy': np.__version__,
        'python': sys.version,
        'platform': platform.platform(),
        'scaled_controls': check_scaled_controls(),
        'scaled_total_derivative': check_scaled_total_derivative(),
        'stiff_reductions': [],
        'option_references': [
            'https://web.casadi.org/python-api/index.html',
            'https://sundials.readthedocs.io/en/latest/cvodes/Mathematics_link.html',
        ],
    }
    for coupled in (False, True):
        for configuration in CONFIGURATIONS:
            for formulation in ('native', 'public'):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    '--case',
                    '--formulation',
                    formulation,
                    '--configuration',
                    configuration,
                ]
                if coupled:
                    command.append('--coupled')
                try:
                    run = subprocess.run(
                        command, capture_output=True, text=True, timeout=30
                    )
                    row = (
                        json.loads(run.stdout)
                        if run.returncode == 0
                        else {
                            'formulation': formulation,
                            'coupled': coupled,
                            'configuration': configuration,
                            'adjoint_accurate': False,
                        }
                    )
                    row['process_exit_code'] = run.returncode
                    # Omit verbose native input dumps, preserving the full Python exception.
                    row['stderr_first_line'] = (
                        run.stderr.splitlines()[0] if run.stderr else ''
                    )
                except subprocess.TimeoutExpired:
                    row = {
                        'formulation': formulation,
                        'coupled': coupled,
                        'configuration': configuration,
                        'timeout_s': 30,
                        'adjoint_accurate': False,
                    }
                report['stiff_reductions'].append(row)
    root = Path(__file__).resolve().parents[3]
    report['source_files'] = {
        p: fingerprint(root / p)
        for p in (
            'pyomo/dae/simulator.py',
            'pyomo/dae/tests/test_simulator_ode_qualification.py',
            'examples/dae/sensitivity/ode_qualification.py',
        )
    }
    report['commands'] = [
        'PYTHONPATH=<checkout> <env-python> examples/dae/sensitivity/ode_qualification.py --output <report.json>',
        'PYTHONPATH=<checkout> <env-python> -m pytest pyomo/dae/tests -q',
    ]
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
