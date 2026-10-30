# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Compare public ODE derivatives with Radau equations and measure map costs.

ASL is required only for the collocation comparison. No optimization solver
is used. Benchmark cases run in isolated processes with a 60-second limit.
Timings are measurements, not unit-test acceptance thresholds.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import casadi as ca
import numpy as np
import scipy
from scipy.sparse.linalg import spsolve
import pyomo
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator


def decay_model(discrete=False):
    """Build identical state/integral equations for native flow and discretization."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = (
        pyo.Var(initialize=0.7) if discrete else pyo.Param(mutable=True, initialize=0.7)
    )
    model.x = pyo.Var(model.t, initialize=1)
    model.q = pyo.Var(model.t, initialize=0)
    model.dx = DerivativeVar(model.x)
    model.dq = DerivativeVar(model.q)
    model.ode = pyo.Constraint(model.t, rule=lambda m, t: m.dx[t] == -m.k * m.x[t])
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    if discrete:
        model.initial = pyo.Constraint(expr=model.x[0] == model.k**2)
        model.q[0].fix(0)
        model.objective = pyo.Objective(
            expr=2 * model.x[1.2] + 3 * model.q[1.2] + model.k**2
        )
    return model


def analytic_cost(rate):
    """Evaluate exact cost and total derivative including x0=k squared."""
    horizon = 1.2
    e = math.exp(-rate * horizon)
    x, q = rate**2 * e, rate * (1 - e)
    dx = (2 * rate - horizon * rate**2) * e
    dq = 1 - e + rate * horizon * e
    return 2 * x + 3 * q + rate**2, 2 * dx + 3 * dq + 2 * rate


def public_comparison():
    """Check compiled direct/adjoint total cost against the continuous oracle."""
    model = decay_model()
    flow = Simulator(model, package='casadi').compile(
        parameters=[model.k],
        times=[0, 0.2, 0.7, 1.2],
        integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
    )
    theta = ca.MX.sym('theta')
    states = flow(x0=ca.vertcat(theta**2, 0), p=theta)['xf']
    cost = ca.Function(
        'cost', [theta], [2 * states[0, -1] + 3 * states[1, -1] + theta**2]
    )
    point = 0.7
    expected, derivative = analytic_cost(point)
    value = cost(point)
    direct = float(cost.forward(1)(point, value, 1))
    adjoint = float(cost.reverse(1)(point, value, 1))
    result = {
        'value': float(value),
        'analytic_value': expected,
        'analytic_derivative': derivative,
        'direct': direct,
        'adjoint': adjoint,
        'value_error': abs(float(value) - expected),
        'direct_error': abs(direct - derivative),
        'adjoint_error': abs(adjoint - derivative),
    }
    if max(result[k] for k in ('value_error', 'direct_error', 'adjoint_error')) >= 3e-7:
        raise AssertionError(result)
    return result


def discrete_comparison(elements):
    """Differentiate the discrete equations including IC and explicit cost terms."""
    model = decay_model(discrete=True)
    pyo.TransformationFactory('dae.collocation').apply_to(
        model, wrt=model.t, nfe=elements, ncp=3, scheme='LAGRANGE-RADAU'
    )
    for t in model.t:
        x = 0.7**2 * math.exp(-0.7 * t)
        model.x[t].set_value(x)
        model.q[t].set_value(0.7 * (1 - math.exp(-0.7 * t)))
        model.dx[t].set_value(-0.7 * x)
        model.dq[t].set_value(x)
    nlp = PyomoNLP(model)
    variables = nlp.get_pyomo_variables()
    parameter = next(i for i, v in enumerate(variables) if v is model.k)
    states = np.array([i for i in range(len(variables)) if i != parameter])
    equalities = nlp.get_pyomo_equality_constraints()
    lower = nlp.constraints_lb()[nlp.get_constraint_indices(equalities)]
    initial = nlp.get_primals().copy()

    def solve(rate):
        """Solve the linear state system at a given rate without an NLP solver."""
        point = initial.copy()
        point[parameter] = rate
        nlp.set_primals(point)
        for _ in range(2):
            jacobian = nlp.evaluate_jacobian_eq().tocsr()
            point[states] -= spsolve(
                jacobian[:, states].tocsc(), nlp.evaluate_eq_constraints() - lower
            )
            nlp.set_primals(point)
        return (
            point.copy(),
            float(nlp.evaluate_objective()),
            float(np.max(abs(nlp.evaluate_eq_constraints() - lower))),
        )

    point, value, residual = solve(0.7)
    nlp.set_primals(point)
    jacobian = nlp.evaluate_jacobian_eq().tocsr()
    tangent = -spsolve(
        jacobian[:, states].tocsc(), jacobian[:, parameter].toarray().ravel()
    )
    objective_gradient = nlp.evaluate_grad_objective()
    derivative = float(
        objective_gradient[parameter] + objective_gradient[states] @ tangent
    )
    expected, continuous = analytic_cost(0.7)
    differences = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5):
        _, plus, _ = solve(0.7 + step)
        _, minus, _ = solve(0.7 - step)
        fd = (plus - minus) / (2 * step)
        differences.append(
            {'step': step, 'derivative': fd, 'discrete_fd_error': abs(fd - derivative)}
        )
    row = {
        'elements': elements,
        'ncp': 3,
        'states': len(states),
        'equalities': len(lower),
        'residual': residual,
        'cost': value,
        'cost_discretization_error': abs(value - expected),
        'discrete_derivative': derivative,
        'continuous_derivative': continuous,
        'derivative_discretization_error': abs(derivative - continuous),
        'finite_differences': differences,
    }
    if residual >= 1e-10 or min(r['discrete_fd_error'] for r in differences) >= 1e-7:
        raise AssertionError(row)
    return row


def benchmark(observations, intervals, tolerance):
    """Time construction and repeated native calls on comparable affine ODEs."""
    model = pyo.ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 1.2))
    model.k = pyo.Param(mutable=True, initialize=-0.7)
    model.u = pyo.Var(model.t) if intervals else pyo.Param(initialize=0.1)
    model.x = pyo.Var(model.t)
    model.q = pyo.Var(model.t)
    model.dx = DerivativeVar(model.x)
    model.dq = DerivativeVar(model.q)
    model.ode = pyo.Constraint(
        model.t,
        rule=lambda m, t: m.dx[t] == m.k * m.x[t] + (m.u[t] if intervals else m.u),
    )
    model.integral = pyo.Constraint(model.t, rule=lambda m, t: m.dq[t] == m.x[t])
    simulator = Simulator(model, package='casadi')
    times = np.linspace(0, 1.2, observations)
    options = {
        'parameters': [model.k],
        'times': times,
        'integrator_options': {'reltol': tolerance, 'abstol': tolerance / 100},
    }
    if intervals:
        options.update(
            controls=[model.u], control_times=np.linspace(0, 1.2, intervals + 1)
        )
    construction = []
    for _ in range(3):
        start = perf_counter()
        flow = simulator.compile(**options)
        construction.append(perf_counter() - start)
    point = {'x0': [1.3, 0], 'p': -0.7}
    if intervals:
        point['u'] = ca.DM.ones(1, intervals) * 0.1
    start = perf_counter()
    tangent = flow.factory('direct', flow.name_in() + ['fwd:p'], ['fwd:xf'])
    adjoint = flow.factory(
        'adjoint', flow.name_in() + ['adj:xf'], ['adj:' + n for n in flow.name_in()]
    )
    derivative_construction = perf_counter() - start
    weights = np.zeros((2, observations))
    weights[:, -1] = [2, 3]
    calls = {
        'primal': lambda: flow(**point),
        'jvp': lambda: tangent(**point, fwd_p=1),
        'vjp': lambda: adjoint(**point, adj_xf=weights),
    }
    durations = {}
    values = {}
    for name, call in calls.items():
        call()
        samples = []
        for _ in range(5):
            start = perf_counter()
            values[name] = call()
            samples.append(perf_counter() - start)
        durations[name] = statistics.median(samples)
    rate = -0.7
    e = np.exp(rate * times)
    g = np.expm1(rate * times) / rate
    expected = np.vstack([1.3 * e + 0.1 * g, 1.3 * g + 0.1 * (g - times) / rate])
    error = float(np.max(abs(np.asarray(values['primal']['xf']) - expected)))
    gp = ((rate * times - 1) * e + 1) / rate**2
    parameter_derivative = np.vstack(
        [1.3 * times * e + 0.1 * gp, 1.3 * gp + 0.1 * (gp - (g - times) / rate) / rate]
    )
    direct_error = float(
        np.max(abs(np.asarray(values['jvp']['fwd_xf']) - parameter_derivative))
    )
    adjoint_error = abs(
        float(values['vjp']['adj_p']) - float(np.sum(weights * parameter_derivative))
    )
    return {
        'observations': observations,
        'control_intervals': intervals,
        'state_count': 2,
        'compile_seconds_median': statistics.median(construction),
        'derivative_construction_seconds': derivative_construction,
        'execution_seconds_median': durations,
        'repeats': 5,
        'compile_repeats': 3,
        'primal_error': error,
        'direct_error': direct_error,
        'adjoint_parameter_error': adjoint_error,
        'accuracy_passed': max(error, direct_error, adjoint_error) < 3e-7,
        'integrator_options': options['integrator_options'],
        'function_nodes': flow.n_nodes(),
        'input_shapes': {n: list(flow.size_in(n)) for n in flow.name_in()},
    }


def fingerprint(path):
    """Identify source and ASL binaries used for the recorded comparison."""
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    """Record collocation qualification and bounded, untimed-assertion cost data."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--observations', type=int, default=10)
    parser.add_argument('--intervals', type=int, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-11)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.benchmark:
        print(
            json.dumps(
                benchmark(args.observations, args.intervals, args.tolerance),
                allow_nan=False,
            )
        )
        return
    report = {
        'checkpoint': 'G2',
        'casadi': ca.__version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'pyomo': pyomo.__version__,
        'python': sys.version,
        'platform': platform.platform(),
        'public_interface': public_comparison(),
        'collocation': {'available': AmplInterface.available()},
        'benchmarks': [],
    }
    if report['collocation']['available']:
        rows = [discrete_comparison(n) for n in (4, 8, 16)]
        decreasing = all(
            a['derivative_discretization_error'] > b['derivative_discretization_error']
            for a, b in zip(rows, rows[1:])
        )
        passed = decreasing and rows[-1]['derivative_discretization_error'] < 3e-7
        report['collocation'].update(
            rows=rows,
            passed=passed,
            asl_library=str(AmplInterface.libname),
            asl_fingerprint=fingerprint(Path(AmplInterface.libname)),
        )
        if not passed:
            raise AssertionError(report['collocation'])
    for observations, intervals in (
        (10, 0),
        (10, 4),
        (50, 0),
        (50, 20),
        (100, 0),
        (100, 100),
        (400, 0),
        (400, 100),
        (100, 400),
    ):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--benchmark',
            '--observations',
            str(observations),
            '--intervals',
            str(intervals),
            '--tolerance',
            str(args.tolerance),
        ]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            row = (
                json.loads(run.stdout)
                if run.returncode == 0
                else {
                    'observations': observations,
                    'control_intervals': intervals,
                    'error': run.stderr[-1500:],
                }
            )
            row['process_exit_code'] = run.returncode
        except subprocess.TimeoutExpired:
            row = {
                'observations': observations,
                'control_intervals': intervals,
                'timeout_s': 60,
            }
        report['benchmarks'].append(row)
        if row.get('process_exit_code') != 0:
            break
    report['benchmark_qualification'] = (
        'Timing medians on one machine; no portable performance guarantee. VJPs return control-amplitude gradients only on controlled maps.'
    )
    report['gate_passed'] = (
        report['collocation'].get('passed', False)
        and len(report['benchmarks']) == 9
        and all(
            row.get('process_exit_code') == 0 and row.get('accuracy_passed')
            for row in report['benchmarks']
        )
    )
    root = Path(__file__).resolve().parents[3]
    report['source_files'] = {
        p: fingerprint(root / p)
        for p in (
            'pyomo/dae/simulator.py',
            'examples/dae/sensitivity/ode_comparison.py',
        )
    }
    report['command'] = (
        'PYTHONPATH=<checkout> <env-python> '
        'examples/dae/sensitivity/ode_comparison.py '
        '--tolerance %s --output <report.json>' % args.tolerance
    )
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
