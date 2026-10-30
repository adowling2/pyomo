# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Compare the public DAE map with refined implicit Radau sensitivities.

The original algebraic equation is present in both formulations. PyNumero
ASL is optional research infrastructure; no optimization solver is used.
The discrete adjoint is a transposed sparse linear solve, independent of IDAS.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import casadi as ca
import numpy as np
import scipy
from scipy.sparse.linalg import splu
import pyomo
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator

POINT = np.array([0.7, 1.4])


def model_at(discrete=False):
    """Keep z implicit; free parameter columns belong only to the discrete oracle."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 1.2))
    m.k = (
        pyo.Var(initialize=POINT[0])
        if discrete
        else pyo.Param(mutable=True, initialize=POINT[0])
    )
    m.a = (
        pyo.Var(initialize=POINT[1])
        if discrete
        else pyo.Param(mutable=True, initialize=POINT[1])
    )
    m.x, m.q, m.z = (
        pyo.Var(m.t, initialize=0),
        pyo.Var(m.t, initialize=0),
        pyo.Var(m.t, initialize=0),
    )
    m.dx, m.dq = DerivativeVar(m.x), DerivativeVar(m.q)
    m.ode = pyo.Constraint(m.t, rule=lambda m, t: m.dx[t] == -m.k * m.z[t])
    m.integral = pyo.Constraint(m.t, rule=lambda m, t: m.dq[t] == m.x[t])
    m.alg = pyo.Constraint(m.t, rule=lambda m, t: m.z[t] == m.a * m.x[t])
    if discrete:
        m.initial = pyo.Constraint(expr=m.x[0] == m.k**2)
        m.q[0].fix(0)
        m.objective = pyo.Objective(
            expr=2 * m.x[1.2] + 3 * m.q[1.2] + 4 * m.z[1.2] + m.k**2 + 0.2 * m.a**2
        )
    return m


def analytic(point):
    """Exact terminal outputs and total cost/gradient, including nonlinear ICs."""
    k, a = point
    t = 1.2
    e = math.exp(-k * a * t)
    x, q = k * k * e, k / a * (1 - e)
    dx = np.array([2 * k * e - a * t * x, -k * t * x])
    dq = np.array([(1 - e) / a + k * t * e, -k / a**2 * (1 - e) + k * k * t * e / a])
    dz = a * dx + np.array([0, x])
    cost = 2 * x + 3 * q + 4 * a * x + k * k + 0.2 * a * a
    gradient = 2 * dx + 3 * dq + 4 * dz + np.array([2 * k, 0.4 * a])
    return np.array([x, q, a * x]), np.vstack([dx, dq, dz]), cost, gradient


def scaled_error(value, reference):
    return float(
        np.max(np.abs(np.asarray(value) - reference) / (1 + np.abs(reference)))
    )


def public_map():
    """Check full direct products and a mixed x/q/z cost adjoint through compile."""
    m = model_at()
    flow = Simulator(m, package='casadi').compile(
        parameters=[m.k, m.a],
        times=[0, 1.2],
        integrator='idas',
        integrator_options={'reltol': 1e-11, 'abstol': 1e-13},
    )
    theta = ca.MX.sym('theta', 2)
    out = flow(x0=ca.vertcat(theta[0] ** 2, 0), p=theta, z0=1)
    terminal = ca.vertcat(out['xf'][:, -1], out['zf'][:, -1])
    cost = (
        2 * terminal[0]
        + 3 * terminal[1]
        + 4 * terminal[2]
        + theta[0] ** 2
        + 0.2 * theta[1] ** 2
    )
    fn = ca.Function(
        'comparison', [theta], [terminal, cost], ['theta'], ['terminal', 'J']
    )
    values = fn(theta=POINT)
    direct = fn.forward(2)(
        theta=POINT,
        out_terminal=values['terminal'],
        out_J=values['J'],
        fwd_theta=ca.DM.eye(2),
    )
    reverse = fn.reverse(1)(
        theta=POINT,
        out_terminal=values['terminal'],
        out_J=values['J'],
        adj_terminal=ca.DM.zeros(3),
        adj_J=1,
    )
    terminal, jac, cost, gradient = analytic(POINT)
    row = {
        'terminal': np.asarray(values['terminal']).ravel().tolist(),
        'cost': float(values['J']),
        'direct_gradient': np.asarray(direct['fwd_J']).ravel().tolist(),
        'adjoint_gradient': np.asarray(reverse['adj_theta']).ravel().tolist(),
        'errors': {
            'terminal': scaled_error(np.asarray(values['terminal']).ravel(), terminal),
            'cost': scaled_error(float(values['J']), cost),
            'terminal_direct': scaled_error(np.asarray(direct['fwd_terminal']), jac),
            'cost_direct': scaled_error(np.asarray(direct['fwd_J']).ravel(), gradient),
            'cost_adjoint': scaled_error(
                np.asarray(reverse['adj_theta']).ravel(), gradient
            ),
        },
    }
    assert max(row['errors'].values()) < 3e-7, row
    return row


def discrete_map(elements):
    """Solve linear state equations, then compare implicit direct/adjoint and FD."""
    m = model_at(discrete=True)
    pyo.TransformationFactory('dae.collocation').apply_to(
        m, wrt=m.t, nfe=elements, ncp=3, scheme='LAGRANGE-RADAU'
    )
    for t in m.t:
        m.dx[t].set_value(0)
        m.dq[t].set_value(0)
    nlp = PyomoNLP(m)
    variables = nlp.get_pyomo_variables()
    parameters = np.array(
        [next(i for i, v in enumerate(variables) if v is p) for p in (m.k, m.a)]
    )
    states = np.array([i for i in range(len(variables)) if i not in parameters])
    terminal_indices = [
        next(i for i, v in enumerate(variables) if v is p)
        for p in (m.x[1.2], m.q[1.2], m.z[1.2])
    ]
    terminal_positions = [int(np.flatnonzero(states == i)[0]) for i in terminal_indices]
    eq = nlp.get_pyomo_equality_constraints()
    lower = nlp.constraints_lb()[nlp.get_constraint_indices(eq)]
    assert len(lower) == len(states)
    initial = nlp.get_primals().copy()

    def solve(parameter):
        point = initial.copy()
        point[parameters] = parameter
        nlp.set_primals(point)
        for _ in range(2):
            matrix = nlp.evaluate_jacobian_eq().tocsc()[:, states]
            point[states] -= splu(matrix).solve(nlp.evaluate_eq_constraints() - lower)
            nlp.set_primals(point)
        residual = float(np.max(abs(nlp.evaluate_eq_constraints() - lower)))
        assert residual < 1e-10, residual
        return point.copy(), float(nlp.evaluate_objective()), residual

    point, value, residual = solve(POINT)
    jac = nlp.evaluate_jacobian_eq().tocsc()
    factor = splu(jac[:, states])
    rp = jac[:, parameters].toarray()
    sensitivity = -factor.solve(rp)
    phi = nlp.evaluate_grad_objective()
    direct = phi[parameters] + phi[states] @ sensitivity
    multiplier = factor.solve(phi[states], trans='T')
    adjoint = phi[parameters] - rp.T @ multiplier
    terminal, continuous_jac, cost, gradient = analytic(POINT)
    fd_rows = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5):
        fd = []
        for i in range(2):
            step = np.eye(2)[i] * h
            fd.append((solve(POINT + step)[1] - solve(POINT - step)[1]) / (2 * h))
        fd_rows.append(
            {'step': h, 'gradient': fd, 'discrete_error': scaled_error(fd, direct)}
        )
    row = {
        'elements': elements,
        'ncp': 3,
        'states': len(states),
        'equalities': len(lower),
        'residual': residual,
        'cost': value,
        'terminal': point[terminal_indices].tolist(),
        'direct_gradient': direct.tolist(),
        'adjoint_gradient': adjoint.tolist(),
        'discrete_duality_error': scaled_error(direct, adjoint),
        'continuous_errors': {
            'cost': scaled_error(value, cost),
            'terminal': scaled_error(point[terminal_indices], terminal),
            'terminal_sensitivity': scaled_error(
                sensitivity[terminal_positions], continuous_jac
            ),
            'cost_gradient': scaled_error(direct, gradient),
        },
        'finite_differences': fd_rows,
    }
    assert row['discrete_duality_error'] < 1e-7, row
    assert min(f['discrete_error'] for f in fd_rows) < 1e-7, row
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not AmplInterface.available():
        raise RuntimeError(
            'This optional comparison requires the PyNumero ASL extension'
        )
    meshes = [discrete_map(n) for n in (4, 8, 16)]
    for key in meshes[0]['continuous_errors']:
        errors = [m['continuous_errors'][key] for m in meshes]
        assert errors[0] > errors[1] > errors[2], (key, errors)
        assert errors[-1] < 3e-7, (key, errors)
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), root / 'pyomo/dae/simulator.py']
    asl = Path(AmplInterface.libname)
    report = {
        'purpose': 'G7 public DAE versus implicit Radau direct/adjoint sensitivities',
        'passed': True,
        'model': "x'=-k*z, q'=x, z=a*x, x0=k^2, q0=0; J=2*xT+3*qT+4*zT+k^2+.2*a^2",
        'point': POINT.tolist(),
        'public': public_map(),
        'meshes': meshes,
        'environment': {
            'python': sys.version,
            'pyomo': pyomo.__version__,
            'casadi': ca.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'platform': platform.platform(),
        },
        'ASL': {
            'path': str(asl),
            'sha256': hashlib.sha256(asl.read_bytes()).hexdigest(),
            'fresh_build': False,
            'optimization_solver_used': False,
        },
        'source_sha256': {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
