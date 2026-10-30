# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

from pyomo.core.base import Constraint, Param, Var, value, Suffix, Block
from pyomo.core.base.param import ParamData
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
    attempt_import,
    check_min_version,
)

logger = logging.getLogger('pyomo.core')

casadi_intrinsic = {}


def _finalize_casadi(casadi, available):
    if available:
        casadi_intrinsic.update(
            {
                'log': casadi.log,
                'log10': casadi.log10,
                'sin': casadi.sin,
                'cos': casadi.cos,
                'tan': casadi.tan,
                'cosh': casadi.cosh,
                'sinh': casadi.sinh,
                'tanh': casadi.tanh,
                'asin': casadi.asin,
                'acos': casadi.acos,
                'atan': casadi.atan,
                'exp': casadi.exp,
                'sqrt': casadi.sqrt,
                'asinh': casadi.asinh,
                'acosh': casadi.acosh,
                'atanh': casadi.atanh,
                'ceil': casadi.ceil,
                'floor': casadi.floor,
            }
        )


casadi, casadi_available = attempt_import('casadi', callback=_finalize_casadi)


def _check_getitemexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    GetItemExpression at expr.arg(i) to see if it is a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the :py:class:`DerivativeVar<DerivativeVar>` and
    the RHS. If not, return None.
    """
    if type(expr.arg(i).arg(0)) is DerivativeVar:
        return [expr.arg(i), expr.arg(1 - i)]
    else:
        return None


def _check_productexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    ProductExpression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    expr_ = expr.arg(i)
    stack = [(expr_, 1)]
    pterms = []
    dv = None

    while stack:
        curr, e_ = stack.pop()
        if curr.__class__ is EXPR.ProductExpression:
            stack.append((curr.arg(0), e_))
            stack.append((curr.arg(1), e_))
        elif curr.__class__ is EXPR.DivisionExpression:
            stack.append((curr.arg(0), e_))
            stack.append((curr.arg(1), -e_))
        elif (
            isinstance(curr, EXPR.GetItemExpression)
            and type(curr.arg(0)) is DerivativeVar
        ):
            dv = (curr, e_)
        else:
            pterms.append((curr, e_))

    if dv is None:
        return None

    numerator = 1
    denom = 1
    for term, e_ in pterms:
        if e_ == 1:
            denom *= term
        else:
            numerator *= term
    curr, e_ = dv
    if e_ == 1:
        return [curr, expr.arg(1 - i) * numerator / denom]
    else:
        return [curr, denom / (expr.arg(1 - i) * numerator)]


def _check_negationexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    NegationExpression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    arg = expr.arg(i).arg(0)

    if isinstance(arg, EXPR.GetItemExpression) and type(arg.arg(0)) is DerivativeVar:
        return [arg, -expr.arg(1 - i)]

    if type(arg) is EXPR.ProductExpression:
        lhs = arg.arg(0)
        rhs = arg.arg(1)

        if not (type(lhs) in native_numeric_types or not lhs.is_potentially_variable()):
            return None
        if not (
            isinstance(rhs, EXPR.GetItemExpression)
            and type(rhs.arg(0)) is DerivativeVar
        ):
            return None

        return [rhs, -expr.arg(1 - i) / lhs]

    return None


def _check_viewsumexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    Sum expression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    # Get the side of the equality expression with the derivative variable
    sumexp = expr.arg(i)
    items = []
    dv = None
    dvcoef = 1

    for idx, item in enumerate(sumexp.args):
        if dv is not None:
            items.append(item)
        elif (
            isinstance(item, EXPR.GetItemExpression)
            and type(item.arg(0)) is DerivativeVar
        ):
            dv = item
        elif type(item) is EXPR.ProductExpression:
            lhs = item.arg(0)
            rhs = item.arg(1)
            if (
                type(lhs) in native_numeric_types or not lhs.is_potentially_variable()
            ) and (
                isinstance(rhs, EXPR.GetItemExpression)
                and type(rhs.arg(0)) is DerivativeVar
            ):
                dv = rhs
                dvcoef = lhs
            elif (
                type(rhs) in native_numeric_types or not rhs.is_potentially_variable()
            ) and (
                isinstance(lhs, EXPR.GetItemExpression)
                and type(lhs.arg(0)) is DerivativeVar
            ):
                dv = lhs
                dvcoef = rhs
            else:
                items.append(item)
        else:
            items.append(item)

    if dv is not None:
        # Form the "other" side of the equality expression
        RHS = expr.arg(1 - i)
        for item in items:
            RHS -= item
        RHS = RHS / dvcoef
        return [dv, RHS]

    return None


class Pyomo2Scipy_Visitor(EXPR.ExpressionReplacementVisitor):
    """
    Expression walker that replaces _GetItemExpression
    instances with mutable parameters.
    """

    def __init__(self, templatemap):
        # Note because we are creating a "nonPyomo" expression tree, we
        # want to remove all Expression nodes (as opposed to replacing
        # them in place)
        super().__init__(
            descend_into_named_expressions=True, remove_named_expressions=True
        )
        self.templatemap = templatemap

    def beforeChild(self, node, child, child_idx):
        if type(child) is IndexTemplate:
            return False, child

        if isinstance(child, EXPR.GetItemExpression):
            _id = _GetItemIndexer(child)
            if _id not in self.templatemap:
                self.templatemap[_id] = Param(mutable=True)
                self.templatemap[_id].construct()
                self.templatemap[_id]._name = "%s[%s]" % (
                    _id.base.name,
                    ','.join(str(x) for x in _id.args),
                )
            return False, self.templatemap[_id]

        return super().beforeChild(node, child, child_idx)


def convert_pyomo2scipy(expr, templatemap):
    """Substitute _GetItem nodes in an expression tree.

    This substitution function is used to replace Pyomo _GetItem
    nodes with mutable Params.

    Args:
        templatemap: dictionary mapping _GetItemIndexer objects to
            mutable params

    Returns:
        a new expression tree with all substitutions done
    """
    if not scipy_available:
        raise DAE_Error(
            "SciPy is not installed. Cannot substitute SciPy intrinsic functions."
        )
    visitor = Pyomo2Scipy_Visitor(templatemap)
    return visitor.walk_expression(expr)


class Substitute_Pyomo2Casadi_Visitor(EXPR.ExpressionReplacementVisitor):
    """
    Expression walker that replaces

       * _UnaryFunctionExpression instances with unary functions that
         point to casadi intrinsic functions.

       * _GetItemExpressions with _GetItemIndexer objects that references
         CasADi variables.
    """

    def __init__(self, templatemap):
        # Note because we are creating a "nonPyomo" expression tree, we
        # want to remove all Expression nodes (as opposed to replacing
        # them in place)
        super().__init__(
            descend_into_named_expressions=True, remove_named_expressions=True
        )
        self.templatemap = templatemap

    def exitNode(self, node, data):
        """Replace a node if it's a unary function."""
        ans = super().exitNode(node, data)
        if type(ans) in (
            EXPR.UnaryFunctionExpression,
            EXPR.NPV_UnaryFunctionExpression,
        ):
            return EXPR.UnaryFunctionExpression(
                ans.args, ans.getname(), casadi_intrinsic[ans.getname()]
            )
        return ans

    def beforeChild(self, node, child, child_idx):
        """Replace a node if it's a _GetItemExpression."""
        if isinstance(child, EXPR.GetItemExpression):
            _id = _GetItemIndexer(child)
            if _id not in self.templatemap:
                name = "%s[%s]" % (_id.base.name, ','.join(str(x) for x in _id.args))
                self.templatemap[_id] = casadi.SX.sym(name)
            return False, self.templatemap[_id]

        elif type(child) is IndexTemplate:
            return False, child

        return super().beforeChild(node, child, child_idx)


class Convert_Pyomo2Casadi_Visitor(EXPR.ExpressionValueVisitor):
    """
    Expression walker that evaluates an expression
    generated by the Substitute_Pyomo2Casadi_Visitor walker.

    In Coopr3 this walker was not necessary because the expression could
    be simply evaluated.  But in Pyomo5, the evaluation logic was
    changed to be non-recursive, which involves checks on the types of
    leaves in the expression tree. Hence, the evaluation logic fails if
    leaves in the tree are not standard Pyomo5 variable types.
    """

    def __init__(self, time=None, substitutions=None):
        """Optionally replace time and selected scalar leaves by CasADi symbols."""
        self.time = time
        self.substitutions = substitutions if substitutions is not None else {}

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_numeric_types:
            return True, node

        if node.__class__ is casadi.SX:
            return True, node

        if id(node) in self.substitutions:
            return True, self.substitutions[id(node)]

        if type(node) is IndexTemplate and self.time is not None:
            return True, self.time

        if node.is_variable_type():
            return True, value(node)

        if not node.is_expression_type():
            return True, value(node)

        return False, None


def substitute_pyomo2casadi(expr, templatemap):
    """Substitute IndexTemplates in an expression tree.

    This substitution function is used to replace Pyomo intrinsic
    functions with CasADi functions.

    Args:
        expr: a Pyomo expression
        templatemap: dictionary mapping _GetItemIndexer objects to
            mutable params

    Returns:
        a new expression tree with all substitutions done
    """
    if not casadi_available:
        raise DAE_Error(
            "CASADI is not installed.  Cannot substitute CasADi "
            "variables and intrinsic functions."
        )
    visitor = Substitute_Pyomo2Casadi_Visitor(templatemap)
    return visitor.walk_expression(expr)


def convert_pyomo2casadi(expr, time=None, substitutions=None):
    """Convert a Pyomo expression tree to Casadi.

    This function replaces a Pyomo expression with a CasADi expression.
    This assumes that the `substitute_pyomo2casadi` function has
    been called, so the Pyomo expression contains CasADi variables
    and intrinsic functions.  The resulting expression can be used
    with the CasADi integrator.

    Args:
        expr: a Pyomo expression with CasADi variables and intrinsic
            functions
        time: optional CasADi expression representing the independent variable
        substitutions: optional dictionary mapping scalar leaf identities to
            CasADi symbols

    Returns:
        a CasADi expression tree.
    """
    if not casadi_available:
        raise DAE_Error(
            "CASADI is not installed.  Cannot convert a Pyomo "
            "expression to a Casadi expression."
        )
    visitor = Convert_Pyomo2Casadi_Visitor(time, substitutions)
    return visitor.dfs_postorder_stack(expr)


class Simulator:
    """
    Simulator objects allow a user to simulate a dynamic model formulated
    using pyomo.dae.

    Parameters
    ----------
    m : Pyomo Model
        The Pyomo model to be simulated should be passed as the first argument

    package : `string`
        The Python simulator package to use. Currently 'scipy' and 'casadi' are
        the only supported packages
    """

    def __init__(self, m, package='scipy'):
        self._intpackage = package
        if self._intpackage not in ['scipy', 'casadi']:
            raise DAE_Error(
                "Unrecognized simulator package %s. Please select from "
                "%s" % (self._intpackage, ['scipy', 'casadi'])
            )

        if self._intpackage == 'scipy':
            if not scipy_available:
                # Converting this to a warning so that Simulator initialization
                # can be tested even when scipy is unavailable
                logger.warning(
                    "The scipy module is not available. "
                    "You may build the Simulator object but you will not "
                    "be able to run the simulation."
                )
        else:
            if not casadi_available:
                # Initializing the simulator for use with casadi requires
                # access to casadi objects. Therefore, we must throw an error
                # here instead of a warning.
                raise ValueError(
                    "The casadi module is not available. Cannot simulate model."
                )

        # Check for active Blocks and throw error if any are found
        if len(list(m.component_data_objects(Block, active=True, descend_into=False))):
            raise DAE_Error(
                "The Simulator cannot handle hierarchical models at the moment."
            )

        temp = m.component_map(ContinuousSet)
        if len(temp) != 1:
            raise DAE_Error(
                "Currently the simulator may only be applied to "
                "Pyomo models with a single ContinuousSet"
            )

        # Get the ContinuousSet in the model
        contset = list(temp.values())[0]

        # Create a index template for the continuous set
        cstemplate = IndexTemplate(contset)

        # Ensure that there is at least one derivative in the model
        derivs = m.component_map(DerivativeVar)
        derivs = list(derivs.keys())

        if hasattr(m, '_pyomo_dae_reclassified_derivativevars'):
            for d in m._pyomo_dae_reclassified_derivativevars:
                derivs.append(d.name)
        if len(derivs) == 0:
            raise DAE_Error("Cannot simulate a model with no derivatives")

        templatemap = {}  # Map for template substituter
        rhsdict = {}  # Map of derivative to its RHS templated expr
        derivlist = []  # Ordered list of derivatives
        alglist = []  # list of templated algebraic equations

        # Loop over constraints to find differential equations with separable
        # RHS. Must find a RHS for every derivative var otherwise ERROR. Build
        # dictionary of DerivativeVar:RHS equation.
        for con in m.component_objects(Constraint, active=True):
            # Skip the discretization equations if model is discretized
            if '_disc_eq' in con.name:
                continue

            # Check dimension of the Constraint. Check if the
            # Constraint is indexed by the continuous set and
            # determine its order in the indexing sets
            if con.dim() == 0:
                continue

            conindex = con.index_set()
            # if not hasattr(conindex, 'set_tuple'):
            #     # Check if the continuous set is the indexing set
            #     if conindex is not contset:
            #         continue
            #     else:
            #         csidx = 0
            #         noncsidx = (None,)
            # else:
            dimsum = 0
            csidx = -1
            noncsidx = None
            for s in conindex.subsets():
                if s is contset:
                    if csidx != -1:
                        raise DAE_Error(
                            "Cannot simulate the constraint %s because "
                            "it is indexed by duplicate ContinuousSets" % con.name
                        )
                    csidx = dimsum
                elif noncsidx is None:
                    noncsidx = s
                else:
                    noncsidx = noncsidx.cross(s)
                dimsum += s.dimen
            if csidx == -1:
                continue
            if noncsidx is None:
                noncsidx = (None,)

            # Get the rule used to construct the constraint
            conrule = con.rule

            for i in noncsidx:
                # Insert the index template and call the rule to
                # create a templated expression
                if i is None:
                    tempexp = conrule(m, cstemplate)
                else:
                    if not isinstance(i, tuple):
                        i = (i,)
                    tempidx = i[0:csidx] + (cstemplate,) + i[csidx:]
                    tempexp = conrule(m, tempidx)

                # Check to make sure it's an EqualityExpression
                if not type(tempexp) is EXPR.EqualityExpression:
                    continue

                # Check to make sure it's a differential equation with
                # separable RHS
                args = None
                # Case 1: m.dxdt[t] = RHS
                if isinstance(tempexp.arg(0), EXPR.GetItemExpression):
                    args = _check_getitemexpression(tempexp, 0)

                # Case 2: RHS = m.dxdt[t]
                if args is None:
                    if isinstance(tempexp.arg(1), EXPR.GetItemExpression):
                        args = _check_getitemexpression(tempexp, 1)

                # Case 3: m.p*m.dxdt[t] = RHS
                if args is None:
                    if (
                        type(tempexp.arg(0)) is EXPR.ProductExpression
                        or type(tempexp.arg(0)) is EXPR.DivisionExpression
                    ):
                        args = _check_productexpression(tempexp, 0)

                # Case 4: RHS =  m.p*m.dxdt[t]
                if args is None:
                    if (
                        type(tempexp.arg(1)) is EXPR.ProductExpression
                        or type(tempexp.arg(1)) is EXPR.DivisionExpression
                    ):
                        args = _check_productexpression(tempexp, 1)

                # Case 5: m.dxdt[t] + sum(ELSE) = RHS
                # or CONSTANT + m.dxdt[t] = RHS
                if args is None:
                    if type(tempexp.arg(0)) is EXPR.SumExpression:
                        args = _check_viewsumexpression(tempexp, 0)

                # Case 6: RHS = m.dxdt[t] + sum(ELSE)
                if args is None:
                    if type(tempexp.arg(1)) is EXPR.SumExpression:
                        args = _check_viewsumexpression(tempexp, 1)

                # Case 7: RHS = m.p*m.dxdt[t] + CONSTANT
                # This case will be caught by Case 6 if p is immutable. If
                # p is mutable then this case will not be detected as a
                # separable differential equation

                # Case 8: - dxdt[t] = RHS
                if args is None:
                    if type(tempexp.arg(0)) is EXPR.NegationExpression:
                        args = _check_negationexpression(tempexp, 0)

                # Case 9: RHS = - dxdt[t]
                if args is None:
                    if type(tempexp.arg(1)) is EXPR.NegationExpression:
                        args = _check_negationexpression(tempexp, 1)

                # At this point if args is not None then args[0] contains
                # the _GetItemExpression for the DerivativeVar and args[1]
                # contains the RHS expression. If args is None then the
                # constraint is considered an algebraic equation
                if args is None:
                    # Constraint is an algebraic equation or unsupported
                    # differential equation
                    if self._intpackage == 'scipy':
                        raise DAE_Error(
                            "Model contains an algebraic equation or "
                            "unrecognized differential equation. Constraint "
                            "'%s' cannot be simulated using Scipy. If you are "
                            "trying to simulate a DAE model you must use "
                            "CasADi as the integration package." % str(con.name)
                        )
                    tempexp = tempexp.arg(0) - tempexp.arg(1)
                    algexp = substitute_pyomo2casadi(tempexp, templatemap)
                    alglist.append(algexp)
                    continue

                # Add the differential equation to rhsdict and derivlist
                dv = args[0]
                RHS = args[1]
                dvkey = _GetItemIndexer(dv)
                if dvkey in rhsdict.keys():
                    raise DAE_Error(
                        "Found multiple RHS expressions for the "
                        "DerivativeVar %s" % str(dvkey)
                    )

                derivlist.append(dvkey)
                if self._intpackage == 'casadi':
                    rhsdict[dvkey] = substitute_pyomo2casadi(RHS, templatemap)
                else:
                    rhsdict[dvkey] = convert_pyomo2scipy(RHS, templatemap)
        # Check to see if we found a RHS for every DerivativeVar in
        # the model
        # FIXME: Not sure how to rework this for multi-index case
        # allderivs = derivs.keys()
        # if set(allderivs) != set(derivlist):
        #     missing = list(set(allderivs)-set(derivlist))
        #     print("WARNING: Could not find a RHS expression for the "
        #     "following DerivativeVar components "+str(missing))

        # Create ordered list of differential variables corresponding
        # to the list of derivatives.
        diffvars = []

        for deriv in derivlist:
            sv = deriv.base.get_state_var()
            diffvars.append(_GetItemIndexer(sv[deriv._args]))

        # Create ordered list of algebraic variables and time-varying
        # parameters
        algvars = []

        for item in templatemap.keys():
            if item.base.name in derivs:
                # Make sure there are no DerivativeVars in the
                # template map
                raise DAE_Error(
                    "Cannot simulate a differential equation with "
                    "multiple DerivativeVars"
                )
            if item not in diffvars:
                # Finds time varying parameters and algebraic vars
                algvars.append(item)

        if self._intpackage == 'scipy':
            # Function sent to scipy integrator
            def _rhsfun(t, x):
                residual = []
                cstemplate.set_value(t)
                for idx, v in enumerate(diffvars):
                    if v in templatemap:
                        templatemap[v].set_value(x[idx])

                for d in derivlist:
                    residual.append(value(rhsdict[d]))

                return residual

            self._rhsfun = _rhsfun

        # Add any diffvars not added by expression walker to self._templatemap
        if self._intpackage == 'casadi':
            for _id in diffvars:
                if _id not in templatemap:
                    name = "%s[%s]" % (
                        _id.base.name,
                        ','.join(str(x) for x in _id.args),
                    )
                    templatemap[_id] = casadi.SX.sym(name)

        self._contset = contset
        self._cstemplate = cstemplate
        self._diffvars = diffvars
        self._derivlist = derivlist
        self._templatemap = templatemap
        self._rhsdict = rhsdict
        self._alglist = alglist
        self._algvars = algvars
        self._model = m
        self._tsim = None
        self._simsolution = None
        # The algebraic vars in the most recent simulation
        self._simalgvars = None
        # The time-varying inputs in the most recent simulation
        self._siminputvars = None

    def get_variable_order(self, vartype=None):
        """
        This function returns the ordered list of differential variable
        names. The order corresponds to the order being sent to the
        integrator function. Knowing the order allows users to provide
        initial conditions for the differential equations using a
        list or map the profiles returned by the simulate function to
        the Pyomo variables.

        Parameters
        ----------
        vartype : `string` or None
            Optional argument for specifying the type of variables to return
            the order for. The default behavior is to return the order of
            the differential variables. 'time-varying' will return the order
            of all the time-dependent algebraic variables identified in the
            model. 'algebraic' will return the order of algebraic variables
            used in the most recent call to the simulate function. 'input'
            will return the order of the time-dependent algebraic variables
            that were treated as inputs in the most recent call to the
            simulate function.

        Returns
        -------
        `list`

        """
        if vartype == 'time-varying':
            return self._algvars
        elif vartype == 'algebraic':
            return self._simalgvars
        elif vartype == 'input':
            return self._siminputvars
        else:
            return self._diffvars

    def compile(
        self,
        *,
        parameters,
        times,
        controls=(),
        control_times=None,
        integrator='cvodes',
        integrator_options=None,
    ):
        """Compile a differentiable ODE or index-one DAE map without evaluation.

        Parameters
        ----------
        parameters : iterable of mutable Param data
            Ordered scalar parameters to retain symbolically. Entries of indexed
            Params may be selected individually. Other scalar data are captured
            at compilation. An empty iterable permits initial-state derivatives.
        times : sequence of float
            Nonempty, finite, strictly increasing observation times within the
            ContinuousSet bounds. The initial time may be included.
        controls : iterable of time-indexed Var or Param components, optional
            Ordered scalar controls, each indexed solely by the ContinuousSet.
            All independent time-varying inputs must be selected. DAE controls
            are not yet supported.
        control_times : sequence of float, optional
            Fixed, strictly increasing interval boundaries, including both
            ContinuousSet bounds. Required when controls are selected. Input
            'u' has one row per control and one column per interval. Amplitudes
            apply on [control_times[j], control_times[j+1]); the terminal state
            is reached with the last interval's amplitude.
        integrator : str
            Use 'cvodes' for ODEs or explicitly select 'idas' for DAEs.
            Requires package='casadi' and CasADi 3.8.1 or later.
        integrator_options : dict, optional
            CasADi integration options, copied before use. Only positive scalar
            tolerances are supported; time-grid and derivative augmentation
            overrides are rejected. Quadrature error control is required and
            defaults to True to control adjoint parameter-gradient accuracy.
            On CasADi 3.8.1, max_num_dir is restricted to 1: CasADi composes
            single-direction calls to avoid its batched-adjoint defect.
            DAEs also require calc_ic=True, calc_icB=True and
            suppress_algebraic=False. Their default abstol is 1e-8; it also
            sets the algebraic root residual target.

        Returns
        -------
        casadi.Function
            A function with inputs 'x0' and 'p', plus 'u' when controls are
            selected, and output 'xf'. DAEs additionally have input 'z0' and
            output 'zf'. z0 is an algebraic solver guess, not an independent
            physical IC; its local derivative is zero on a fixed branch.
            Differential state rows follow
            get_variable_order(), parameter entries follow parameters, and output
            columns follow times. Algebraic rows follow the remaining
            get_variable_order('time-varying') entries after controls are removed.
            CasADi forward/reverse factories provide direct
            and adjoint derivative products. x0, p, and u are independent inputs;
            compose their dependencies explicitly for total derivatives.

        Notes
        -----
        DAEs require a locally nonsingular algebraic Jacobian on the selected
        branch. Initial and observed algebraic states are reconstructed with a
        differentiable root solve. No global index or branch analysis is done.
        Derivatives of switch times are not supported. The
        model, caller options, and previous simulation results are not modified.
        This method does not call initialize_model().
        """
        if self._intpackage != 'casadi' or integrator not in ('cvodes', 'idas'):
            raise DAE_Error(
                "Simulator.compile requires package='casadi' and integrator='cvodes' or 'idas'"
            )
        if not numpy_available:
            raise ValueError(
                "The numpy module is not available. Cannot compile the model."
            )
        if not check_min_version(casadi, '3.8.1'):
            raise DAE_Error("Simulator.compile requires CasADi 3.8.1 or later")
        if not self._diffvars:
            raise DAE_Error(
                "Simulator.compile requires at least one differential equation"
            )
        if self._alglist and integrator != 'idas':
            raise DAE_Error("Algebraic equations require integrator='idas'")
        if not self._alglist and integrator != 'cvodes':
            raise DAE_Error("ODE compilation requires integrator='cvodes'")

        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or times.size == 0
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError(
                "times must be a nonempty, finite, strictly increasing sequence"
            )
        if times[0] < self._contset.first() or times[-1] > self._contset.last():
            raise ValueError("times must lie within the ContinuousSet bounds")

        if isinstance(controls, (set, frozenset, Mapping)):
            raise TypeError("controls must be an ordered iterable")
        controls = list(controls)
        control_variables = []
        for control in controls:
            if not isinstance(control, (Var, Param)) or not control.is_indexed():
                raise ValueError(
                    "Each control must be a time-indexed Var or Param component"
                )
            if control.model() is not self._model.model():
                raise ValueError(
                    "Selected control %s belongs to a different model" % control.name
                )
            if control.index_set() is not self._contset:
                raise ValueError("Controls must be indexed solely by the ContinuousSet")
            matches = [v for v in self._algvars if v.base is control]
            if not matches:
                raise ValueError(
                    "Selected control %s is unused in the ODE" % control.name
                )
            if (
                len(matches) != 1
                or len(matches[0]._args) != 1
                or matches[0]._args[0] is not self._cstemplate
            ):
                raise ValueError("Controls must be indexed by the current time only")
            if matches[0] in control_variables:
                raise ValueError("Duplicate selected control %s" % control.name)
            control_variables.append(matches[0])
        algebraic_variables = [v for v in self._algvars if v not in control_variables]
        if not self._alglist and algebraic_variables:
            raise DAE_Error("Select all time-varying inputs through controls")
        if self._alglist and len(algebraic_variables) != len(self._alglist):
            raise DAE_Error(
                "Algebraic equations must form a square system; select independent inputs through controls"
            )
        if self._alglist and controls:
            raise DAE_Error("Simulator.compile does not yet support DAE controls")
        if controls:
            control_times = np.asarray(control_times, dtype=float)
            if (
                control_times.ndim != 1
                or control_times.size < 2
                or not np.isfinite(control_times).all()
                or (np.diff(control_times) <= 0).any()
            ):
                raise ValueError(
                    "control_times must be a finite, strictly increasing sequence of interval boundaries"
                )
            if (
                control_times[0] != self._contset.first()
                or control_times[-1] != self._contset.last()
            ):
                raise ValueError("control_times must include both ContinuousSet bounds")
        elif control_times is not None:
            raise ValueError("control_times requires selected controls")

        if isinstance(parameters, (set, frozenset, Mapping)):
            raise TypeError(
                "parameters must be an ordered iterable of mutable Param data"
            )
        parameters = list(parameters)
        symbols = casadi.SX.sym('p', len(parameters))
        substitutions = {}
        for index, parameter in enumerate(parameters):
            if (
                not isinstance(parameter, ParamData)
                or not parameter.parent_component().mutable
            ):
                raise ValueError(
                    "Each selected parameter must be mutable scalar Param data"
                )
            if parameter.model() is not self._model.model():
                raise ValueError(
                    "Selected parameter %s belongs to a different model"
                    % parameter.name
                )
            if any(
                s is self._contset
                for s in parameter.parent_component().index_set().subsets()
            ):
                raise ValueError(
                    "Select time-indexed parameters through controls, not parameters"
                )
            if id(parameter) in substitutions:
                raise ValueError("Duplicate selected parameter %s" % parameter.name)
            substitutions[id(parameter)] = symbols[index]

        if integrator_options is not None and not isinstance(
            integrator_options, Mapping
        ):
            raise TypeError("integrator_options must be a mapping")
        options = (
            deepcopy(dict(integrator_options)) if integrator_options is not None else {}
        )
        unsupported = {
            'abstolv',
            'augmented_options',
            'nfwd',
            'nadj',
            't0',
            'tf',
            'grid',
            'output_t0',
        }
        if unsupported.intersection(options):
            raise ValueError(
                "Simulator.compile does not support these integrator options: %s"
                % ', '.join(sorted(unsupported.intersection(options)))
            )
        for name in ('abstol', 'reltol'):
            if name in options and (
                not isinstance(options[name], Real)
                or not np.isfinite(options[name])
                or options[name] <= 0
            ):
                raise ValueError("%s must be a positive finite scalar" % name)
        if options.get('quad_err_con', True) is not True:
            raise ValueError(
                "Simulator.compile requires quad_err_con=True for adjoint accuracy"
            )
        options['quad_err_con'] = True
        if self._alglist:
            for name, required in (
                ('calc_ic', True),
                ('calc_icB', True),
                ('suppress_algebraic', False),
            ):
                if options.get(name, required) is not required:
                    raise ValueError(
                        "DAE compilation requires %s=%s" % (name, required)
                    )
                options[name] = required
            options.setdefault('abstol', 1e-8)
        if not check_min_version(casadi, '3.8.2'):
            if options.get('max_num_dir', 1) != 1:
                raise ValueError(
                    "Simulator.compile requires max_num_dir=1 on CasADi 3.8.1"
                )
            options['max_num_dir'] = 1

        time = casadi.SX.sym('t')
        states, ode = self._get_casadi_ode(time, substitutions)
        dae = {'x': states, 'p': symbols, 'ode': ode, 't': time}
        equations = ode
        if self._alglist:
            dae['z'] = casadi.vertcat(
                *(self._templatemap[v] for v in algebraic_variables)
            )
            dae['alg'] = casadi.vertcat(
                *(convert_pyomo2casadi(v, time, substitutions) for v in self._alglist)
            )
            equations = casadi.vertcat(ode, dae['alg'])
        for index, parameter in enumerate(parameters):
            if not casadi.depends_on(equations, symbols[index]):
                raise ValueError(
                    "Selected parameter %s is unused in the equations" % parameter.name
                )
        if self._alglist:
            return self._compile_casadi_dae(dae, times, options)
        x0 = casadi.MX.sym('x0', len(self._diffvars))
        p = casadi.MX.sym('p', len(parameters))
        if controls:
            control_symbols = casadi.vertcat(
                *(self._templatemap[v] for v in control_variables)
            )
            for index, control in enumerate(controls):
                if not casadi.depends_on(ode, control_symbols[index]):
                    raise ValueError(
                        "Selected control %s is unused in the ODE" % control.name
                    )
            u = casadi.MX.sym('u', len(controls), len(control_times) - 1)
            grid = np.union1d(times, control_times[control_times <= times[-1]])
            profile = x0
            intervals = len(grid) - 1
            if intervals:
                # Normalize every observation/switch interval, retaining physical time.
                start, duration = casadi.SX.sym('start'), casadi.SX.sym('duration')
                dae['p'] = casadi.vertcat(symbols, control_symbols, start, duration)
                dae['ode'] = duration * casadi.substitute(
                    ode, time, start + duration * time
                )
                integrate = casadi.integrator('F', integrator, dae, 0.0, 1.0, options)
                phases = np.searchsorted(control_times, grid[:-1], side='right') - 1
                inputs = casadi.vertcat(
                    casadi.repmat(p, 1, intervals),
                    u[:, phases.tolist()],
                    casadi.DM(grid[:-1]).T,
                    casadi.DM(np.diff(grid)).T,
                )
                profile = casadi.horzcat(
                    x0,
                    integrate.mapaccum('intervals', intervals)(x0=x0, p=inputs)['xf'],
                )
            return casadi.Function(
                'simulator',
                [x0, p, u],
                [profile[:, np.searchsorted(grid, times).tolist()]],
                ['x0', 'p', 'u'],
                ['xf'],
            )
        integrate = casadi.integrator(
            'F', integrator, dae, self._contset.first(), times.tolist(), options
        )
        return casadi.Function(
            'simulator', [x0, p], [integrate(x0=x0, p=p)['xf']], ['x0', 'p'], ['xf']
        )

    def _compile_casadi_dae(self, dae, times, options):
        """Compose implicit IDAS intervals and differentiable consistent outputs."""
        time, x, z, p = (dae[k] for k in ('t', 'x', 'z', 'p'))
        packed = casadi.vertcat(time, x, p)
        root = casadi.rootfinder(
            'root',
            'newton',
            {'x': z, 'p': packed, 'g': dae['alg']},
            {
                'abstol': options['abstol'],
                'abstolStep': 0,
                'max_iter': 100,
                'regularity_check': True,
                'error_on_fail': True,
            },
        )
        guess, data = casadi.MX.sym('guess', z.numel()), casadi.MX.sym(
            'data', packed.numel()
        )
        solution = root(guess, data)
        check = casadi.Function(
            'check',
            [z, packed],
            # QR avoids factorial symbolic expansion for dense algebraic Jacobians.
            [casadi.det(casadi.jacobian(dae['alg'], z), 'symbolicqr'), dae['alg']],
        )
        determinant, residual = check(solution, data)
        valid = casadi.logic_and(
            casadi.logic_and(
                casadi.sum1(casadi.fabs(solution)) < casadi.inf,
                casadi.logic_and(
                    casadi.fabs(determinant) > 0, casadi.fabs(determinant) < casadi.inf
                ),
            ),
            casadi.logic_and(
                casadi.sum1(casadi.fabs(residual)) < casadi.inf,
                casadi.mmax(casadi.fabs(residual)) <= options['abstol'],
            ),
        )
        consistent = casadi.Function(
            'consistent',
            [guess, data],
            [
                solution.attachAssert(
                    valid,
                    'Algebraic root must be finite, locally regular and satisfy its residual',
                )
            ],
        )
        x0, z0, parameters = (
            casadi.MX.sym('x0', x.numel()),
            casadi.MX.sym('z0', z.numel()),
            casadi.MX.sym('p', p.numel()),
        )
        initial_time = self._contset.first()
        initial_z = consistent(z0, casadi.vertcat(initial_time, x0, parameters))
        grid = np.union1d([initial_time], times)
        intervals = len(grid) - 1
        xf, zf = x0, initial_z
        if intervals:
            start, duration = casadi.SX.sym('start'), casadi.SX.sym('duration')
            phase_parameters = casadi.MX.sym('p', p.numel() + 2)
            phase_dae = dict(dae)
            phase_dae['p'] = casadi.vertcat(p, start, duration)
            phase_dae['ode'] = duration * casadi.substitute(
                dae['ode'], time, start + duration * time
            )
            phase_dae['alg'] = casadi.substitute(
                dae['alg'], time, start + duration * time
            )
            # IDAS explicit-time adjoints can fail even on a polynomial ODE.
            # An internal physical clock makes the same DAE autonomous.
            clocked = casadi.depends_on(casadi.vertcat(dae['ode'], dae['alg']), time)
            phase_x0 = x0
            if clocked:
                clock = casadi.SX.sym('clock')
                phase_dae['x'] = casadi.vertcat(x, clock)
                phase_dae['ode'] = casadi.vertcat(
                    duration * casadi.substitute(dae['ode'], time, clock), duration
                )
                phase_dae['alg'] = casadi.substitute(dae['alg'], time, clock)
                phase_x0 = casadi.vertcat(x0, phase_parameters[-2])
            integrate = casadi.integrator('F', 'idas', phase_dae, 0.0, 1.0, options)
            phase_z = consistent(
                z0, casadi.vertcat(phase_parameters[-2], x0, phase_parameters[:-2])
            )
            out = integrate(x0=phase_x0, z0=phase_z, p=phase_parameters)
            phase_xf = out['xf'][: x.numel()]
            final_z = consistent(
                out['zf'],
                casadi.vertcat(
                    phase_parameters[-2] + phase_parameters[-1],
                    phase_xf,
                    phase_parameters[:-2],
                ),
            )
            phase = casadi.Function(
                'phase',
                [x0, z0, phase_parameters],
                [phase_xf, final_z],
                ['x0', 'z0', 'p'],
                ['xf', 'zf'],
            )
            inputs = casadi.vertcat(
                casadi.repmat(parameters, 1, intervals),
                casadi.DM(grid[:-1]).T,
                casadi.DM(np.diff(grid)).T,
            )
            out = phase.mapaccum('intervals', intervals, ['x0', 'z0'], ['xf', 'zf'])(
                x0=x0, z0=initial_z, p=inputs
            )
            xf, zf = casadi.horzcat(x0, out['xf']), casadi.horzcat(initial_z, out['zf'])
        columns = np.searchsorted(grid, times).tolist()
        return casadi.Function(
            'simulator',
            [x0, parameters, z0],
            [xf[:, columns], zf[:, columns]],
            ['x0', 'p', 'z0'],
            ['xf', 'zf'],
        )

    def simulate(
        self,
        numpoints=None,
        tstep=None,
        integrator=None,
        varying_inputs=None,
        initcon=None,
        integrator_options=None,
    ):
        """
        Simulate the model. Integrator-specific options may be specified as
        keyword arguments and will be passed on to the integrator.

        Parameters
        ----------
        numpoints : int
            The number of points for the profiles returned by the simulator.
            Must be at least 2. Default is 100.

        tstep : int or float
            The time step to use in the profiles returned by the simulator.
            This is not the time step used internally by the integrators.
            This is an optional parameter that may be specified in place of
            'numpoints'. The final time is always included; the last interval
            may be shorter than tstep.

        integrator : string
            The string name of the integrator to use for simulation. The
            default is 'lsoda' when using Scipy and 'idas' when using CasADi

        varying_inputs : ``pyomo.environ.Suffix``
            A :py:class:`Suffix<pyomo.environ.Suffix>` object containing the
            piecewise constant profiles to be used for certain time-varying
            algebraic variables.

        initcon : list of floats
            The initial conditions for the the differential variables. This
            is an optional argument. If not specified then the simulator
            will use the current value of the differential variables at the
            lower bound of the ContinuousSet for the initial condition.
            Algebraic variable values at that point are used as guesses for
            CasADi's consistent initialization; unset guesses default to zero.

        integrator_options : dict
            Dictionary containing options that should be passed to the
            integrator. See the documentation for a specific integrator for a
            list of valid options. The caller's dictionary is not modified.

        Returns
        -------
        numpy array, numpy array
            The first return value is a 1D array of time points corresponding
            to the second return value which is a 2D array of the profiles for
            the simulated differential and algebraic variables.
        """

        if not numpy_available:
            raise ValueError(
                "The numpy module is not available. Cannot simulate the model."
            )

        if integrator_options is None:
            integrator_options = {}
        else:
            integrator_options = deepcopy(integrator_options)

        if self._intpackage == 'scipy':
            # Specify the scipy integrator to use for simulation
            valid_integrators = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
            if integrator is None:
                integrator = 'lsoda'
            elif integrator == 'odeint':
                integrator = 'lsoda'
        else:
            # Specify the casadi integrator to use for simulation.
            # Only a subset of these integrators may be used for
            # DAE simulation. We defer this check to CasADi.
            valid_integrators = ['cvodes', 'idas', 'collocation', 'rk']
            if integrator is None:
                integrator = 'idas'

        if integrator not in valid_integrators:
            raise DAE_Error(
                "Unrecognized %s integrator \'%s\'. Please select"
                " an integrator from %s"
                % (self._intpackage, integrator, valid_integrators)
            )

        # Set the time step or the number of points for the lists
        # returned by the integrator
        if tstep is not None and numpoints is not None:
            raise ValueError(
                "Cannot specify both the step size and the number of "
                "points for the simulator"
            )
        if tstep is None and numpoints is None:
            # Use 100 points by default
            numpoints = 100

        if tstep is None:
            if (
                isinstance(numpoints, bool)
                or not isinstance(numpoints, Integral)
                or numpoints < 2
            ):
                raise ValueError(
                    "numpoints must be an integer greater than or equal to 2"
                )
            tsim = np.linspace(
                self._contset.first(), self._contset.last(), num=numpoints
            )

            # Consider adding an option for log spaced time points. Can be
            # important for simulating stiff systems.
            # tsim = np.logspace(-4,6, num=100)
            # np.log10(self._contset.first()),np.log10(
            # self._contset.last()),num=1000, endpoint=True)

        else:
            if (
                isinstance(tstep, bool)
                or not isinstance(tstep, Real)
                or not np.isfinite(tstep)
                or tstep <= 0
            ):
                raise ValueError("tstep must be a positive finite number")
            if tstep > self._contset.last() - self._contset.first():
                raise ValueError("tstep must not exceed the span of the ContinuousSet")
            tsim = np.arange(self._contset.first(), self._contset.last(), tstep)
            tsim = np.append(tsim[tsim < self._contset.last()], self._contset.last())

        switchpts = []
        siminputvars = {}
        simalgvars = []
        if varying_inputs is not None:
            if type(varying_inputs) is not Suffix:
                raise TypeError(
                    "Varying input values must be specified using a "
                    "Suffix. Please refer to the simulator documentation."
                )

            for alg in self._algvars:
                if alg._base in varying_inputs:
                    profile = varying_inputs[alg._base]
                    if not isinstance(profile, Mapping):
                        raise TypeError("Varying input profiles must be dictionaries")
                    if self._contset.first() not in profile:
                        raise ValueError(
                            "Varying input profiles must include the initial time"
                        )
                    if any(
                        not isinstance(t, Real) or not np.isfinite(t) for t in profile
                    ):
                        raise ValueError(
                            "Varying input switching times must be finite numbers"
                        )
                    if any(
                        not isinstance(v, Real) or not np.isfinite(v)
                        for v in profile.values()
                    ):
                        raise ValueError("Varying input values must be finite numbers")
                    # Find all the switching points
                    switchpts += profile.keys()
                    # Add to dictionary of siminputvars
                    siminputvars[alg._base] = alg
                else:
                    simalgvars.append(alg)

            if self._intpackage == 'scipy' and len(simalgvars) != 0:
                raise DAE_Error(
                    "When simulating with Scipy you must "
                    "provide values for all parameters "
                    "and algebraic variables that are indexed "
                    "by the ContinuoutSet using the "
                    "'varying_inputs' keyword argument. "
                    "Please refer to the simulator documentation "
                    "for more information."
                )

            # Get the set of unique points
            switchpts = list(set(switchpts))
            switchpts.sort()

            # Make sure all the switchpts are within the bounds of
            # the ContinuousSet
            if switchpts and (
                switchpts[0] < self._contset.first()
                or switchpts[-1] > self._contset.last()
            ):
                raise ValueError(
                    "Found a switching point for one or more of "
                    "the time-varying inputs that is not within "
                    "the bounds of the ContinuousSet."
                )

            # Update tsim to include input switching points
            # This numpy function returns the unique, sorted points
            # Keep exact switch times when the output grid differs only by roundoff.
            for switch in switchpts:
                tolerance = 4 * np.finfo(float).eps * max(1, abs(switch))
                tsim[np.abs(tsim - switch) <= tolerance] = switch
            tsim = np.union1d(tsim, switchpts)
        else:
            simalgvars = self._algvars

        # Check if initial conditions were provided, otherwise obtain
        # them from the current variable values
        if initcon is not None:
            if len(initcon) > len(self._diffvars):
                raise ValueError(
                    "Too many initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    % len(self._diffvars)
                )
            if len(initcon) < len(self._diffvars):
                raise ValueError(
                    "Too few initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    % len(self._diffvars)
                )
        else:
            initcon = self._get_initial_conditions(self._diffvars)

        initcon = np.asarray(initcon, dtype=float)
        if initcon.ndim != 1 or not np.isfinite(initcon).all():
            raise ValueError(
                "Initial conditions must be a one-dimensional sequence of finite numbers"
            )

        # Call the integrator
        previous_inputs, previous_algebraics = self._siminputvars, self._simalgvars
        self._siminputvars, self._simalgvars = siminputvars, simalgvars
        try:
            if self._intpackage == 'scipy':
                if not scipy_available:
                    raise ValueError(
                        "The scipy module is not available. Cannot simulate the model."
                    )
                tsim, profile = self._simulate_with_scipy(
                    initcon,
                    tsim,
                    switchpts,
                    varying_inputs,
                    integrator,
                    integrator_options,
                )
            else:
                if len(switchpts) != 0:
                    tsim, profile = self._simulate_with_casadi_with_inputs(
                        initcon, tsim, varying_inputs, integrator, integrator_options
                    )
                else:
                    tsim, profile = self._simulate_with_casadi_no_inputs(
                        initcon, tsim, integrator, integrator_options
                    )
        except BaseException:
            self._siminputvars, self._simalgvars = previous_inputs, previous_algebraics
            raise

        self._tsim = tsim
        self._simsolution = profile

        return [tsim, profile]

    def _get_initial_conditions(self, variables, default=None):
        """Read state values at the initial time, optionally defaulting unset guesses."""
        initial = []
        for variable in variables:
            index = tuple(
                self._contset.first() if type(i) is IndexTemplate else i
                for i in variable._args
            )
            val = value(variable._base[index], exception=default is None)
            initial.append(default if val is None else val)
        return initial

    def _simulate_with_scipy(
        self, initcon, tsim, switchpts, varying_inputs, integrator, integrator_options
    ):
        scipyint = scipy.integrate.ode(self._rhsfun).set_integrator(
            integrator, **integrator_options
        )
        scipyint.set_initial_value(initcon, tsim[0])

        profile = np.array(initcon)
        i = 1
        while scipyint.successful() and scipyint.t < tsim[-1]:
            # check if tsim[i-1] is a switching time and update value
            if tsim[i - 1] in switchpts:
                for v in self._siminputvars.keys():
                    if tsim[i - 1] in varying_inputs[v]:
                        p = self._templatemap[self._siminputvars[v]]
                        p.set_value(varying_inputs[v][tsim[i - 1]])
                # Discard multistep history and cached RHS values at a discontinuity.
                scipyint.set_initial_value(scipyint.y, scipyint.t)

            profilestep = scipyint.integrate(tsim[i])
            profile = np.vstack([profile, profilestep])
            i += 1

        if not scipyint.successful():
            raise DAE_Error(
                "The Scipy integrator %s did not terminate "
                "successfully." % integrator
            )
        return [tsim, profile]

    def _get_casadi_ode(self, time, substitutions=None):
        """Assemble the ordered differential states and converted ODE expressions."""
        states = casadi.vertcat(*(self._templatemap[i] for i in self._diffvars))
        ode = casadi.vertcat(
            *(
                convert_pyomo2casadi(self._rhsdict[i], time, substitutions)
                for i in self._derivlist
            )
        )
        return states, ode

    def _simulate_with_casadi_no_inputs(
        self, initcon, tsim, integrator, integrator_options
    ):
        # Old way (10 times faster, but can't incorporate time
        # varying parameters/controls)
        time = casadi.SX.sym('t')
        xall, odeall = self._get_casadi_ode(time)
        dae = {'x': xall, 'ode': odeall, 't': time}

        if len(self._algvars) != 0:
            zalltemp = [self._templatemap[i] for i in self._simalgvars]
            zall = casadi.vertcat(*zalltemp)

            algalltemp = [convert_pyomo2casadi(i, time) for i in self._alglist]
            algall = casadi.vertcat(*algalltemp)
            dae['z'] = zall
            dae['alg'] = algall

        integrator_options['grid'] = tsim
        integrator_options['output_t0'] = True
        F = casadi.integrator('F', integrator, dae, integrator_options)
        algcon = self._get_initial_conditions(self._simalgvars, default=0)
        if not np.isfinite(algcon).all():
            raise ValueError("Algebraic initial guesses must be finite numbers")
        sol = F(x0=initcon, z0=algcon)
        profile = sol['xf'].full().T

        if len(self._algvars) != 0:
            algprofile = sol['zf'].full().T
            profile = np.concatenate((profile, algprofile), axis=1)

        return [tsim, profile]

    def _simulate_with_casadi_with_inputs(
        self, initcon, tsim, varying_inputs, integrator, integrator_options
    ):
        duration = casadi.SX.sym('duration')
        start = casadi.SX.sym('start')
        local_time = casadi.SX.sym('t')
        time = start + duration * local_time

        xall, odeall = self._get_casadi_ode(time)
        odeall = duration * odeall

        # Time-varying inputs
        ptemp = [self._templatemap[i] for i in self._siminputvars.values()]
        pall = casadi.vertcat(duration, start, *ptemp)

        dae = {'x': xall, 'p': pall, 'ode': odeall, 't': local_time}

        if len(self._algvars) != 0:
            zalltemp = [self._templatemap[i] for i in self._simalgvars]
            zall = casadi.vertcat(*zalltemp)
            algalltemp = [convert_pyomo2casadi(i, time) for i in self._alglist]
            algall = casadi.vertcat(*algalltemp)
            dae['z'] = zall
            dae['alg'] = algall

        integrator_options['tf'] = 1.0
        F = casadi.integrator('F', integrator, dae, integrator_options)
        N = len(tsim)

        # This approach removes the time scaling from tsim so must
        # create an array with the time step between consecutive
        # time points
        tsimtemp = np.hstack([0, tsim[1:] - tsim[0:-1]])
        tsimtemp = tsimtemp.reshape(1, -1)

        starts = np.hstack([tsim[0], tsim[:-1]]).reshape(1, -1)
        palltemp = [casadi.DM(tsimtemp), casadi.DM(starts)]

        # Need a similar np array for each time-varying input
        for p in self._siminputvars.keys():
            profile = varying_inputs[p]
            tswitch = list(profile.keys())
            tswitch.sort()
            tidx = [tsim.searchsorted(i) for i in tswitch] + [len(tsim) - 1]
            ptemp = [profile[tsim[0]]] + [
                casadi.repmat(profile[tswitch[i]], 1, tidx[i + 1] - tidx[i])
                for i in range(len(tswitch))
            ]
            temp = casadi.horzcat(*ptemp)
            palltemp.append(temp)

        I = F.mapaccum('simulator', N)
        algcon = self._get_initial_conditions(self._simalgvars, default=0)
        if not np.isfinite(algcon).all():
            raise ValueError("Algebraic initial guesses must be finite numbers")
        sol = I(x0=initcon, z0=algcon, p=casadi.vertcat(*palltemp))
        profile = sol['xf'].full().T

        if len(self._algvars) != 0:
            algprofile = sol['zf'].full().T
            if self._simalgvars:
                # At a switch, return algebraic states consistent with the new inputs.
                switches = sorted(
                    {t for p in self._siminputvars for t in varying_inputs[p]}
                )
                for switch in switches:
                    if switch == tsim[0]:
                        continue
                    index = tsim.searchsorted(switch)
                    inputs = [0, switch] + [
                        varying_inputs[p][
                            max(t for t in varying_inputs[p] if t <= switch)
                        ]
                        for p in self._siminputvars
                    ]
                    consistent = F(
                        x0=sol['xf'][:, index], z0=sol['zf'][:, index], p=inputs
                    )
                    algprofile[index, :] = consistent['zf'].full().ravel()
            profile = np.concatenate((profile, algprofile), axis=1)

        return [tsim, profile]

    def initialize_model(self):
        """
        This function will initialize the model using the profile obtained
        from simulating the dynamic model.
        """
        if self._tsim is None:
            raise DAE_Error("Tried to initialize the model without simulating it first")

        tvals = list(self._contset)

        # Build list of state and algebraic variables
        # that can be initialized
        initvars = self._diffvars + self._simalgvars

        for idx, v in enumerate(initvars):
            for idx2, i in enumerate(v._args):
                if type(i) is IndexTemplate:
                    break
            valinit = np.interp(tvals, self._tsim, self._simsolution[:, idx])
            for i, t in enumerate(tvals):
                vidx = tuple(v._args[0:idx2]) + (t,) + tuple(v._args[idx2 + 1 :])
                v._base[vidx] = valinit[i]
