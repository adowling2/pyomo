#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

###
# create the original unscaled model
###
model = pyo.ConcreteModel()
model.x = pyo.Var([1, 2, 3], bounds=(-10, 10), initialize=5.0)
model.z = pyo.Var(bounds=(10, 20))
model.obj = pyo.Objective(expr=model.z + model.x[1])

# demonstrate scaling of duals as well
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)


def con_rule(m, i):
    if i == 1:
        return m.x[1] + 2 * m.x[2] + 1 * m.x[3] == 4.0
    if i == 2:
        return m.x[1] + 2 * m.x[2] + 2 * m.x[3] == 5.0
    if i == 3:
        return m.x[1] + 3.0 * m.x[2] + 1 * m.x[3] == 5.0


model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
model.zcon = pyo.Constraint(expr=model.z >= model.x[2])

###
# set the scaling parameters
###
model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
model.scaling_factor[model.obj] = 2.0
model.scaling_factor[model.x] = 0.5
model.scaling_factor[model.z] = -10.0
model.scaling_factor[model.con[1]] = 0.5
model.scaling_factor[model.con[2]] = 2.0
model.scaling_factor[model.con[3]] = -5.0
model.scaling_factor[model.zcon] = -3.0

###
# build and solve the scaled model
###
scaled_model = pyo.TransformationFactory('core.scale_model').create_using(model)
pyo.SolverFactory('glpk').solve(scaled_model)


###
# propagate the solution back to the original model
###
pyo.TransformationFactory('core.scale_model').propagate_solution(scaled_model, model)

# print the scaled model
scaled_model.pprint()

# print the solution on the original model after backmapping
model.pprint()

compare_solutions = True
if compare_solutions:
    # compare the solution of the original model with a clone of the
    # original that has a backmapped solution from the scaled model

    # solve the original (unscaled) model
    original_model = model.clone()
    pyo.SolverFactory('glpk').solve(original_model)

    # create and solve the scaled model
    scaling_tx = pyo.TransformationFactory('core.scale_model')
    scaled_model = scaling_tx.create_using(model)
    pyo.SolverFactory('glpk').solve(scaled_model)

    # propagate the solution from the scaled model back to a clone of the original model
    backmapped_unscaled_model = model.clone()
    scaling_tx.propagate_solution(scaled_model, backmapped_unscaled_model)

    # compare the variable values
    print('\n\n')
    print('%s\t%12s           %18s' % ('Var', 'Orig.', 'Scaled -> Backmapped'))
    print('=====================================================')
    for v in original_model.component_data_objects(ctype=pyo.Var, descend_into=True):
        cuid = pyo.ComponentUID(v)
        bv = cuid.find_component_on(backmapped_unscaled_model)
        print('%s\t%.16f\t%.16f' % (v.local_name, pyo.value(v), pyo.value(bv)))
    print('=====================================================')
