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

import logging

from pyomo.core.base import Transformation, TransformationFactory, Block, SortComponents
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct

logger = logging.getLogger('pyomo.core')


#
# This transformation reworks each Complementarity block to
# setup a standard form.
#
@TransformationFactory.register(
    'mpec.standard_form', doc="Standard reformulation of complementarity condition"
)
class MPEC3_Transformation(Transformation):
    def __init__(self):
        super(MPEC3_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        #
        # Iterate over the model finding Complementarity components
        #
        for complementarity in instance.component_objects(
            Complementarity,
            active=True,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        ):
            block = complementarity.parent_block()
            for index in sorted(complementarity.keys()):
                _data = complementarity[index]
                if not _data.active:
                    continue
                _data.to_standard_form()
                #
            block.reclassify_component_type(complementarity, Block)
