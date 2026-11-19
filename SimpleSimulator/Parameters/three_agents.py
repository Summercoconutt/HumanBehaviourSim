"""Three simulator agents that share the three-class model.

The historical filename ``3LCAgentParameters.py`` means three agents, not a
second model.
"""

from Parameters.LinearClassifierParameters import LinearClassifierParameters


class SimpleParameters(LinearClassifierParameters):
    """Three shared-model agents.

    The class name is historical. New simulator runs should use
    LinearClassifierParameters.
    """
