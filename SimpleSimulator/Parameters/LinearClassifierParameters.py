from Models.three_class import LCAgent, train_three_class_model
from Simulator.SimulatorCore import AbstractParameters, AgentConfiguration


class LinearClassifierParameters(AbstractParameters):
    """Several identical agents that share one trained three-class model.

    The default count remains three. Sharing the model does not create
    different consumer preferences.
    """

    def __init__(self, count: int = 3) -> None:
        if count < 1:
            raise ValueError("At least one classifier agent is required.")
        self.count = count
        self.model = train_three_class_model()

    @property
    def agent_configurations(self) -> list[AgentConfiguration]:
        return [
            AgentConfiguration(
                agent_type=LCAgent,
                count=self.count,
                init_kwargs={"model": self.model},
            )
        ]


class SingleLinearClassifierParameters(LinearClassifierParameters):
    """One-agent run. It uses the same three-class training procedure."""

    def __init__(self) -> None:
        super().__init__(count=1)
