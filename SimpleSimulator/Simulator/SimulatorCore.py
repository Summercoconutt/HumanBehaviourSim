from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias

import simpy


ResourceState: TypeAlias = dict[str, object]


@dataclass
class State:
    agent_resources: dict[str, ResourceState] = field(default_factory=dict)
    context: ResourceState = field(default_factory=dict)
    step: int = 0


@dataclass
class Decision:
    text_body: str
    decision: list[object]


@dataclass
class AgentDecision:
    agent_id: str
    decision: Decision
    choice: object
    resource_updates: dict[str, ResourceState] = field(default_factory=dict)
    context_updates: ResourceState = field(default_factory=dict)


class AbstractAgent(ABC):
    agent_id: str

    @abstractmethod
    def decide(self, decision: Decision, state: State) -> AgentDecision:
        pass


@dataclass
class AgentConfiguration:
    agent_type: type[AbstractAgent]
    count: int


class AbstractScenario(ABC):
    @property
    @abstractmethod
    def initial_resources(self) -> dict[type[AbstractAgent], ResourceState]:
        pass

    @property
    @abstractmethod
    def context(self) -> ResourceState:
        pass

    @property
    @abstractmethod
    def decisions(self) -> list[Decision]:
        pass

    @abstractmethod
    def is_complete(self, state: State) -> bool:
        pass


class AbstractParameters(ABC):
    @property
    @abstractmethod
    def agent_configurations(self) -> list[AgentConfiguration]:
        pass


@dataclass
class SimulationResult:
    final_state: State
    decision_log: list[AgentDecision] = field(default_factory=list)


class SimulatorCore:
    def __init__(self) -> None:
        self.environment = simpy.Environment()
        self.agents: dict[str, AbstractAgent] = {}
        self.parameters: AbstractParameters | None = None
        self.scenario: AbstractScenario | None = None
        self.state = State()
        self.decisions: list[Decision] = []
        self.decision_log: list[AgentDecision] = []

    def __call__(
        self,
        scenario: AbstractScenario,
        parameters: AbstractParameters,
    ) -> SimulationResult:
        # 每次调用从零开始，避免把上一次模拟的时间和记录带进来。
        self.environment = simpy.Environment()
        self.decision_log = []
        self.load_parameters(parameters)
        self.load_agents(parameters)
        self.load_scenario(scenario)
        return self.run_simulation()

    def load_agents(self, parameters: AbstractParameters) -> None:
        self.agents = {}
        for configuration in parameters.agent_configurations:
            if configuration.count < 0:
                raise ValueError("Agent count cannot be negative.")

            for _ in range(configuration.count):
                agent = configuration.agent_type()
                # 每个实例获得自己的 ID，即使它们属于同一种 Agent。
                agent.agent_id = (
                    f"{configuration.agent_type.__name__}_{len(self.agents) + 1}"
                )
                self.agents[agent.agent_id] = agent

    def load_scenario(self, scenario: AbstractScenario) -> None:
        self.scenario = scenario
        self.decisions = deepcopy(scenario.decisions)
        initial_resources = scenario.initial_resources
        self.state = State(
            agent_resources={
                agent_id: deepcopy(initial_resources.get(type(agent), {}))
                for agent_id, agent in self.agents.items()
            },
            context=deepcopy(scenario.context),
            step=0,
        )

    def load_parameters(self, parameters: AbstractParameters) -> None:
        self.parameters = parameters

    def run_simulation(self) -> SimulationResult:
        if self.scenario is None:
            raise RuntimeError("Load a scenario before running the simulation.")

        while not self.scenario.is_complete(self.state):
            if self.state.step >= len(self.decisions):
                raise RuntimeError(
                    "The scenario ran out of decisions before its completion "
                    "condition was met. Check the decision count and is_complete()."
                )

            # step 从 0 开始，所以第 0 步取出列表中的第一道题。
            decision = self.decisions[self.state.step]
            answers = self.calculate_step(self.agents, self.state, decision)

            # 全部 Agent 回答后，再应用本轮的状态更新。
            # update 表示设置字段的新值；同一字段多次更新时，后写入的值生效。
            for answer in answers:
                for agent_id, updates in answer.resource_updates.items():
                    self.state.agent_resources[agent_id].update(deepcopy(updates))
                self.state.context.update(deepcopy(answer.context_updates))

            self.decision_log.extend(answers)
            self.state.step += 1

            # 每轮推进 1 个虚拟时间单位，不会让电脑实际等待 1 秒。
            self.environment.run(until=self.environment.timeout(1))

        return SimulationResult(
            final_state=deepcopy(self.state),
            decision_log=deepcopy(self.decision_log),
        )

    def calculate_step(
        self,
        agents: dict[str, AbstractAgent],
        state: State,
        decision: Decision,
    ) -> list[AgentDecision]:
        answers = []
        for agent_id, agent in agents.items():
            # 独立副本保证各 Agent 看到同一轮开始时的状态。
            answer = agent.decide(decision, deepcopy(state))
            if answer.agent_id != agent_id:
                raise ValueError("The returned decision has the wrong agent ID.")
            if answer.choice not in decision.decision:
                raise ValueError("The agent chose an option that is not available.")
            answers.append(answer)
        return answers

    def show_result(self, result: SimulationResult) -> None:
        print(f"Completed steps: {result.final_state.step}")
        print(f"Agent count: {len(result.final_state.agent_resources)}")
        print(f"Decision records: {len(result.decision_log)}")
        for answer in result.decision_log:
            print(
                f"{answer.agent_id} | {answer.decision.text_body} "
                f"| choice={answer.choice}"
            )
