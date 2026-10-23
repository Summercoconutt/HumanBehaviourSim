from Parameters.SimpleParameters1 import SimpleParameters
from Scenarios.SimpleScenario1 import SimpleScenario1
from Simulator.SimulatorCore import SimulatorCore


def main() -> None:
    # 1. 创建场景：五道选数字的题目。
    scenario = SimpleScenario1()

    # 2. 创建参数：使用 SimpleAgent1。
    parameters = SimpleParameters()

    # 3. 创建模拟器；这一步还没有开始回答题目。
    simulator = SimulatorCore()

    # 4. 调用模拟器的 __call__ 方法，准备数据并运行全部轮次。
    result = simulator(scenario, parameters)

    # 5. 展示最终状态的摘要和选择记录。
    simulator.show_result(result)


if __name__ == "__main__":
    main()
