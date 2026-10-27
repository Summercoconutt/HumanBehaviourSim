# 最小模拟器：跟着执行顺序把骨架填完整

本次目标：让原来的一个 SimpleAgent1 回答五道题，完成初始化、逐轮运行、记录和结果返回。Agent、Scenario、Parameters 三个示例文件保持原样。主要实现都在 Simulator/SimulatorCore.py，另外新增 main.py 作为入口。

## 1. 先运行，确认完整流程已经接上

需要 Python 3.10 或更高版本。在解压后的 SimpleSimulator 文件夹打开终端，运行：

```bash
python --version
python -m pip install -r requirements.txt
python main.py
python main.py SimpleScenario1 SimpleParameters
```

命令行的两个可选参数依次是场景类名和参数类名；省略时使用 `SimpleScenario1` 和 `SimpleParameters`。新增场景或参数类后，需要在 `main.py` 中将名称加入对应的映射表。

成功输出：

```text
Completed steps: 5
Agent count: 1
Decision records: 5
SimpleAgent1_1 | Round 1: Choose a number. | choice=1
SimpleAgent1_1 | Round 2: Choose a number. | choice=1
SimpleAgent1_1 | Round 3: Choose a number. | choice=1
SimpleAgent1_1 | Round 4: Choose a number. | choice=1
SimpleAgent1_1 | Round 5: Choose a number. | choice=1
```

五个 1 就是当前规则的预期结果。验收重点是：完成五步，存在一个不同身份的 Agent，返回五条有效选择记录。

## 2. 文件为什么要放进这些文件夹？

原代码中的 `from Agents.SimpleAgent1 import SimpleAgent1` 会寻找 Agents 文件夹内的 SimpleAgent1.py。按原来的导入路径整理文件，就不需要改动这三个示例类。

| 相对于项目根目录的路径 | 内容 |
| --- | --- |
| main.py | 启动程序 |
| Simulator/SimulatorCore.py | 数据结构、接口、补好的核心方法 |
| Agents/SimpleAgent1.py | 永远选择第一个选项 |
| Parameters/SimpleParameters1.py | 创建一个 SimpleAgent1 |
| Scenarios/SimpleScenario1.py | 五道题、初始条件、结束条件 |
| requirements.txt | SimPy 依赖 |
| tests/test_simulator.py | 验证完整流程的检查 |
| expected_output.txt | 示例运行的真实输出 |

每个包文件夹里的 __init__.py 用来标识 Python 包。

## 3. 从 main.py 开始读

```python
scenario = SimpleScenario1()
parameters = SimpleParameters()
simulator = SimulatorCore()
result = simulator(scenario, parameters)
simulator.show_result(result)
```

前面三行分别创建场景对象、参数对象和模拟器对象。第四行才开始完整模拟；第五行展示结果。

创建对象时，Python 会调用其 __init__ 方法。把对象当函数调用时，例如 simulator(scenario, parameters)，Python 会调用该对象的 __call__ 方法。

main.py 最后的 `if __name__ == "__main__":` 表示直接执行这个文件时调用 main()。

## 4. 第一个空：__init__ 准备存储位置

在已有的 SimPy 环境之外，新建几个属性：

| 属性 | 初始内容 | 后面存什么 |
| --- | --- | --- |
| self.agents | 空字典 | ID 到具体 Agent 对象的对应关系 |
| self.parameters | None | 参数对象 |
| self.scenario | None | 场景对象 |
| self.state | State() | 当前世界状态 |
| self.decisions | 空列表 | 五道待回答的题目 |
| self.decision_log | 空列表 | 已收集到的回答记录 |

self 表示当前这个模拟器对象。self.agents 与普通的局部变量不同：其他方法以后仍然能通过 self.agents 找到它。

## 5. 第二个空：__call__ 负责安排调用顺序

```python
self.environment = simpy.Environment()
self.decision_log = []
self.load_parameters(parameters)
self.load_agents(parameters)
self.load_scenario(scenario)
return self.run_simulation()
```

前两行重置上一次运行的时间和记录，确保同一个模拟器对象可以再次运行。

然后先读参数、创建具体 Agent，再初始化场景。这个顺序有原因：初始化每个人的资源之前，需要知道已经创建了哪些人。

最后 return 把 run_simulation() 得到的 SimulationResult 交回 main.py 的 result 变量。

## 6. 第三个空：load_parameters 保存参数

```python
self.parameters = parameters
```

这个方法目前只需要这一行。传来的 parameters 是参数对象；self.parameters 是模拟器中保存它的位置。

## 7. 第四个空：load_agents 真正创建 Agent

原来的配置是：

```python
AgentConfiguration(agent_type=SimpleAgent1, count=1)
```

它只是计划，还不是一个具体的 Agent。核心实现中：

```python
for configuration in parameters.agent_configurations:
    for _ in range(configuration.count):
        agent = configuration.agent_type()
```

外层循环读取每种 Agent 的配置。内层循环按 count 重复创建。下划线 _ 表示这里不需要使用循环编号。

由于 agent_type 存放的是 SimpleAgent1，`configuration.agent_type()` 就相当于调用 `SimpleAgent1()`。

接着，给创建出来的对象分配唯一 ID，再放进字典：

```python
agent.agent_id = (
    f"{configuration.agent_type.__name__}_{len(self.agents) + 1}"
)
self.agents[agent.agent_id] = agent
```

__name__ 取出类名，len(self.agents) 是目前已创建的人数。第一个人的 ID 是 SimpleAgent1_1。

最终字典中，键是 ID，值是那个具体的 Agent 对象。这样，多个人就能分别记录。

## 8. 第五个空：load_scenario 准备题目和初始状态

该方法保存场景、读取题目，并新建 State：

```python
self.scenario = scenario
self.decisions = deepcopy(scenario.decisions)
initial_resources = scenario.initial_resources
```

然后按每个实际 Agent 的类型，找到相应的初始资源。当前示例得到的是空字典。初始化后，状态相当于：

```python
State(
    agent_resources={"SimpleAgent1_1": {}},
    context={"rounds": 5},
    step=0,
)
```

源代码中的字典推导式是在对所有 Agent 做同样的初始化。`.items()` 每次取出一对 ID 和 Agent；`type(agent)` 取得 Agent 类型；`.get(type(agent), {})` 按类型查找资源，没有配置时使用空字典。

deepcopy 来自 Python 标准库 copy。它创建独立副本，避免不同 Agent 共享同一份可修改资源，也避免运行时修改场景的原始数据。

## 9. 第六个空：calculate_step 只收集一轮回答

最核心的三行：

```python
for agent_id, agent in agents.items():
    answer = agent.decide(decision, deepcopy(state))
    answers.append(answer)
```

遍历每个 Agent，把同一道题和本轮开始时的状态副本交给它，再把回答放进列表。

这个方法尚不应用资源更新。所有 Agent 回答之后，Core 才统一更新状态。因此这一版按顺序调用 Agent，同时保证它们都看到同一轮开始时的状态。

完整实现还检查返回的 ID 是否正确、选择是否在选项中。这些检查通过后，return answers 返回本轮回答列表。

当前一个 Agent，所以每轮这个列表只有一条记录。

## 10. 第七个空：run_simulation 反复运行各轮

主循环的意思是：只要场景还没有完成，就继续。

```python
while not self.scenario.is_complete(self.state):
```

初始 step=0，场景检查 0 >= 5 得到 False，not False 得到 True，所以进入循环。

每轮依次做六件事：

1. 根据 step 取出当前题目：self.decisions[self.state.step]。
2. 调用 calculate_step，收集所有 Agent 的回答。
3. 应用回答中携带的 resource_updates 和 context_updates。
4. 使用 self.decision_log.extend(answers)，把这一轮的各条回答加入总记录。
5. 执行 self.state.step += 1，标记完成一轮。
6. 推进 SimPy 的虚拟时钟一个单位。

`extend` 会把列表中的各条记录加入另一个列表；`append` 会把整个传入对象作为一项加入。这里要保存每一份回答，所以使用 extend。

状态更新在当前示例中都是空字典，所以没有预算等变化。引擎已经支持这些已有接口：以后 Agent 返回更新时，Core 会设置字段的新值。

约定：resource_updates 按 Agent ID 指定字段的新值，context_updates 指定公共字段的新值。更新使用字典 update，不做数值相加；同一轮同一字段被多次更新时，按 Agent 遍历顺序后写入的值生效。嵌套字段整体替换，不做递归合并。

```python
self.environment.run(until=self.environment.timeout(1))
```

这里推进的是虚拟时间，不是让电脑等待一秒。当前实现没有对 Agent 做并发事件调度。

当 step 变成 5，is_complete 返回 True，循环结束。随后将最终状态和全部记录装成 SimulationResult 返回。

运行中如果题目已用完但结束条件还没满足，程序会报出明确错误，便于发现题目数量与停止条件不一致的问题。

## 11. 第八个空：show_result 展示已经算好的数据

这个方法输出最终步数、Agent 人数和记录数量，再逐条打印回答。它不会重新模拟，也不会计算有实际意义的预测准确率。

## 12. 检查是否跑通

可运行基础检查：

```bash
python -m unittest discover -s tests -v
```

检查内容包括：原始五轮示例、多 Agent 的独立 ID、状态副本和更新、重复运行的隔离、题目不足时的错误信息。

如果想做第一次小修改，只改 Parameters/SimpleParameters1.py 的 count：

```python
return [AgentConfiguration(agent_type=SimpleAgent1, count=3)]
```

重新执行 python main.py。预期完成步数仍是 5，Agent 人数变成 3，选择记录变成 15。五轮乘以三个人，所以是十五条回答。

## 13. 这次先做到哪里

保留原来的数据结构与抽象接口；抽象方法里的 pass 应继续保留。已实现的是 SimulatorCore 内的具体操作方法。

当前 Agent 仍然永远选择第一个选项；没有加入 DAO、购车规则或预测模型。接入这些内容可以在完整运行流程确认后再做。
