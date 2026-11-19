# 文档放在哪里

车辆可接受性只有三分类模型。代码在 `SimpleSimulator/Models/three_class.py`，报告代码在 `SimpleSimulator/Evaluation/three_class.py`。

| 要看的内容 | 位置 |
| --- | --- |
| 三分类说明 | `docs/three_class/ThreeClassAcceptability.md` |
| 重新跑出的报告 | `docs/three_class/outputs/three_class_scenario_report.txt` |
| 共用的数据和划分 | `SimpleSimulator/Models/data.py` |

在 `SimpleSimulator` 目录运行：

```text
python main.py CarPurchaseScenario LinearClassifierParameters --three-class
python main.py CarPurchaseScenario LinearClassifierParameters --evaluate
```

`--three-class` 打印训练划分和场景结果。`--evaluate` 在此之外再加上开发集交叉验证和单属性敏感度。模拟器里的 agent 也使用这个三分类模型，选项是 0、1、2。

场景审查、数据来源、偏好设计和仓库可见性留在 `docs/` 根目录，它们不单独定义另一个模型。
