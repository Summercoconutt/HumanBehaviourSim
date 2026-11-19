# Three-class acceptability

**Executed report:** `docs/three_class/outputs/three_class_scenario_report.txt`

This is the only acceptability model. The simulator agent uses it and answers 0, 1, or 2.

## Label map

| Original label | Class | Name |
| --- | --- | --- |
| `unacc` | 0 | unacceptable |
| `acc` | 1 | acceptable |
| `good` | 2 | good |
| `vgood` | 2 | good |

There is no neutral class. A score is a predicted acceptability class, not an observed purchase.

## Fit

The split is stratified 70/15/15 on these three classes (`random_state=42`).

| Split | n | unacceptable | acceptable | good |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,208 | 846 | 268 | 94 |
| Validation | 260 | 182 | 58 | 20 |
| Test | 260 | 182 | 58 | 20 |

`DictVectorizer` and logistic regression are fit on the 1,208 training rows only. The decision is the class with the highest probability. That rule was not tuned on the holdout. This fit stopped at 1,320 iterations, under the limit of 2,000.

From `SimpleSimulator`:

```text
python main.py CarPurchaseScenario LinearClassifierParameters --three-class
python main.py CarPurchaseScenario LinearClassifierParameters --evaluate
```

`--evaluate` is the report saved in `docs/three_class/outputs/`. It adds the development-set check and the single-attribute sensitivity table.

## Validation

The 260 validation rows were not used to fit the model.

- Accuracy 0.9154
- Macro F1 0.8343
- Confusion matrix, rows true and columns predicted, in the order unacceptable, acceptable, good:

```text
177  5  0
  7 48  3
  0  7 13
```

Of the 20 validation cars labeled good, 13 are predicted good and 7 are predicted acceptable.

## Exploratory holdout

The 260 test rows were already used to design the scenarios, so this is not a fresh final exam.

- Accuracy 0.9308
- Macro F1 0.8871

```text
174  8  0
  4 52  2
  0  4 16
```

Predicted counts from this matrix are 178 unacceptable, 64 acceptable, and 18 good. Those match the `full` scenario row. Equal counts do not mean the same cars were correct.

## Scenarios

| Scenario | n | Unacceptable | Acceptable | Good | Mean P(good) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 260 | 178 | 64 | 18 | 0.0776 |
| budget | 67 | 37 | 12 | 18 | 0.2635 |
| premium | 124 | 94 | 30 | 0 | 0.0081 |
| safe_family | 57 | 11 | 31 | 15 | 0.2570 |
| low_safety_family | 46 | 46 | 0 | 0 | 0.0081 |

Budget's good-class share is 0.2687 higher than premium, and its mean probability of good is 0.2554 higher. Premium has no car predicted good.

The safe family's unacceptable share is 0.8070 lower than the low-safety family. Every low-safety family car is predicted unacceptable. The safe family is mostly predicted acceptable, with 15 of 57 predicted good.

## Development check and sensitivity

Inside the training-plus-validation rows, five stratified folds give mean logistic accuracy 0.9332 and mean macro F1 0.8926. A majority-class baseline on the same folds has mean accuracy 0.7003 and mean macro F1 0.2746. Those folds do not use the 260 test rows.

On one constructed medium car, changing buying price from medium to low moves the predicted class from acceptable to good. Changing safety from medium to low moves it to unacceptable, and changing safety from medium to high moves it to good. This is model sensitivity, not observed consumer behaviour.
