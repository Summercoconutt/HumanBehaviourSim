# Three-class acceptability

**Issues:** #13 three-class model, #14 scenario test  
**Branch:** `feature/issue-2-configurable-car-scenarios`  
**Executed report:** `docs/outputs/three_class_scenario_report.txt`

## Label map

`Data/final.csv` has four `Car_Acceptability` strings. The three-class model uses three of those groups. It does not add a neutral class.

| Original label | Class | Name |
| --- | --- | --- |
| `unacc` | 0 | unacceptable |
| `acc` | 1 | acceptable |
| `good` | 2 | good |
| `vgood` | 2 | good |

`vgood` is grouped with `good` because the meeting described three groups: unacceptable, acceptable, and good. The binary model is unchanged: `unacc` is 0, and `acc`, `good`, and `vgood` are 1. The simulator agents still use that binary model. A three-class score is a predicted acceptability class, not an observed purchase.

## Fit

The split is the existing binary 70/15/15 split (`random_state=42`): 1,208 training rows, 260 validation rows, and 260 test rows. The three-class labels were not used to redraw those indices. On this split the class counts are:

| Split | n | unacceptable | acceptable | good |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,208 | 846 | 265 | 97 |
| Validation | 260 | 182 | 58 | 20 |
| Test | 260 | 182 | 61 | 17 |

`DictVectorizer` and logistic regression are fit on the 1,208 training rows only. The decision is the class with the highest probability. That rule was set before looking at the holdout. It was not tuned there.

The binary model still uses a 1,000-iteration limit. This three-class fit did not converge at 1,000 iterations, so its limit is 2,000. The executed fit stopped at 1,339 iterations.

Run it from `SimpleSimulator`:

```text
python main.py CarPurchaseScenario LinearClassifierParameters --three-class
```

`--compare` and `--evaluate` still run the binary model.

## Validation

The validation partition was not used to fit the model.

- Accuracy 0.9385
- Macro F1 0.8908
- Confusion matrix, rows true and columns predicted, in the order unacceptable, acceptable, good:

```text
177  5  0
  5 52  1
  0  5 15
```

Of the 20 validation cars whose label is good, 15 are predicted good and 5 are predicted acceptable. None of the true unacceptable cars are predicted good.

## Exploratory holdout

The 260 test rows were already used to design the scenarios, so this is not a fresh final exam. Accuracy is also 0.9385 and macro F1 is 0.8777. The matching accuracy is 244 correct rows in each partition. The mistakes differ: validation macro F1 is higher, and the holdout confusion matrix is:

```text
177  5  0
  5 56  0
  0  6 11
```

The predicted counts from this matrix are 182 unacceptable, 67 acceptable, and 11 good. Those are the same counts as the `full` scenario row below. Equal counts do not mean the same cars were correct.

## Scenario results

The scenario filters are unchanged. Counts below are decisions. Groups have different sizes and some groups overlap, so the comparison uses class shares and mean probabilities.

| Scenario | n | Unacceptable | Acceptable | Good | Mean P(good) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 260 | 182 | 67 | 11 | 0.0625 |
| budget | 62 | 37 | 14 | 11 | 0.2052 |
| premium | 124 | 98 | 26 | 0 | 0.0079 |
| safe_family | 56 | 12 | 36 | 8 | 0.1900 |
| low_safety_family | 46 | 46 | 0 | 0 | 0.0103 |

Budget minus premium: the good share is 0.1774 higher for budget, and the mean probability of good is 0.1972 higher. Premium has no car assigned to good. The acceptable shares are close, a difference of 0.0161.

Safe family minus low-safety family: the unacceptable share is 0.7857 lower for the safe family. Every low-safety family car is predicted unacceptable. The safe family is mostly predicted acceptable, with 8 of 56 predicted good.

The same model therefore gives different class distributions under these scenarios. This is a concept check on the current vehicle table. A later study would need properly sourced choice data.

## What this does not change

The default simulator still asks the binary acceptability question and still runs three copies of the binary agent. `--three-class` only prints this report.
