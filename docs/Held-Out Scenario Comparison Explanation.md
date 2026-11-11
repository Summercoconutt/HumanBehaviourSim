# Held-Out Scenario Comparison — Output and Explanation

**Project:** Honda R&D — Human Behaviour Simulation  
**Issue:** #3 — Design and Implement an Initial Car Purchase Test Scenario  
**Command result:** `docs/outputs/held_out_scenario_comparison.txt`  
**Date of this run:** 6 November 2026  
**Branch:** `feature/issue-2-configurable-car-scenarios` (local working tree)

This is the issue #3 deliverable. The selected initial comparison is budget versus premium, using the held-out filters from the dataset proposal. The safe-family contrast is included as the second predefined pair. Both use the same baseline model and the same test partition.

This document records one executed comparison and explains every line. The figures below are predicted car acceptability from the baseline logistic regression. They are not observed purchases and not stated willingness to buy.

## 1. Command that produced the output

From `SimpleSimulator`, using the project virtual environment:

```text
python main.py CarPurchaseScenario LinearClassifierParameters --compare
```

The run exited 0. The verbatim printout is:

```text
Held-out baseline comparison (one model, trained on the training partition)
scenario                   n      mean_p   accept_rate  accept_votes
full                     260      0.3017        0.3000            78
budget                    62      0.4072        0.4032            25
premium                  124      0.2309        0.2097            26
safe_family               56      0.7437        0.7857            44
low_safety_family         46      0.0203        0.0000             0
budget minus premium: mean_p +0.1762, accept_rate +0.1935
safe_family minus low_safety_family: mean_p +0.7233, accept_rate +0.7857
Scores are predicted acceptability, not observed willingness to buy.
```

`--compare` does not print the step-by-step simulator log. It trains one `LCAgent` and scores each selected car once. A normal simulation with `LinearClassifierParameters` still creates three identical copies of that agent, so its decision log is three times as long and repeats the same vote. Those copies are not used in this table.

## 2. What was trained, and what was scored

`Data/final.csv` has 1,728 vehicle rows. `car_data_split_indices()` divides them with `random_state=42` into:

| Partition | Rows | Role in this run |
|---|---:|---|
| Training | 1,208 | Fit the `DictVectorizer` and the logistic regression |
| Validation | 260 | Not used for fitting or for the 0.5 threshold |
| Test | 260 | The only rows that appear in the table |

`full` is that entire test partition. The other four rows are rule-based subsets of those 260 cars. No test label is passed to `classifier.fit`. The threshold stays fixed at 0.5.

The positive class is `Car_Acceptability` in `{acc, good, vgood}`. `unacc` is the negative class. The model outputs the probability of the positive class.

## 3. How to read the columns

| Column | Meaning |
|---|---|
| `scenario` | Name of the held-out filter |
| `n` | Number of held-out cars in that filter |
| `mean_p` | Average predicted probability of acceptability |
| `accept_rate` | Share of those cars whose probability is at least 0.5 |
| `accept_votes` | Count of those cars. It equals `accept_rate × n` |

The printed probabilities and rates are rounded to four decimal places. Differences are calculated from the unrounded averages and then rounded. Subtracting the displayed four-decimal values can disagree by 0.0001. Section 6 uses the unrounded values.

`mean_p` and `accept_rate` answer different questions. The mean keeps the size of every probability. The rate only asks whether a car crossed 0.5. A group can have a moderate mean while few cars cross the threshold, or a high acceptance rate while a minority of low probabilities pulls the mean down.

## 4. What each scenario contains

| Scenario | Rule applied to the 260 held-out cars | Cars |
|---|---|---:|
| `full` | Every held-out car | 260 |
| `budget` | `Buying_Price` is `low` or `med`, and `Maintenance_Price` is `low` or `med` | 62 |
| `premium` | `Buying_Price` is `high` or `vhigh` | 124 |
| `safe_family` | `Person_Capacity` is `4` or `more`, and `Safety` is `high` | 56 |
| `low_safety_family` | `Person_Capacity` is `4` or `more`, and `Safety` is `low` | 46 |

These groups are all inside the test partition. They are not five separate datasets, and they do not split the 260 cars into non-overlapping blocks.

Price groups, checked on this run:

- Budget and premium share 0 cars. A car cannot have a low or medium buying price and a high buying price at the same time.
- 74 held-out cars are in neither group. Their buying price is low or medium, but maintenance is high or very high, so the budget rule fails and the premium rule fails. `62 + 124 + 74 = 260`.

Safety groups, checked on this run:

- Safe family and low-safety family share 0 cars.
- 158 held-out cars are in neither family group. They have two seats, or they have four or more seats with medium safety. `56 + 46 + 158 = 260`.

The price groups and the family groups do overlap each other, because they filter different attributes:

|  | Safe family | Low-safety family |
|---|---:|---:|
| Budget | 11 | 13 |
| Premium | 30 | 22 |

A budget-versus-premium gap is therefore not a pure price experiment. Luggage, doors, safety, capacity, fuel type, and horsepower can still differ inside each price group. The family contrast holds passenger capacity in the same band and changes the safety rating, but the other attributes are not held constant either.

The default random scenario of 10 cars is not in this table. It is a smoke sample, not one of the predefined comparison groups.

## 5. Row by row

### `full` — 260 cars, mean 0.3017, rate 0.3000, 78 votes

This is the baseline for the whole holdout. The unrounded mean probability is 0.3016910950. Predicted probabilities on these 260 cars run from 0.0000 to 0.9979. 182 cars are below 0.5 and 78 are at or above it.

`78 / 260 = 0.3000`, so the rate is exact at four decimals. The mean is almost the same as the rate. The model is not clustering every car near 0 or 1 only; the average still lands close to the share of cars that cross the threshold.

### `budget` — 62 cars, mean 0.4072, rate 0.4032, 25 votes

Unrounded mean: 0.4071866149. Probabilities run from 0.0000 to 0.9979. 37 cars are below 0.5 and 25 are at or above it.

`25 / 62 = 0.4032258065`, which prints as 0.4032. The mean and the rate are close. Relative to the full holdout, budget cars are more often scored as acceptable: 0.4032 versus 0.3000.

### `premium` — 124 cars, mean 0.2309, rate 0.2097, 26 votes

Unrounded mean: 0.2309445310. Probabilities run from 0.0000 to 0.9796. 98 cars are below 0.5 and 26 are at or above it.

`26 / 124 = 0.2096774194`, which prints as 0.2097. The mean sits above the rate. Some rejected premium cars still carry probability mass below 0.5, so the average is higher than the share of cars that actually cross the threshold. The highest premium probability is 0.9796, so the group is not uniformly rejected. Most of it is.

### `safe_family` — 56 cars, mean 0.7437, rate 0.7857, 44 votes

Unrounded mean: 0.7436531280. Probabilities run from 0.0175 to 0.9979. 12 cars are below 0.5 and 44 are at or above it.

`44 / 56 = 0.7857142857`, which prints as 0.7857. Here the rate is higher than the mean. Most of the group is accepted, and the 12 rejected cars pull the average down from that 78.57% acceptance rate. This is the highest mean and the highest rate in the table.

### `low_safety_family` — 46 cars, mean 0.0203, rate 0.0000, 0 votes

Unrounded mean: 0.0203292587. Probabilities run from 0.0000 to 0.1547. All 46 cars are below 0.5.

The acceptance rate is exactly zero because the highest probability in the group is 0.1547, well under the 0.5 threshold. The mean is small but not zero: the model still assigns these cars a little acceptability probability. It never turns that probability into a positive vote.

## 6. The two comparison lines

### Budget minus premium

```text
mean_p +0.1762, accept_rate +0.1935
```

Unrounded calculation:

- Mean probability: `0.4071866149 − 0.2309445310 = 0.1762420839`, printed as `+0.1762`.
- Acceptance rate: `0.4032258065 − 0.2096774194 = 0.1935483871`, printed as `+0.1935`.

`0.4072 − 0.2309 = 0.1763` if someone subtracts the already rounded column values. The program does not do that. It subtracts the full averages, then rounds.

The positive signs mean the budget group is higher than the premium group on both metrics. On this holdout, the baseline model predicts higher acceptability for the lower purchase-and-maintenance group than for the high purchase-price group. The rate gap is about 19 percentage points: 40.32% versus 20.97%.

This supports the proposal's first research question: the predefined price scenarios produce different predicted acceptability. It does not isolate price. Other vehicle attributes also differ between the 62 and the 124 cars.

### Safe family minus low-safety family

```text
mean_p +0.7233, accept_rate +0.7857
```

Unrounded calculation:

- Mean probability: `0.7436531280 − 0.0203292587 = 0.7233238693`, printed as `+0.7233`.
- Acceptance rate: `0.7857142857 − 0.0000000000 = 0.7857142857`, printed as `+0.7857`.

`0.7437 − 0.0203 = 0.7234` from the rounded column. The printed `+0.7233` is the rounded full-precision difference.

This is the larger of the two contrasts. Moving from low safety to high safety, within cars that seat four or more, raises the mean predicted probability by about 0.72 and raises the acceptance rate from 0% to 78.57%. The low-safety group contributes no positive votes at all.

Capacity is matched at the rule level. Doors, luggage, price, fuel type, and horsepower are not individually controlled, so the gap is a safety-focused scenario difference rather than a one-factor sensitivity test.

## 7. Final line

```text
Scores are predicted acceptability, not observed willingness to buy.
```

`final.csv` is the UCI car-evaluation table plus `Fuel_Type` and `Horse_power`. Its label is a vehicle-evaluation class. The logistic regression learns that class. A positive vote means "the model scores this configuration as acceptable," not "a buyer would purchase this car."

## 8. Held-out labels, kept separate from the model output

The command output does not print dataset labels. A separate check on the same selected indices counted how many holdout cars are actually labelled acceptable. That check is not a row-by-row accuracy score. Equal counts do not mean the same cars were correct.

| Scenario | Model accept votes | Holdout cars labelled acceptable |
|---|---:|---:|
| full | 78 of 260 | 78 of 260 |
| budget | 25 of 62 | 25 of 62 |
| premium | 26 of 124 | 27 of 124 |
| safe_family | 44 of 56 | 41 of 56 |
| low_safety_family | 0 of 46 | 0 of 46 |

The full, budget, and low-safety counts match. Premium is off by one car at the count level. Safe family is higher in the model (44) than in the labels (41). No confusion matrix was calculated, so this table does not say which individual cars the model got right.

The dataset proposal also reports label rates on all 1,728 rows, not on this 260-row holdout. Those full-dataset label rates are a different calculation and are not repeated here as model results.

## 9. What this output answers

The comparison shows measurable differences in predicted acceptability across the predefined groups:

- Budget is higher than premium on both the average probability and the acceptance rate.
- Safe family is much higher than low-safety family on both metrics. Low-safety family receives no positive votes, and none of its probabilities reach 0.5.

The same trained model produced every row. Repeating `--compare` in this environment uses the same split and the same classifier seed, so the table is reproducible here.

## 10. What this output does not say

- It does not measure real purchase behaviour or willingness to buy.
- It does not show that the three simulator agents have different preferences. They are the same model.
- It does not provide a confidence interval or a resampling uncertainty estimate.
- It does not prove that price alone, or safety alone, caused the gap. The filters leave other attributes free to vary.
- It does not report car-by-car classification accuracy.
- The validation partition was not used to choose the model or the threshold.
