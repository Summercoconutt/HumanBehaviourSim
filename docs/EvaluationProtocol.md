# Evaluation protocol

**Issues:** #8 predictive quality, #9 validation protocol  
**Branch:** `feature/issue-2-configurable-car-scenarios`

## What is being predicted

Each row is one vehicle configuration in `Data/final.csv`. The label is `Car_Acceptability`, mapped as `unacc -> 0` and `acc`, `good`, or `vgood -> 1`. The model predicts that acceptability label. It does not predict an observed purchase or a willingness to pay.

## Splits

`car_data_split_indices()` keeps the existing deterministic 70/15/15 split (`random_state=42`):

| Partition | Role |
|---|---|
| Training, 1,208 rows | Fit the deployed `DictVectorizer` and logistic regression |
| Validation, 260 rows | Part of development data only |
| Test, 260 rows | Frozen holdout |

Development data is the training partition plus the validation partition. Stratified 5-fold cross-validation runs only inside development data. Each fold fits its own vectorizer on that fold's training rows. The deployed scenario model is still the single training-partition model, so older baseline runs stay comparable.

The decision threshold is 0.5. It was chosen in advance. It is not selected by scanning the holdout.

## Two different reports

`--compare` answers: how do predicted probabilities differ across predefined vehicle groups?

`--evaluate` answers two further questions, and it labels them separately:

1. Exploratory holdout quality. For `full`, the original groups, and the proxy groups, it reports sample size, true positive prevalence, predicted positive rate, accuracy, the confusion matrix, macro F1, balanced accuracy, Brier score, and log loss. A group with only one true class leaves macro F1 and balanced accuracy unset instead of printing a misleading average.
2. Development-set check. Logistic regression is compared with a majority-class baseline inside the five development folds. This check does not read the frozen test labels.

The holdout section is exploratory because those same test rows were already inspected while the scenarios were designed. A fresh final estimate would need a new, untouched test set. Predicted positive rate is not accuracy: the rate only counts probabilities of at least 0.5, while accuracy compares each prediction with the row's label.

## Commands

From `SimpleSimulator`:

```text
python main.py CarPurchaseScenario LinearClassifierParameters --compare
python main.py CarPurchaseScenario LinearClassifierParameters --evaluate
python main.py CarPurchaseScenario LinearClassifierParameters --sensitivity
```

## What remains historical

The scenario differences already reported from this holdout stay available. They are not reclassified as a final accuracy claim.

## Executed check on 12 November 2026

`--evaluate` exited 0. On the exploratory holdout, the full group has accuracy 0.9615 (confusion: 177 true negatives, 5 false positives, 5 false negatives, 73 true positives), macro F1 0.9542, balanced accuracy 0.9542, and Brier score 0.0393. `low_safety_family` has only the negative class, so macro F1 and balanced accuracy are left undefined; its accuracy of 1.0000 only means every car was correctly called unacceptable.

Inside development data, mean logistic balanced accuracy is 0.9487 against a majority baseline of 0.5000. Mean Brier score is 0.0367 against 0.2099 for that baseline. Lower Brier is better. These development numbers do not use the 260 frozen test rows.

`--sensitivity` on the constructed medium car moves acceptability from 0.9604 to 0.0490 when safety alone changes from medium to low, which flips the 0.5 decision. Changing buying price from medium to very high moves the probability from 0.9604 to 0.5400 and does not flip that particular car. This is model sensitivity, not a consumer effect.
