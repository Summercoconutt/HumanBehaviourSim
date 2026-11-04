# CarPurchaseScenario.py — Code Review and Initial Refactoring

**Project:** Honda R&D — Human Behaviour Simulation  
**Issue:** #2 — Review and Refactor CarPurchaseScenario.py  
**Date:** 4 November 2026

## 1. Original code — execution flow

1. CarPurchaseScenario() calls load_data() to read Data/final.csv. The eight vehicle attributes are model features; Car_Acceptability is converted to 0 = unacc and 1 = acc/good/vgood.
2. The first stratified 15% train/test split creates a held-out test partition with random_state=42.
3. A second random operation selects ten cars from that test partition using seed 42.
4. context exposes the selected cars to agents.
5. decisions constructs one binary Decision per car and embeds all vehicle features in the prompt.
6. SimulatorCore invokes the agent's decide() method. The agent obtains the car from state.context and uses a logistic-regression model with threshold 0.5 to return a binary choice.
7. is_complete stops when the step count reaches sample_count.

**Interpretation:** The model predicts car acceptability, not observed consumer willingness to buy. Three identical LCAgent instances do not represent distinct consumer preferences.

## 2. Findings and risks

| ID | Finding | Risk | Refactoring |
|---|---|---|---|
| F1 | Hard-coded sample size of ten | No whole-test-set or variable-size testing | Configurable sample_count |
| F2 | Random selection only | No interpretable categories | Named rule-based scenario filters |
| F3 | Separate data splitting in Agent and Scenario | Risk of partition divergence and leakage | Shared car_data_split_indices() |
| F4 | DictVectorizer fitted before splitting | Test distribution influences preprocessing | Fit on training rows only |
| F5 | Hard-coded sampling seed | Limited robustness checks | Configurable random_state |
| F6 | Only binary decisions in simulation log | Cannot report probability-level scenario differences | Defer metrics to Issue #3 |
| F7 | Fuel_Type and Horse_power provenance unknown | Uncertain behavioural interpretation | Prioritise original interpretable features |

## 3. Implemented changes

- Preserve the default CarPurchaseScenario(): ten randomly sampled held-out cars using seed 42.
- Support scenario_type, sample_count, and random_state constructor arguments.
- Add six named scenario types: random, full, budget, premium, safe_family, and low_safety_family.
- Named scenarios use all matching held-out cars unless sample_count is specified.
- **Full means the entire held-out test partition, not the full training and test dataset.**
- Store selected_indices to allow audit of overlap, without passing IDs to the classifier.
- Use one deterministic 70/15/15 train/validation/test split for both classifier training and scenario sampling.
- Fit the classifier's DictVectorizer on training features only.
- Preserve existing split_data() helper and default simulator behaviour.
- Add CLI configuration options and tests for split isolation and scenario reproducibility.

## 4. Scenario definitions

| Name | Filter | Why use it |
|---|---|---|
| random | Random rows from holdout | Existing reference smoke test |
| full | All held-out rows | Complete held-out baseline |
| budget | Buying_Price and Maintenance_Price in low/med | Transparent cost-oriented category |
| premium | Buying_Price in high/vhigh | High-price comparison |
| safe_family | Person_Capacity in 4/more; Safety high | Safety-focused family vehicle category |
| low_safety_family | Person_Capacity in 4/more; Safety low | Contrasting safety condition |

The rules are explicit, reproducible and explainable to business stakeholders. They create **vehicle evaluation sets**, not empirically observed consumer segments. Scenario data remain strictly outside training.

## 5. Running scenarios

From the SimpleSimulator directory, with dependencies installed:

    python main.py CarPurchaseScenario LinearClassifierParameters
    python main.py CarPurchaseScenario LinearClassifierParameters --scenario-type full
    python main.py CarPurchaseScenario LinearClassifierParameters --scenario-type budget --sample-count 12 --seed 42
    python main.py CarPurchaseScenario LinearClassifierParameters --scenario-type premium
    python main.py CarPurchaseScenario LinearClassifierParameters --scenario-type safe_family
    python main.py CarPurchaseScenario LinearClassifierParameters --scenario-type low_safety_family

Defaults: random / ten cars / seed 42. Named filters select all matching holdout cars unless sample_count is supplied. The seed controls sampling, not the fixed data split.

## 6. Tests and remaining validation

Run the full project tests from SimpleSimulator:

    python -m unittest discover -s tests -v

Added tests check:
- train, validation and test partitions are disjoint and cover the original rows;
- original ten-car behaviour remains intact;
- identical seeds select identical samples;
- scenario groups only contain held-out rows matching their filter;
- full scenario covers every held-out row;
- errors are raised for invalid names and impossible sample requests.

**Testing limitation:** Six added tests and one agent decision smoke test were exercised locally with a synthetic fixture and a minimal SimulatorCore stub because the sandbox did not have SimPy installed. The full end-to-end simulator test suite using the repository dataset and the actual SimPy engine **must still be run before merging**.

## 7. Acceptance and next steps

Issue #2 supplies a documented review, initial code refactoring, configurable scenario construction, shared split logic, and unit tests. Before merging, run the full suite with the actual data, address any integration problems, and obtain George's review.

Issue #3 can then add scenario-level mean probabilities, predicted acceptability rates, uncertainty estimates and agent preference comparisons. Behavioural claims about actual consumers will require respondent-level choice or willingness-to-buy data.
