# Car Purchase Dataset Review & Scenario Design Proposal

**Project:** Honda R&D — Human Behaviour Simulation  
**Workstream:** Car Purchase Scenario Design  
**Related Issue:** #1 — Review Car Purchase Datasets for Scenario Design  
**Date:** 4 November 2026

## 1. Objective

The objective of this review is to assess the suitability of the existing car purchase dataset, explore alternative datasets, and define an explainable set of test scenarios for evaluating the Human Behaviour Simulation framework.

The central research question is:

**Can we demonstrate measurable differences in predicted car acceptability across different, predefined vehicle scenarios?**

The current phase focuses on establishing a reproducible scenario-based evaluation framework using the existing baseline linear classifier. More sophisticated agent-specific preference modelling will be considered in a later phase.

Importantly, the current dataset measures **car acceptability rather than actual purchasing behaviour or stated willingness to buy**. The two concepts should not be treated as interchangeable.

## 2. Existing Dataset Review

### 2.1 Dataset Overview

**Source:** `Data/final.csv`  
**Records:** 1,728  
**Features:** 8  
**Target:** `Car_Acceptability`

| Feature | Description |
|---|---|
| Buying_Price | Purchase price category |
| Maintenance_Price | Maintenance cost category |
| No_of_Doors | Number of vehicle doors |
| Person_Capacity | Passenger capacity |
| Size_of_Luggage | Luggage capacity |
| Safety | Safety rating |
| Fuel_Type | Fuel / powertrain type |
| Horse_power | Engine power |

The dataset contains four target categories: `unacc`, `acc`, `good` and `vgood`.

For the existing binary classifier, these are mapped into acceptable (1) and unacceptable (0).

The dataset contains no blank feature or target values. The six original attributes form 1,728 unique vehicle configurations.

The dataset corresponds to the UCI Car Evaluation dataset, with two additional attributes: Fuel_Type and Horse_power. The provenance of these additional fields has not yet been established.

### 2.2 Strengths

- **Interpretability:** The features have clear and understandable meanings, allowing scenarios to be defined using explicit rules.
- **Compatibility:** The dataset is already integrated into the existing Linear Classifier Agent and Car Purchase Scenario implementation.
- **Reproducibility:** A fixed set of attributes and configurations supports deterministic scenario filtering and repeatable testing.
- **Implementation efficiency:** The current data allows the team to prioritise scenario design without introducing a new data integration dependency.

### 2.3 Limitations

**1. Limited behavioural validity**

The dataset is derived from a rule-based vehicle evaluation model rather than observed consumer purchase decisions. Its labels do not establish real-world willingness to buy.

**2. No individual consumer characteristics**

There is no information about consumer income, household circumstances, preferences or previous purchasing behaviour. It therefore cannot directly support empirically estimated differences between individual consumers.

**3. Simplified vehicle descriptions**

Price is represented categorically rather than as a monetary value. Brand, financing, vehicle availability and other real purchasing considerations are unavailable.

**4. Unverified additional features**

Fuel_Type and Horse_power are not included in the original UCI dataset. Their generation methodology and relevance to the target must be verified before drawing behavioural conclusions from them.

**5. Class imbalance**

Approximately 70% of records are classified as unacceptable, making accuracy alone an insufficient evaluation metric.

**Overall assessment:** The existing dataset is suitable for a proof of concept demonstrating the simulator's ability to evaluate different vehicle scenarios. However, it is insufficient for validating actual consumer willingness to buy.

## 3. Alternative Dataset Assessment

Four alternative or supplementary sources were reviewed.

| Dataset | Strengths | Limitations | Recommendation |
|---|---|---|---|
| UCI Automobile | 205 vehicles with real specifications, including price, horsepower, engine and fuel economy | Historical, missing values, no consumer purchase or intention labels | Supplementary vehicle data |
| Ecdat Car — Stated Preferences for Car Choice | 4,654 US individuals making choices among six vehicle alternatives; contains household and vehicle attributes | Historical stated preferences; different task structure; not revealed purchases | Strong candidate for Advanced Agent modelling |
| Norwegian Car Discrete Choice Experiment | More than 1,000 respondents; controlled choices involving price, safety, luggage and energy efficiency | Norwegian context; historic experimental conditions; requires data integration | Promising preference-validation dataset |
| UK DfT Public Attitudes Tracker | UK-specific transport and electric-vehicle attitudes, with survey documentation and datasets | Not directly compatible with the current vehicle-level classifier | Market-context and future research reference |

### Dataset Selection Decision

**The existing dataset is recommended for the immediate scenario-design phase.**

Its key advantage is that it supports rapid implementation, controlled comparisons and reproducible evaluation within the current simulator architecture.

Replacing the dataset now would introduce additional integration work before the evaluation methodology has been established.

However, the Ecdat Car dataset and Norwegian choice experiment are stronger candidates for the later stage of research, when the focus shifts towards modelling heterogeneous consumer preferences.

The alternative datasets have been identified and reviewed at the source-documentation level. They have not yet been imported or validated in the project's simulation pipeline.

## 4. Scenario Design Methodology

### 4.1 Proposed Approach: Rule-Based Segmentation

The initial scenarios will be constructed using **expert-defined filtering rules**, rather than unsupervised clustering algorithms.

For example:

- Budget Cars: Buying_Price and Maintenance_Price are both low or medium.
- High Purchase Price Cars: Buying_Price is high or very high.
- Safe Family Cars: Passenger capacity is at least four and Safety is high.
- Low-Safety Family Cars: Passenger capacity is at least four and Safety is low.

### 4.2 Why This Method Is Appropriate

**Interpretability:** Each scenario represents a clear and recognisable vehicle category. Its definition can be explained without relying on complex clustering models.

**Reproducibility:** The same filtering rules can generate equivalent datasets in repeated experiments.

**Consistency:** All scenarios can be evaluated using the same trained baseline model, ensuring that model changes do not confound the comparisons.

**Experimental transparency:** The scenarios are defined using vehicle attributes rather than selecting samples according to favourable prediction outcomes.

**Flexibility:** Additional scenarios can be introduced without redesigning the underlying simulator.

Automatic clustering could be investigated later, but it is not required for the initial proof of concept.

### 4.3 Data Separation

The scenario categories will be applied to a held-out test partition.

The training and evaluation processes must remain separate:

1. Divide data into reproducible training, validation and testing partitions.
2. Fit preprocessing and train the baseline model using training data only.
3. Select scenario samples from the held-out partition.
4. Evaluate all scenarios using the same fixed model.
5. Record predicted probabilities, binary outcomes and relevant evaluation statistics.

This approach helps reduce data leakage and ensures consistency across comparisons.

## 5. Proposed Scenario Categories

The following descriptive results were calculated from the complete existing dataset.

**These figures represent original acceptability labels, not model predictions or real purchase rates.**

| Scenario | Definition | Records | Acceptable |
|---|---|---:|---:|
| Full Dataset | All vehicles | 1,728 | 29.98% |
| Budget Cars | Low/medium purchase and maintenance prices | 432 | 42.59% |
| High Purchase Price Cars | High/very high purchase price | 864 | 20.83% |
| Safe Family Cars | Four or more passengers; high safety | 384 | 77.86% |
| Low-Safety Family Cars | Four or more passengers; low safety | 384 | 0.00% |
| Sporty Proxy | Horsepower ≥250; two/three doors | 373 | 26.54% |
| Electric Cars | Electric fuel type | 446 | 31.17% |

### 5.1 Budget Cars vs High Purchase Price Cars

**Research question:** Does the baseline classifier predict different acceptability levels across vehicle groups with different price characteristics?

**Rationale:**

Purchase price and maintenance cost are directly relevant to vehicle evaluation and affordability.

These two scenarios have clear, mutually exclusive purchase-price categories and a substantial number of records.

This makes them appropriate for an initial, interpretable comparison.

**Expected value:**

The experiment will measure differences in predicted acceptability across price-defined groups.

However, an aggregate group difference cannot be attributed exclusively to price if other vehicle attributes also differ. A controlled single-feature comparison will be required to examine the model's price sensitivity separately.

### 5.2 Safe Family Cars vs Low-Safety Family Cars

**Research question:** Does the baseline model respond differently to vehicles with similar passenger capacity but different safety ratings?

**Rationale:**

Safety and passenger capacity are original dataset attributes with explicit interpretations.

The two scenario definitions are transparent and allow a focused safety-related comparison.

The exploratory data show substantial differences between these groups.

**Expected value:**

This comparison is useful for validating whether the scenario-based evaluation framework can capture the baseline model's sensitivity to safety-related attributes.

**Limitation:** Because the underlying dataset follows a rule-based evaluation structure, the observed differences should be interpreted as model and dataset behaviour rather than evidence of real consumer safety preferences.

### 5.3 Sporty and Electric Vehicle Scenarios

These categories are potential extensions but are not recommended for the primary evaluation until the additional attributes have been validated.

Sporty vehicles could be approximated using horsepower and door count, while electric vehicles can be selected using Fuel_Type.

However, these categories should not be presented as empirically validated consumer-preference segments without evidence supporting the additional features.

## 6. Evaluation Framework

The following metrics will be used to compare scenario-level outputs.

**Mean Predicted Acceptability Probability**

Average probability of an acceptable classification within each scenario.

**Predicted Acceptance Rate**

Proportion of vehicles predicted as acceptable using the same classification threshold.

**Difference Between Scenarios**

Absolute difference in average predicted probabilities between predefined vehicle groups.

**Robustness and Interpretation**

Where appropriate, report sample sizes, resampling-based uncertainty and relevant model-performance measures.

For controlled sensitivity testing, vary one vehicle attribute at a time while holding the remaining model inputs constant.

The evaluation will distinguish between:

- Differences caused by the composition of the scenario dataset.
- Model sensitivity to individual vehicle characteristics.
- Future differences attributable to agent-specific preferences.

A successful experiment should produce reproducible and interpretable results. It does not require a guaranteed difference or a particular direction of effect.

## 7. Recommendation for Initial Implementation

**Recommended initial experiment: Budget Cars vs High Purchase Price Cars.**

This comparison should be implemented first because it provides:

- Clear and straightforward scenario definitions.
- A direct connection to an economically meaningful vehicle attribute.
- Non-overlapping purchase-price categories.
- Sufficient data for an initial held-out comparison.
- Low additional implementation complexity.

### Initial Implementation Tasks

1. Establish a shared, reproducible train/test split for the Agent and Scenario components.
2. Refactor CarPurchaseScenario.py to support configurable filtering rules, sample sizes and random seeds.
3. Implement Full Holdout, Random Subset, Budget and High Purchase Price scenarios.
4. Run the existing baseline classifier on each scenario.
5. Compare predicted acceptability probabilities and binary decisions.
6. Document experimental results and limitations.
7. Extend the framework to family safety scenarios after the initial implementation is reviewed.

The longer-term objective will be to apply the same scenarios to agents with different preference parameters, allowing the team to test whether heterogeneous agent types respond differently to identical vehicle configurations.

## 8. Final Conclusion

The current dataset is **suitable for the initial scenario-based simulation proof of concept**, providing a practical and reproducible foundation for the next development phase.

Rule-based scenario segmentation is recommended because it prioritises interpretability, transparency and controlled testing over unnecessary modelling complexity.

The initial Budget vs High Purchase Price comparison should establish the evaluation pipeline, followed by a safety-oriented family vehicle comparison.

For subsequent advanced agent development, real stated-choice datasets should be investigated to support more credible modelling of individual preferences and purchasing decisions.

**Deliverable status:** Dataset review, candidate scenario definitions, selection rationale and initial implementation recommendation completed. Predictive scenario experiments and integration of alternative datasets remain future work.

## References

- Current repository: https://github.com/Summercoconutt/HumanBehaviourSim
- UCI Car Evaluation: https://archive.ics.uci.edu/dataset/19/car+evaluation
- UCI Automobile: https://archive.ics.uci.edu/dataset/10/automobile
- Ecdat Car: https://rdrr.io/cran/Ecdat/man/Car.html
- Norwegian discrete choice experiment: https://data.mendeley.com/datasets/m26wv676yn/1
- UK DfT Public Attitudes Tracker: https://www.gov.uk/government/publications/transport-and-transport-technology-public-attitudes-tracker
