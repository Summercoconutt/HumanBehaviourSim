# Agent preference design

**Issue:** #12

## Three different things

| Piece | What it is | Where it lives |
|---|---|---|
| Acceptability classifier | One three-class logistic regression trained on vehicle labels | `ThreeClassModel`, shared by agents |
| Preference weights | Experimenter-chosen cost and safety adjustments | `PreferenceAgent` |
| Decision noise | Optional Gaussian noise, drawn separately for each class | `noise_scale` and `seed` |

Sharing the trained classifier does not make agents into different buyers. The weights are not estimated from observed consumers. Noise is off unless `noise_scale` is set. The same car, the same weights, and the same seed produce the same choice. Different weights can produce different classes on a constructed car.

The plain `LCAgent` path does not add preference weights. It returns the class with the highest probability: 0 unacceptable, 1 acceptable, 2 good.

## Demonstration

From `SimpleSimulator`:

```text
python main.py CarPurchaseScenario LinearClassifierParameters --preferences
```

The printed profiles use one constructed high-price, high-safety car and the three-class probabilities from the trained model. `cost_sensitive` penalises high price. `safety_seeking` rewards high safety. A third profile has no preference weight and noise turned on, so repeated draws can change class. This is a synthetic illustration, kept separate from the holdout report.
