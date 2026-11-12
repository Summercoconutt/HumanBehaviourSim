# Agent preference design

**Issue:** #12

## Three different things

| Piece | What it is | Where it lives |
|---|---|---|
| Acceptability classifier | One logistic regression trained on vehicle labels | `BaselineModel`, shared by agents |
| Preference weights | Experimenter-chosen cost and safety adjustments | `PreferenceAgent` |
| Decision noise | Optional Gaussian noise with an agent-specific seed | `noise_scale` and `seed` |

Sharing the trained classifier does not make agents into different buyers. The weights are not estimated from observed consumers. Noise is off unless `noise_scale` is set. The same car, the same weights, and the same seed produce the same choice. Different weights can produce different choices on a constructed car. With noise on, repeated draws vary, and the same seed repeats that sequence.

The baseline `LCAgent` path is unchanged: it still reports the classifier's own 0.5 threshold and does not add preference weights.

## Demonstration

From `SimpleSimulator`:

```text
python main.py CarPurchaseScenario LinearClassifierParameters --preferences
```

The printed profiles use one constructed high-price, high-safety car and a fixed probability of 0.5. `cost_sensitive` penalises high price. `safety_seeking` rewards high safety. A third profile has no preference weight and noise turned on, so repeated draws vary. This is a synthetic illustration, kept separate from the holdout accuracy report.
