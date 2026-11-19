# Vehicle attributes and proxy scenarios

**Issue:** #10

## Source of the columns

The original UCI Car Evaluation data has six features: buying price, maintenance price, doors, passenger capacity, luggage size, and safety. `Data/final.csv` also contains `Fuel_Type` and `Horse_power`.

This repository does not record who added those two columns or how their values were generated. Until that provenance is established, they are extra fields, not measured efficiency or style.

| Column | Status |
|---|---|
| Buying_Price, Maintenance_Price, No_of_Doors, Person_Capacity, Size_of_Luggage, Safety | Present in the original car-evaluation data |
| Fuel_Type | Extra column. Values in this file are petrol, diesel, hybrid, and electric. Generation method unknown |
| Horse_power | Extra numeric column. Generation method unknown |
| Energy efficiency | Not in the file. Electric is not treated as a measurement of efficiency |
| Aesthetic style | Not in the file. Horsepower and door count are not treated as a measurement of style |

## Scenarios

`budget`, `premium`, `safe_family`, and `low_safety_family` use only the original attributes.

`budget` requires both buying price and maintenance price to be low or medium. `premium` requires only the buying price to be high or very high. They are descriptive groups. They are not a symmetric price experiment. The single-feature comparison is `--sensitivity`.

Two wider groups are named as proxies:

| Name | Rule | What it is not |
|---|---|---|
| `sporty_proxy` | Horsepower at least 250 and two or three doors | Not a measured style segment |
| `electric_proxy` | `Fuel_Type` is `electric` | Not a measured efficiency segment |

Both are sampled only from the held-out test partition. Their predictions are vehicle-configuration scores from the fixed acceptability model. They do not show that different buyers prefer different cars.

A later phase can bring in a stated-preference dataset, such as the Ecdat car-choice data named in the dataset proposal, if the project needs attributes this file does not contain.
