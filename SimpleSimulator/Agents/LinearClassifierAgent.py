"""Import path for the three-class acceptability model.

The model itself is ``Models.three_class``.
Shared loading and the frozen split are ``Models.data``.
The fit counter lives on ``Models.three_class.three_class_fit_count``.
"""

from Models.data import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FeatureRow,
    car_data_split_indices,
    encode_features,
    features_from_context,
    load_labeled_rows,
    split_data,
)
from Models.three_class import (
    THREE_CLASS_MAPPING,
    THREE_CLASS_NAMES,
    LCAgent,
    ThreeClassModel,
    class_targets_from_labels,
    initialize_three_class_classifier,
    load_data,
    score_three_class,
    train_three_class_model,
)


def __getattr__(name: str):
    if name == "three_class_fit_count":
        from Models import three_class

        return three_class.three_class_fit_count
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
