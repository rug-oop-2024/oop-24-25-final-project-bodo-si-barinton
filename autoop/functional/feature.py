
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    df = dataset.read()
    features : List[Feature] = []

    if df.empty:
        raise ValueError("The provided dataset is empty.")

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column, feature_type=feature_type)
        feature.set_data(df[column].values)
        features.append(feature)
    
    return features
