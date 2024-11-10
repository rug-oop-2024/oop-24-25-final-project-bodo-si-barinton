import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

class TestFeatures(unittest.TestCase):

    def setUp(self) -> None:
        self.iris_data = load_iris()
        self.iris_df = pd.DataFrame(
            self.iris_data.data,
            columns=self.iris_data.feature_names,
        )
        self.adult_data = fetch_openml(name="adult", version=1, parser="auto")
        self.adult_df = pd.DataFrame(
            self.adult_data.data,
            columns=self.adult_data.feature_names,
        )

    def test_detect_features_continuous(self):
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=self.iris_df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in self.iris_data.feature_names, True)
            self.assertEqual(feature.type, "numerical")
        
    def test_detect_features_with_categories(self):
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=self.adult_df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in self.adult_data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in numerical_columns, features):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(lambda x: x.name in categorical_columns, features):
            self.assertEqual(detected_feature.type, "categorical")

if __name__ == "__main__":
    unittest.main()

