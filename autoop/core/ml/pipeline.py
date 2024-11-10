import pickle
from typing import List

import numpy as np

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


class Pipeline:
    """
    A class representing a machine learning pipeline.
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ) -> None:
        """
        Initialize a Pipeline instance.

        Args:
            metrics (List[Metric]): List of metrics to evaluate the model.
            dataset (Dataset): The dataset to use in the pipeline.
            model (Model): The model to train and evaluate.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): The target feature.
            split (float): The train-test split ratio. Default is 0.8.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and \
            model.type != "classification":
            raise ValueError(
                "Model type must be classification for categorical target feature"
            )
        if target_feature.type == "continuous" and \
            model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Return a string representation of the pipeline.

        Returns:
            str: A string representation of the pipeline.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> "Model":
        """
        Get the model used in the pipeline.

        Returns:
            Model: The model used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Get the artifacts generated during the pipeline execution to be saved.

        Returns:
            List[Artifact]: The list of artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """
        Register an artifact in the pipeline.

        Args:
            name (str): The name of the artifact.
            artifact: The artifact to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocess the input and target features.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """
        Split the data into training and testing sets.
        """
        split = self._split
        self._train_X = [
            vector[ : int(split * len(vector))] 
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)) :] 
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)) :
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compact a list of vectors into a single array.

        Args:
            vectors (List[np.array]): The list of vectors.

        Returns:
            np.array: The compacted array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Train the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluate the model using the testing data.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        Execute the pipeline.

        Returns:
            dict: A dictionary containing important data.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        X_train = self._compact_vectors(self._train_X)
        Y_train = self._train_y
        self._training_metrics_results = []
        train_predictions = self._model.predict(X_train)

        for metric in self._metrics:
            train_result = metric.evaluate(train_predictions, Y_train)
            self._training_metrics_results.append((metric, train_result))

        self._evaluate()
        return {
            "training_metrics": self._training_metrics_results,
            "evaluation_metrics": self._metrics_results,
            "predictions": self._predictions,
        }
