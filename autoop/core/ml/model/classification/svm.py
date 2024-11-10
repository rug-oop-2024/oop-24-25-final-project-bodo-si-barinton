from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import PrivateAttr
from sklearn import svm

from autoop.core.ml.model.model import Model


class SVM(Model):
    """
    Support Vector Machine classification model implementation.
    Inherits from the base Model class and uses scikit-learn's SVC.
    """

    _classifier: svm.SVC = PrivateAttr()

    def __init__(
        self,
        asset_path: str = "default_path",
        version: str = "1.0.0",
        name: str = "SVM",
        data: Optional[bytes] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> None:
        """
        Initialize the SVM Classification model.

        Args:
            asset_path (str): Path to save model artifacts. Defaults to
                "default_path".
            version (str): Model version. Defaults to "1.0.0".
            name (str): Model name. Defaults to "SVM".
            data (Optional[bytes]): Serialized model data. Defaults to None.
            tags (Optional[List[str]]): Model tags. Defaults to None.
            metadata (Optional[Dict[Any, Any]])
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b"",
            type="classification",
            tags=tags,
            metadata=metadata,
            **kwargs
        )
        self._classifier = svm.SVC(kernel="linear", gamma="auto", C=2)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the SVM model on the provided data.

        Args:
            observations (np.ndarray): Training features of shape
                (n_samples, n_features).
            ground_truth (np.ndarray): Target values of shape (n_samples,).
        """
        self._classifier.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._classifier.coef_
        self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            observations (np.ndarray): Input features of shape
                (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).

        Raises:
            ValueError: If the model hasn't been trained before predictions.
        """
        if not self._is_trained:
            raise ValueError("The model must be fitted.")
        predictions = self._classifier.predict(observations)
        return predictions
