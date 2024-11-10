from typing import Any, Dict, List, Optional

import numpy as np

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model implementation.
    Implements ordinary least squares regression using the normal equation.
    """

    def __init__(
        self,
        asset_path: str = "default_path",
        version: str = "1.0.0",
        name: str = "MultipleLinearRegression",
        data: Optional[bytes] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> None:
        """
        Initialize the Multiple Linear Regression model.

        Args:
            asset_path: Path to save model artifacts.
            version: Model version. Defaults to "1.0.0".
            name: Model name. Defaults to "MultipleLinearRegression".
            data: Serialized model data. Defaults to None.
            tags: Model tags. Defaults to None.
            metadata: Model metadata. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b"",
            type="regression",
            tags=tags,
            metadata=metadata,
            **kwargs
        )

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model using the normal equation method.

        Args:
            observations: Training features of shape (n_samples, n_features).
            ground_truth: Target values of shape (n_samples,).
        """
        X = np.hstack((np.ones((observations.shape[0], 1)), observations))
        y = ground_truth

        X_transpose = np.transpose(X)
        self._parameters["coefficients"] = (
            np.linalg.pinv(X_transpose @ X) @ X_transpose @ y
        )
        self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            observations: Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).

        Raises:
            ValueError: If the model hasn't been fitted.
        """
        if not self._is_trained:
            raise ValueError("Model must be fitted.")

        X = np.hstack((np.ones((observations.shape[0], 1)), observations))
        coefficients = self._parameters["coefficients"]
        predictions = np.dot(X, coefficients)

        return predictions
