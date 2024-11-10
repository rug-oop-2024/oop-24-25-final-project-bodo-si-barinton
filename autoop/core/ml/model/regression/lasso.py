from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import Lasso

from autoop.core.ml.model.model import Model


class LassoRegression(Model):
    """
    Lasso Regression model implementation using scikit-learn.
    Inherits from base Model class and performs L1 regularized regression.
    """

    _model: Lasso = PrivateAttr()

    def __init__(
        self,
        asset_path: str = "default_path",
        version: str = "1.0.0",
        name: str = "LassoRegression",
        data: Optional[bytes] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> None:
        """
        Initialize the Lasso Regression model.

        Args:
            asset_path: Path to save model artifacts.
            version: Model version. Defaults to "1.0.0".
            name: Model name. Defaults to "LassoRegression".
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
        self._model = Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Lasso Regression model.

        Args:
            observations: Training input data of shape (n_samples, n_features).
            ground_truth: Target output values of shape (n_samples,).
        """
        self._model.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._model.get_params()
        self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input data.

        Args:
            observations: Input data of shape (n_samples, n_features).

        Returns:
            Predicted output values of shape (n_samples,).

        Raises:
            ValueError: If the model hasn't been fitted.
        """
        if not self._is_trained:
            raise ValueError("The model must be fitted.")
        return self._model.predict(observations)
