from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso
import numpy as np
from pydantic import PrivateAttr

class LassoRegression(Model):
    _model: Lasso = PrivateAttr()
    _is_trained: bool = PrivateAttr(default=False)


    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Lasso Regression model.

        Args:
            observations (np.ndarray): Training input data.
            ground_truth (np.ndarray): Target output labels.
        """
        self._model.fit(observations, ground_truth)
        self._parameters = self._model.get_params()
        self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input data.

        Args:
            observations (np.ndarray): Input data for making predictions.

        Returns:
            np.ndarray: Predicted output values.
        """
        if not self._is_trained:
            raise ValueError("The model must be trained before making predictions.")
        return self._model.predict(observations)
