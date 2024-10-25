from autoop.core.ml.model import Model, PrivateAttr
import numpy as np
from sklearn.linear_model import LinearRegression as LS


class LinearRegression(Model):
    _model = LS()
    _is_trained : bool = PrivateAttr(default = False)

    def fit(self, observations : np.ndarray, ground_truth : np.ndarray) -> None:
        self._model.fit(observations, ground_truth)
        self._parameters = self._model.get_params()
        self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("The model must be trained before making predictions.")
        return self._model.predict(observations)