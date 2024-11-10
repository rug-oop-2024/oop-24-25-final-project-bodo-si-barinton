from autoop.core.ml.model.model import Model
from sklearn.linear_model import LogisticRegression
import numpy as np
from pydantic import PrivateAttr


class LogisticClassification(Model):

    _classifier: LogisticRegression = PrivateAttr()

    def __init__(self, asset_path="default_path", version="1.0.0", name="LogisticClassification", 
                 data=None, tags=None, metadata=None, **kwargs):
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b'',
            type="classification",
            tags=tags,
            metadata=metadata,
            **kwargs
        )
        self._classifier = LogisticRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
         if observations.ndim != 2 or ground_truth.ndim != 1:
            raise ValueError("Observations must be 2D and ground_truth must be 1D.")
         
         self._classifier.fit(observations, ground_truth)
         self._parameters["coefficients"] = self._classifier.coef_
         self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("The model must be trained before making predictions.")
        predictions = self._classifier.predict(observations)
        return predictions
       