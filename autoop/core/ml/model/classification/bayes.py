from autoop.core.ml.model.model import Model
from sklearn.naive_bayes import GaussianNB
import numpy as np
from pydantic import PrivateAttr


class BayesClassification(Model):

    _classifier: GaussianNB = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(model_type="classification", *kwargs)
        self._classifier = GaussianNB()

    
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._classifier.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._classifier.coef_
        self._is_trained = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("The model must be trained before making predictions.")
        predcitions = self._classifier.predict(observations)
        return predcitions
        
    