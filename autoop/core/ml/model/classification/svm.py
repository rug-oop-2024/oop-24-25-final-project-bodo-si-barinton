import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from autoop.core.ml.model.model import Model
from pydantic import PrivateAttr

class SVM(Model):

    _classifier: svm.SVC = PrivateAttr()
    

    def __init__(self, asset_path="default_path", version="1.0.0", name="SVM", 
                 data=None, tags=None, metadata=None, **kwargs):
        # Initialize the base class (Model) and set the model type to "regression"
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
        self._classifier = svm.SVC(kernel="linear", gamma="auto", C=2)

    def fit(self, obseravations: np.ndarray, ground_truth: np.ndarray) -> None:
        self._classifier.fit(obseravations, ground_truth)
        self._parameters["coefficients"] = self._classifier.coef_
        self._is_trained = True
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        if self._is_trained == False:
            raise ValueError("The model must be trained before making predictions.") 
        predictions = self._classifier.predict(observations)
        return predictions
