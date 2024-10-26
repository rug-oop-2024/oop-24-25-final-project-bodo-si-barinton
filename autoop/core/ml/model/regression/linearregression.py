from autoop.core.ml.model.model import Model
import numpy as np



class MultipleLinearRegression(Model):
    
    def __init__(self, **kwargs):
        # Initialize the base class and set the model type to "regression"
        super().__init__(model_type="regression", **kwargs)
        
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        X = np.hstack((np.ones((observations.shape[0], 1)), observations))
        y = ground_truth

        X_transpose = np.transpose(X)
        self._parameters["coefficients"] = (
            np.linalg.pinv(X_transpose @ X) @ X_transpose @ y
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        X = np.hstack((np.ones((observations.shape[0], 1)), observations))
        coefficiants = self._parameters["coefficients"]
        prediction = np.dot(X, coefficiants)

        return prediction