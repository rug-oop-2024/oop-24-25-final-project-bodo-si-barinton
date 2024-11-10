from autoop.core.ml.model.model import Model
import numpy as np



class MultipleLinearRegression(Model):
    
    def __init__(self, asset_path="default_path", version="1.0.0", name="MultipleLinearRegression", 
                 data=None, tags=None, metadata=None, **kwargs):
        # Initialize the base class (Model) and set the model type to "regression"
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b'',
            type="regression",
            tags=tags,
            metadata=metadata,
            **kwargs
        )
        
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