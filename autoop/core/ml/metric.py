from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name == "mean_squared_error":
        return MeanSquaredErrorMetric()
    elif name =="accuracy":
        return AccuracyMetric()
    else :
        raise ValueError(f"Metric '{name}' is not implemented.")
    

    

class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    @abstractmethod
    def calculate(self, observations : np.ndarray, ground_truth : np.ndarray) -> int|float:
        pass

    def __call__(self, observations : np.ndarray, ground_truth : np.ndarray) -> int|float:
        if len(observations) != len(ground_truth):
            raise ValueError("Parameters not of equal lenght")
        return self.calculate(observations, ground_truth)

        
class AccuracyMetric(Metric):

    def calculate(self, observations : np.ndarray, ground_truth : np.ndarray) -> int|float :
        correct_predictions = (observations == ground_truth)
        no_correct = np.sum(correct_predictions)
        return no_correct / len(observations)

class MeanSquaredErrorMetric(Metric):

    def calculate(self, observations : np.ndarray, ground_truth : np.ndarray) -> int|float :
        diference = observations - ground_truth
        sq_diference = diference ** 2
        return np.mean(sq_diference)
        
        