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
    pass
    

class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    name : str = Field()

    @abstractmethod
    def calculate(self, observations : np.ndarray, ground_truth : np.ndarray) -> float:
        pass

    def __call__(self, observations : np.ndarray, ground_truth : np.ndarray) -> float:
        if len(observations) != len(ground_truth):
            raise ValueError("Parameters not of equal lenght")
        return self.calculate(observations, ground_truth)

        

# add here concrete implementations of the Metric class
    