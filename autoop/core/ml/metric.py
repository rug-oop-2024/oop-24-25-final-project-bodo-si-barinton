from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

# Updated list of metric names
METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "f1",
    "recall"
]

def get_metric(name: str) -> 'Metric':
    # Factory function to get a metric by name.
    if name == "mean_squared_error":
        return MeanSquaredErrorMetric()
    elif name == "accuracy":
        return AccuracyMetric()
    elif name == "precision":
        return PrecisionMetric()
    elif name =="recall":
        return RecallMetric()
    elif name == "f1":
        return F1Metric()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")

def count_positives(observations: np.ndarray, ground_truth: np.ndarray) -> int:
    """Count the number of true positives where observation matches the ground truth."""
    return np.sum(observations == ground_truth)

def count_predicted_positives(observations: np.ndarray) -> int:
    """Count the number of predicted positive values."""
    return np.sum(observations == 1)

def count_false_negatives(observations : np.ndarray, ground_truth : np.ndarray) -> int:
    """Count the number of false negatives predicted"""
    return np.sum((observations == 0) & (ground_truth == 1))

class Metric(ABC):
    """Base class for all metrics."""
    @abstractmethod
    def calculate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        pass

    def __call__(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        if len(observations) != len(ground_truth):
            raise ValueError("Parameters not of equal length")
        return self.calculate(observations, ground_truth)

class AccuracyMetric(Metric):
    """Metric for calculating accuracy."""

    def calculate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        no_correct = count_positives(observations, ground_truth)
        return no_correct / len(observations)

class MeanSquaredErrorMetric(Metric):
    """Metric for calculating Mean Squared Error (MSE)."""

    def calculate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        difference = observations - ground_truth
        sq_difference = difference ** 2
        return np.mean(sq_difference)

class PrecisionMetric(Metric):
    """Metric for calculating precision."""

    def calculate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        true_positives = count_positives(observations, ground_truth)
        predicted_positives = count_predicted_positives(observations)
        if predicted_positives == 0:
            raise ZeroDivisionError("Predictided positives is 0")
        return true_positives / predicted_positives
    
class RecallMetric(Metric):
    """Metric for calculating the recall"""
    def calculate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        true_positives = count_positives
        false_negatives = count_false_negatives(observations, ground_truth)
        if (true_positives + false_negatives == 0):
            raise ZeroDivisionError("The sum of ture positives and false negatives is 0")
        return true_positives / (true_positives + false_negatives)
    
class F1Metric(Metric):

    def calculate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
       recall = RecallMetric()
       precision = PrecisionMetric()
       recall_calc = recall.calculate(observations, ground_truth)
       precision_calc = precision.calculate(observations, ground_truth)
       if recall_calc + precision_calc == 0:
           raise ZeroDivisionError("The sum of Precision and Recall is 0")
       return 2 * ((recall_calc * precision_calc)/(recall_calc + precision_calc))

