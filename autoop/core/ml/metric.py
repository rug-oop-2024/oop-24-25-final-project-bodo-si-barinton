from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pydantic import BaseModel, Field


# Updated list of metric names
METRICS = [
    "mean_squared_error",
    "accuracy",
    "logloss",
    "micro",
    "macro",
    "mean_absolute_error",
    "root_mean_squared_error"
]

def get_metric(name: str) -> 'Metric':
    # Factory function to get a metric by name.
    if name == "mean_squared_error":
        return MeanSquaredErrorMetric()
    elif name == "accuracy":
        return AccuracyMetric()
    elif name == "logloss":
        return LogLossMetric()
    elif name == "micro":
        return MicroAverageMetric()
    elif name == "macro":
        return MacroAverageMetric()
    elif name == "mean_absolute_error":
        return MeanAbsoluteErrorMetric()
    elif name == "root_mean_squared_error":
        return RootMeanSquaredErrorMetric()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")

def count_metrics_per_class(observations: np.ndarray, ground_truth: np.ndarray, cls: int):
    """Count metrics for a specific class."""
    TP = np.sum((observations == cls) & (ground_truth == cls))
    FP = np.sum((observations == cls) & (ground_truth != cls))
    FN = np.sum((observations != cls) & (ground_truth == cls))
    TN = np.sum((observations != cls) & (ground_truth != cls))
    return TP, FP, FN, TN

def get_unique_classes(ground_truth: np.ndarray) -> np.ndarray:
    """Get unique classes from the ground truth labels."""
    return np.unique(ground_truth)

def difference(observation : np.ndarray, ground_truth : np.ndarray) -> np.ndarray : 
    return ground_truth - observation

class Metric(ABC):
    """Base class for all metrics."""
    @abstractmethod
    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        pass

    def __call__(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        if len(observations) != len(ground_truth):
            raise ValueError("Parameters not of equal length")
        return self.evaluate(observations, ground_truth)

class AccuracyMetric(Metric):
    """Metric for calculating accuracy."""

    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        no_correct = np.sum(observations == ground_truth)
        return no_correct / len(observations)

class MeanSquaredErrorMetric(Metric):
    """Metric for calculating Mean Squared Error (MSE)."""

    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        difference = observations - ground_truth
        sq_difference = difference ** 2
        return np.mean(sq_difference)
    
class MeanAbsoluteErrorMetric(Metric):
    """_summary_
    Metric for Mean Absolute Error
    Args:
        Metric (_type_): _description_
    """
    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        dif = difference(observations, ground_truth)
        return np.mean(np.abs(dif))
    
class RootMeanSquaredErrorMetric(MeanSquaredErrorMetric):
    """_summary_
    Metric for RootM Mean Squared Error
    Args:
        MeanSquaredErrorMetric (_type_): _description_
    """
    def evaluate(self, observations, ground_truth):
        rmse = super().evaluate(observations, ground_truth)
        return np.sqrt(rmse)

class LogLossMetric(Metric):
    """Metric for calculating Log Loss for multi-class classification."""

    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        observations = np.clip(observations, 1e-15, 1 - 1e-15)
        log_loss_sum = 0.0
        classes = get_unique_classes(ground_truth)

        for cls in classes:
            ground_truth_binary = (ground_truth == cls).astype(float)
            observation_prob = observations[:, cls]
            log_loss_sum += -np.sum(ground_truth_binary * np.log(observation_prob) + 
                                    (1 - ground_truth_binary) * np.log(1 - observation_prob))

        return log_loss_sum / len(ground_truth)

class MicroAverageMetric(Metric):
    """Metric for calculating Micro-Averaged Precision, Recall, and F1 Score."""

    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        TP, FP, FN, _ = 0, 0, 0, 0
        classes = get_unique_classes(ground_truth)

        for cls in classes:
            tp, fp, fn, _ = count_metrics_per_class(observations, ground_truth, cls)
            TP += tp
            FP += fp
            FN += fn

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        if precision + recall == 0:
            raise ZeroDivisionError("Sum is 0")
        return 2 * (precision * recall) / (precision + recall)

class MacroAverageMetric(Metric):
    """Metric for calculating Macro-Averaged Precision, Recall, and F1 Score."""

    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> int | float:
        classes = get_unique_classes(ground_truth)
        precision_per_class = []
        recall_per_class = []

        for cls in classes:
            TP, FP, FN, _ = count_metrics_per_class(observations, ground_truth, cls)

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)

        precision_avg = np.mean(precision_per_class)
        recall_avg = np.mean(recall_per_class)

        if precision_avg + recall_avg == 0:
            raise ZeroDivisionError("Sum is 0")

        return 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)

