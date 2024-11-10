from abc import ABC, abstractmethod

import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "logloss",
    "micro",
    "macro",
    "mean_absolute_error",
    "root_mean_squared_error",
]


def get_metric(name: str) -> "Metric":
    """
    Factory function to get a metric by name.

    Args:
        name (str): The name of the metric.

    Returns:
        Metric: An instance of the requested metric.

    Raises:
        ValueError: If the metric name is not implemented.
    """
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


def count_metrics_per_class(
    observations: np.ndarray, ground_truth: np.ndarray, cls: int
) -> tuple:
    """
    Count metrics for a specific class.

    Args:
        observations (np.ndarray): The observed values.
        ground_truth (np.ndarray): The ground truth values.
        cls (int): The class to count metrics for.

    Returns:
        Tuple[int, int, int, int]: The counts of TP, FP, FN, and TN.
    """
    TP = np.sum((observations == cls) & (ground_truth == cls))
    FP = np.sum((observations == cls) & (ground_truth != cls))
    FN = np.sum((observations != cls) & (ground_truth == cls))
    TN = np.sum((observations != cls) & (ground_truth != cls))
    return TP, FP, FN, TN


def get_unique_classes(ground_truth: np.ndarray) -> np.ndarray:
    """
    Get unique classes from the ground truth labels.

    Args:
        ground_truth (np.ndarray): The ground truth labels.

    Returns:
        np.ndarray: The unique classes.
    """
    return np.unique(ground_truth)


def difference(observation: np.ndarray, ground_t: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between observations and ground truth.

    Args:
        observation (np.ndarray): The observed values.
        ground_truth (np.ndarray): The ground truth values.

    Returns:
        np.ndarray: The difference between observations and ground truth.
    """
    return ground_t - observation


class Metric(ABC):
    """
    Base class for all metrics.
    """

    @abstractmethod
    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The evaluated metric value.
        """
        pass

    def __call__(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Call the evaluate method.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The evaluated metric value.

        Raises:
            ValueError: If the lengths of parameters do not match.
        """
        if len(observations) != len(ground_truth):
            raise ValueError("Parameters not of equal length")
        return self.evaluate(observations, ground_truth)


class AccuracyMetric(Metric):
    """
    Metric for calculating accuracy.
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the accuracy metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The accuracy value.
        """
        no_correct = np.sum(observations == ground_truth)
        return no_correct / len(observations)


class MeanSquaredErrorMetric(Metric):
    """
    Metric for calculating Mean Squared Error (MSE).
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the Mean Squared Error (MSE) metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The MSE value.
        """
        difference = observations - ground_truth
        sq_difference = difference**2
        return np.mean(sq_difference)


class MeanAbsoluteErrorMetric(Metric):
    """
    Metric for calculating Mean Absolute Error (MAE).
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the Mean Absolute Error (MAE) metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The MAE value.
        """
        dif = difference(observations, ground_truth)
        return np.mean(np.abs(dif))


class RootMeanSquaredErrorMetric(MeanSquaredErrorMetric):
    """
    Metric for calculating Root Mean Squared Error (RMSE).
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the Root Mean Squared Error (RMSE) metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The RMSE value.
        """
        rmse = super().evaluate(observations, ground_truth)
        return np.sqrt(rmse)


class LogLossMetric(Metric):
    """
    Metric for calculating Log Loss for binary or multi-class classification.
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the Log Loss metric.

        Args:
            observations (np.ndarray): The predicted probabilities.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The Log Loss value.
        """
        observations = np.clip(observations, 1e-15, 1 - 1e-15)

        if observations.ndim == 1:
            ground_truth_binary = ground_truth.astype(float)
            log_loss = -np.mean(
                ground_truth_binary * np.log(observations) +
                (1 - ground_truth_binary) * np.log(1 - observations)
            )
        else:
            log_loss_sum = 0.0
            classes = get_unique_classes(ground_truth)
            
            for cls in classes:
                ground_truth_binary = (ground_truth == cls).astype(float)
                
                if cls < observations.shape[1]:
                    observation_prob = observations[:, cls]
                    log_loss_sum += -np.sum(
                        ground_truth_binary * np.log(observation_prob) +
                        (1 - ground_truth_binary) * 
                        np.log(1 - observation_prob)
                    )
                else:
                    raise ValueError("Class index exceeds no of columns in observations.")

            log_loss = log_loss_sum / len(ground_truth)

        return log_loss


class MicroAverageMetric(Metric):
    """
    Metric for calculating Micro-Averaged Precision, Recall, and F1 Score.
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the Micro-Averaged Precision, Recall, and F1 Score metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The Micro-Averaged Precision, Recall, and F1 value.
        """
        TP, FP, FN, _ = 0, 0, 0, 0
        classes = get_unique_classes(ground_truth)

        for cls in classes:
            tp, fp, fn, _ = count_metrics_per_class(
                observations, ground_truth, cls
            )
            TP += tp
            FP += fp
            FN += fn

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        if precision + recall == 0:
            raise ZeroDivisionError("Sum is 0")
        return 2 * (precision * recall) / (precision + recall)


class MacroAverageMetric(Metric):
    """
    Metric for calculating Macro-Averaged Precision, Recall, and F1 Score.
    """

    def evaluate(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> int | float:
        """
        Evaluate the Macro-Averaged Precision, Recall, and F1 Score metric.

        Args:
            observations (np.ndarray): The observed values.
            ground_truth (np.ndarray): The ground truth values.

        Returns:
            int | float: The Macro-Averaged Precision, Recall, and F1 value.
        """
        classes = get_unique_classes(ground_truth)
        precision_per_class = []
        recall_per_class = []

        for cls in classes:
            TP, FP, FN, _ = count_metrics_per_class(
                observations, ground_truth, cls
            )

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)

        precision_avg = np.mean(precision_per_class)
        recall_avg = np.mean(recall_per_class)

        if precision_avg + recall_avg == 0:
            raise ZeroDivisionError("Sum is 0")

        return 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg)
