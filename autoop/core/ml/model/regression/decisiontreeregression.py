from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import PrivateAttr

from autoop.core.ml.model.model import Model


class Node:
    """
    A node in the decision tree representing either a decision point or leaf.
    Contains split information and child nodes references.
    """

    def __init__(self) -> None:
        """Initialize an empty node with default values."""
        self._split_feature: Optional[int] = None
        self._split_value: Optional[float] = None
        self._left_child: Optional["Node"] = None
        self._right_child: Optional["Node"] = None
        self._prediction: Optional[float] = None
        self._is_leaf: bool = False

    @property
    def split_feature(self) -> Optional[int]:
        """Get the feature index used for splitting at this node."""
        return self._split_feature

    @split_feature.setter
    def split_feature(self, value: int) -> None:
        """Set the feature index used for splitting at this node."""
        self._split_feature = value

    @property
    def split_value(self) -> Optional[float]:
        """Get the threshold value used for splitting at this node."""
        return self._split_value

    @split_value.setter
    def split_value(self, value: float) -> None:
        """Set the threshold value used for splitting at this node."""
        self._split_value = value

    @property
    def left_child(self) -> Optional["Node"]:
        """Get the left child node."""
        return self._left_child

    @left_child.setter
    def left_child(self, value: "Node") -> None:
        """Set the left child node."""
        self._left_child = value

    @property
    def right_child(self) -> Optional["Node"]:
        """Get the right child node."""
        return self._right_child

    @right_child.setter
    def right_child(self, value: "Node") -> None:
        """Set the right child node."""
        self._right_child = value

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value: bool) -> None:
        """Set whether the node is a leaf node."""
        if not isinstance(value, bool):
            raise ValueError("Value must be of type bool")
        self._is_leaf = value

    @property
    def prediction(self) -> Optional[float]:
        """Get the prediction value for leaf nodes."""
        return self._prediction

    @prediction.setter
    def prediction(self, value: float) -> None:
        """Set the prediction value for leaf nodes."""
        self._prediction = value


class DecisionTreeRegressor(Model):
    """
    Decision Tree Regressor implementation using binary recursive partitioning.
    Inherits from base Model class.
    """

    _max_depth: int = PrivateAttr()
    _min_samples_split: int = PrivateAttr()
    _root: Optional[Node] = PrivateAttr(default=None)

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        asset_path: str = "default_path",
        version: str = "1.0.0",
        name: str = "DecisionTreeRegression",
        data: Optional[bytes] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> None:
        """
        Initialize the Decision Tree Regressor.

        Args:
            max_depth: Maximum depth of the tree.
            min_samples_split: Minimum samples required to split a node.
            asset_path: Path to save model artifacts.
            version: Model version.
            name: Model name.
            data: Serialized model data.
            tags: Model tags.
            metadata: Model metadata.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b"",
            type="regression",
            tags=tags,
            metadata=metadata,
            **kwargs
        )
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._root = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Decision Tree Regressor.

        Args:
            observations: Training input data of shape (n_samples, n_features).
            ground_truth: Target values of shape (n_samples,).
        """
        self._root = self._build_tree(observations, ground_truth, depth=0)
        self._parameters["max_depth"] = self._max_depth
        self._parameters["min_samples_split"] = self._min_samples_split

    def _build_tree(
        self, observations: np.ndarray, ground_truth: np.ndarray, depth: int
    ) -> Node:
        """
        Recursively build the decision tree.

        Args:
            observations: Training data at current node.
            ground_truth: Target values at current node.
            depth: Current depth in the tree.

        Returns:
            A Node representing the current position in the tree.
        """
        node = Node()

        if (
            depth >= self._max_depth
            or len(ground_truth) < self._min_samples_split
        ):
            node.is_leaf = True
            node.prediction = np.mean(ground_truth)
            return node

        best_split_feature = 0
        best_split_value = np.median(observations[:, best_split_feature])

        node.split_feature = best_split_feature
        node.split_value = best_split_value

        left_indices = observations[:, best_split_feature] <= best_split_value
        right_indices = observations[:, best_split_feature] > best_split_value

        node.left_child = self._build_tree(
            observations[left_indices], ground_truth[left_indices], depth + 1
        )
        node.right_child = self._build_tree(
            observations[right_indices], ground_truth[right_indices], depth + 1
        )

        return node

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input data.

        Args:
            observations: Input data of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        if not self._root:
            raise ValueError("Model must be fitted.")
        prediction = [self._traverse_tree(x, self._root) for x in observations]
        return np.array(prediction)

    def _traverse_tree(self, observation: np.ndarray, node: Node) -> float:
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            observation: Single input sample of shape (n_features,).
            node: Current node in the tree.

        Returns:
            Predicted value for the input sample.
        """
        if node.is_leaf:
            return node.prediction

        if observation[node.split_feature] <= node.split_value:
            return self._traverse_tree(observation, node.left_child)
        return self._traverse_tree(observation, node.right_child)
