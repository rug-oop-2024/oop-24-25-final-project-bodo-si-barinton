from autoop.core.ml.model.model import Model, PrivateAttr
import numpy as np
from typing import Any, Optional


class Node:
    def __init__(self):
        self._split_feature = None
        self._split_value = None
        self._left_child = None
        self._right_child = None

        self._prediction = None
        self._is_leaf = False

    @property
    def split_feature(self) -> Optional[Any]:
        return self._split_feature

    @split_feature.setter
    def split_feature(self, value: Any) -> None:
        self._split_feature = value

    @property
    def split_value(self) -> Optional[float]:
        return self._split_value

    @split_value.setter
    def split_value(self, value: float) -> None:
        self._split_value = value

    @property
    def left_child(self) -> Optional['Node']:
        return self._left_child

    @left_child.setter
    def left_child(self, value: 'Node') -> None:
        self._left_child = value

    @property
    def right_child(self) -> Optional['Node']:
        return self._right_child

    @right_child.setter
    def right_child(self, value: 'Node') -> None:
        self._right_child = value

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf
    
    @is_leaf.setter
    def is_leaf(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("Value not of type bool")
        self._is_leaf = value
    
    @property
    def prediction(self) -> Any:
        return self._prediction
    
    @prediction.setter
    def prediction(self, value: float) -> None:
        self._prediction = value


class DecisionTreeRegressor(Model):
    _max_depth: int = PrivateAttr()
    _min_samples_split: int = PrivateAttr()
    _root: Optional[Any] = None

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,asset_path="default_path", version="1.0.0", name="DecisionTreeRegression", 
                 data=None, tags=None, metadata=None, **kwargs):
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
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._root = None
    
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Decision Tree Regressor.

        Args:
            X (np.ndarray): Training input data.
            y (np.ndarray): Target output labels.
        """
        self._root = self._build_tree(observations, ground_truth, depth=0)
        self._parameters['max_depth'] = self._max_depth
        self._parameters['min_samples_split'] = self._min_samples_split

    def _build_tree(self, observations: np.ndarray, ground_truth: np.ndarray, depth: int) -> Node:
        """
        Recursively build the decision tree.

        Args:
            observations (np.ndarray): Training input data at current node.
            ground_truth (np.ndarray): Target output labels at current node.
            depth (int): Current depth of the tree.

        Returns:
            Node: The current node of the tree.
        """
        node = Node()

        
        if depth >= self._max_depth or len(ground_truth) < self._min_samples_split:
            node.is_leaf = True
            node.prediction = np.mean(ground_truth)
            return node

        
        best_split_feature = 0
        best_split_value = np.median(observations[:, best_split_feature])

        
        node._split_feature = best_split_feature
        node._split_value = best_split_value

        
        left_indices = observations[:, best_split_feature] <= best_split_value
        right_indices = observations[:, best_split_feature] > best_split_value

        
        node._left_child = self._build_tree(observations[left_indices], ground_truth[left_indices], depth + 1)
        node._right_child = self._build_tree(observations[right_indices], ground_truth[right_indices], depth + 1)

        return node

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input data.

        Args:
            X (np.ndarray): Input data for making predictions.

        Returns:
            np.ndarray: Predicted output values.
        """
        predictions = [self._traverse_tree(x, self._root) for x in observations]
        return np.array(predictions)

    def _traverse_tree(self, observations: np.ndarray, node: Node) -> float:
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            x (np.ndarray): Input data sample.
            node (Node): Current node in the tree.

        Returns:
            float: Predicted value for the sample.
        """
        if node.is_leaf:
            return node.prediction

        if observations[node._split_feature] <= node._split_value:
            return self._traverse_tree(observations, node._left_child)
        else:
            return self._traverse_tree(observations, node._right_child)


