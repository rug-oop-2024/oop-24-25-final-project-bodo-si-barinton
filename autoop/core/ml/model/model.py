import pickle
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import PrivateAttr

from autoop.core.ml.artifact import Artifact


class Model(Artifact):
    """
    Abstract base class for machine learning models.
    Inherits from Artifact and provides common model functionality.
    """

    _parameters: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _is_trained: bool = PrivateAttr(default=False)

    def __init__(
        self,
        asset_path: str,
        version: str,
        name: str,
        data: bytes,
        type: str = "model",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Model instance.

        Args:
            asset_path: Path to save model artifacts.
            version: Version of the model.
            name: Name of the model.
            data: Binary data containing the model.
            type: Type of the model. Defaults to "model".
            tags: Optional list of tags.
            metadata: Optional metadata dictionary.
            model_type: Optional specific model type.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b"",
            type=type,
            tags=tags,
            metadata=metadata,
        )

    def to_artifact(self, name: str) -> Artifact:
        """
        Convert the model to an Artifact instance for saving.

        Args:
            name: Name for the artifact.

        Returns:
            Artifact: A new Artifact instance containing the serialized model.
        """
        model_data = pickle.dumps(self)
        return Artifact(
            name=name,
            data=model_data,
            type="Model",
            asset_path=f"path/to/{name}",
            version=self.version,
            metadata={"model_type": self.type},
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get a deep copy of model parameters.

        Returns:
            Dict[str, Any]: Copy of model parameters.
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model on given data.

        Args:
            observations: Training features of shape (n_samples, n_features).
            ground_truth: Target values of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            observations: Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        pass

    def read(self) -> Optional[bytes]:
        """
        Read the model's binary data.

        Returns:
            Optional[bytes]: The model's binary data if available.
        """
        return self._data

    def save(self, data: bytes) -> None:
        """
        Save binary data to the model.

        Args:
            data: Binary data to save.
        """
        self._data = data

    class Config:
        """Pydantic configuration class."""

        protected_namespaces = ()
