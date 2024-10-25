
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Optional
from pydantic import PrivateAttr, Field

class Model(Artifact):

    _parameters: dict = PrivateAttr(default_factory=dict)
    model_type : Literal["classification" , "regression"]

    @property
    def parameters(self):
        return deepcopy(self._parameters)
    
    @abstractmethod
    def fit(self, observations : np.ndarray, ground_truth : np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations : np.ndarray) -> np.ndarray:
        pass

    def read(self) -> Optional[bytes]:
        return self._data
    
    def save(self, data : bytes) -> None:
        self._data = data

    
