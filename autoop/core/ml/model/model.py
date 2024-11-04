
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Optional
from pydantic import PrivateAttr
from typing import Optional, List, Dict, Any

class Model(Artifact):

    _parameters: dict = PrivateAttr(default_factory=dict)
    _is_trained: bool = PrivateAttr(default= False)

    def __init__(self, asset_path: str, version: str, name: str, data: bytes, type: str = "model",
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[Any, Any]] = None, model_type: str = None , **kwargs):
        super().__init__(asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b'',
            type=type,
            tags=tags,
            metadata=metadata)

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
    class Config:
        protected_namespaces = () 

  

    
