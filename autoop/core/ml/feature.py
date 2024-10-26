
from pydantic import BaseModel, Field, PrivateAttr
from typing import Literal, Optional, Any
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    name : str = Field(...)
    type: Literal["categorical", "numerical"] = Field(...)
    _data: Optional[Any] = PrivateAttr(None)

    def set_data(self, data) -> None:
        """Sets the data for the feature, enforcing NumPy array for numerical features."""
        if self.type == "numerical":
            self._data = np.array(data)
        else:
            self._data = data

