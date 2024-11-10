from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr


class Feature(BaseModel):
    """
    A class representing a feature in a dataset.
    """

    name: str = Field(...)
    type: Literal["categorical", "numerical"] = Field(...)
    _data: Optional[Union[np.ndarray, list]] = PrivateAttr(None)

    def set_data(self, data: Union[np.ndarray, list]) -> None:
        """
        Sets the data for the features.

        Args:
            data: The data to set for the feature.
        """
        if self.type == "numerical":
            self._data = np.array(data)
        else:
            self._data = data

    def __str__(self) -> str:
        """
        Returns a string representation of the feature.

        Returns:
            str: A string representation of the feature.
        """
        return f"Feature(name={self.name},type={self.type}, data={self._data})"
