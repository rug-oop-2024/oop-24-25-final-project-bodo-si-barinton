from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr, Field
import base64
from typing import Optional

class Artifact(BaseModel, ABC):
    _asset_path: str = PrivateAttr()
    _version: str = PrivateAttr()
    _name: str = PrivateAttr()
    _data: bytes = PrivateAttr()
    _type: str = PrivateAttr()

    @property
    def id(self) -> str:
        encoded_path = base64.b64encode(self._asset_path.encode()).decode()
        return f"{encoded_path}:{self._version}"
    
    @property
    def details(self) -> dict:
        """Returns the details of the artifact as a dictionary."""
        return {
            "asset_path": self._asset_path,
            "version": self._version,
            "name": self._name,
            "type": self._type,
            "has_data": self._data is not None
        }

    @abstractmethod
    def read(self) -> Optional[bytes]:
        """Returns the stored data in bytes if available."""
        pass

    @abstractmethod
    def save(self, data: bytes) -> None:
        """Saves the provided data in the artifact."""
        pass

