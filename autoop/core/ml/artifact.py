from abc import ABC, abstractmethod
from pydantic import BaseModel
import base64
from typing import Optional

class Artifact(BaseModel, ABC):
    _asset_path: str
    _version: str
    name: Optional[str] = None
    data: Optional[bytes] = None
    type: Optional[str] = None

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
            "name": self.name,
            "type": self.type,
            "has_data": self.data is not None
        }

    @abstractmethod
    def read(self) -> Optional[bytes]:
        """Returns the stored data in bytes if available."""
        pass

    @abstractmethod
    def save(self, data: bytes) -> None:
        """Saves the provided data in the artifact."""
        pass

    class Config:
        underscore_attrs_are_private = True
