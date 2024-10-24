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
