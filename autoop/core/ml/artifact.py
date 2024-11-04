from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr
import base64
from typing import Optional, List, Dict, Any

class Artifact(BaseModel, ABC):
    _asset_path: str = PrivateAttr()
    _version: str = PrivateAttr()
    _name: str = PrivateAttr()
    _data: bytes = PrivateAttr()
    _type: str = PrivateAttr()
    _tags : List[str] = PrivateAttr()
    _metadata : Dict [Any, Any ] = PrivateAttr()

    def __init__(self, asset_path: str, version: str, name: str, data: bytes, type: str,
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[Any, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._asset_path = asset_path
        self._version = version
        self._name = name
        self._data = data
        self._type = type
        self._tags = tags if tags is not None else []
        self._metadata = metadata if metadata is not None else {}

    @property
    def asset_path(self) -> str:
        return self._asset_path

    @property
    def version(self) -> str:
        return self._version

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def type(self) -> str:
        return self._type

    @property
    def tags(self) -> List[str]:
        return self._tags
    
    @property
    def metadata(self) -> Dict[Any, Any]:
        return self._metadata

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
            "has_data": self._data is not None,
            "tags" : self._tags
        }

   
    def read(self) -> Optional[bytes]:
        """Returns the stored data in bytes if available."""
        return self._data

    
    def save(self, data: bytes) -> None:
        """Saves the provided data in the artifact."""
        self._data = data