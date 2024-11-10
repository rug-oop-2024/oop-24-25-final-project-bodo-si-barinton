import base64
from abc import ABC
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, PrivateAttr


class Artifact(BaseModel, ABC):
    """
    A base class representing an artifact.
    """

    _asset_path: str = PrivateAttr(default="path")
    _version: str = PrivateAttr(default="1.0")
    _name: str = PrivateAttr()
    _data: bytes = PrivateAttr()
    _type: str = PrivateAttr()
    _tags: List[str] = PrivateAttr(default_factory=list)
    _metadata: Dict[Any, Any] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        name: str,
        data: bytes,
        type: str = "GenericType",
        asset_path: Optional[str] = "path",
        version: Optional[str] = "1.0",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):
        """
        Initialize an Artifact instance.

        Args:
            name (str): The name of the artifact.
            data (bytes): The data associated with the artifact.
            type (str): The type of the artifact.
            asset_path (Optional[str]): The path to the asset.
            version (Optional[str]): The version of the artifact.
            tags (Optional[List[str]]):tags associated with the artifact.
            metadata (Optional[Dict[Any, Any]])
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._asset_path = asset_path if asset_path is not None else "path"
        self._version = (
            version.replace(";", "_")
            .replace(".", "_")
            .replace(",", "_")
            .replace("=", "_")
        )
        self._name = name
        self._data = data
        self._type = type
        self._tags = tags if tags is not None else []
        self._metadata = metadata if metadata is not None else {}

    @property
    def asset_path(self) -> str:
        """
        Get the asset path of the artifact.

        Returns:
            str: The asset path.
        """
        return self._asset_path

    @property
    def version(self) -> str:
        """
        Get the version of the artifact.

        Returns:
            str: The version.
        """
        return self._version

    @property
    def name(self) -> str:
        """
        Get the name of the artifact.

        Returns:
            str: The name.
        """
        return self._name

    @property
    def data(self) -> bytes:
        """
        Get the data of the artifact.

        Returns:
            bytes: The data.
        """
        return self._data

    @property
    def type(self) -> str:
        """
        Get the type of the artifact.

        Returns:
            str: The type.
        """
        return self._type

    @property
    def tags(self) -> List[str]:
        """
        Get the tags associated with the artifact.

        Returns:
            List[str]: The tags.
        """
        return self._tags

    @property
    def metadata(self) -> Dict[Any, Any]:
        """
        Get the metadata of the artifact.

        Returns:
            Dict[Any, Any]: The metadata.
        """
        return self._metadata

    @property
    def id(self) -> str:
        """
        Get the unique identifier of the artifact.

        Returns:
            str: The unique identifier.
        """
        encoded_path = base64.b64encode(
            self._asset_path.encode()
            ).decode().rstrip("=")
        return f"{encoded_path}_{self._version}"

    @property
    def details(self) -> dict:
        """
        Get the details of the artifact as a dictionary.

        Returns:
            dict: The details of the artifact.
        """
        return {
            "asset_path": self._asset_path,
            "version": self._version,
            "name": self._name,
            "type": self._type,
            "has_data": self._data is not None,
            "tags": self._tags,
        }

    def read(self) -> Optional[bytes]:
        """
        Returns the stored data in bytes if available.

        Returns:
            Optional[bytes]: The stored data or None if not available.
        """
        return self._data

    def save(self, data: bytes) -> None:
        """
        Saves the provided data in the artifact.

        Args:
            data (bytes): The data to save.
        """
        self._data = data

    def get(self, attribute_type: str) -> Optional[Any]:
        """
        Retrieve an attribute by type if it exists.

        Args:
            attribute_type (str): The type of the attribute to retrieve.

        Returns:
            Optional[Any]: The attribute value or None if it does not exist.
        """
        return getattr(self, attribute_type, None)
