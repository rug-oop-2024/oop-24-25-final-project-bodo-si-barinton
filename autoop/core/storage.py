import os
from abc import ABC, abstractmethod
from glob import glob
from typing import List


class NotFoundError(Exception):
    """
    Exception raised when a path is not found.
    """

    def __init__(self, path :str) -> None:
        """
        Initialize the NotFoundError instance.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for storage backends.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Local storage backend implementation.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize the LocalStorage instance.

        Args:
            base_path (str): The base path for storage. Default is "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a given key.

        Args:
            data (bytes): Data to save.
            key (str): Key to save data.
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a given key.

        Args:
            key (str): Key to load data.

        Returns:
            bytes: Loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a given key.

        Args:
            key (str): Key to delete data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all keys under a given prefix.

        Args:
            prefix (str): Prefix to list keys.

        Returns:
            List[str]: List of keys.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Assert that a path exists.

        Args:
            path (str): Path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with a given path.

        Args:
            path (str): Path to join.

        Returns:
            str: The joined path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
