import json
import os
from typing import Dict, List, Optional, Tuple

from autoop.core.storage import Storage


class Database:
    """
    A key-value database implementation with collection-based storage.
    """

    def __init__(self, storage: Storage) -> None:
        """
        Initialize the Database with a storage backend.

        Args:
            storage (Storage): The storage backend to use for persistence.
        """
        self._storage: Storage = storage
        self._data: Dict[str, Dict[str, dict]] = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """
        Set a value in the database under the specified collection and ID.

        Args:
            collection: The collection to store the data in.
            id: The unique identifier for the data.
            entry: The data to store (must be a dictionary).

        Returns:
            dict: The stored data entry.

        Raises:
            AssertionError: If entry is not a dictionary or if collection/id
                not strings.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"

        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Optional[dict]:
        """
        Retrieve a value from the database by collection and ID.

        Args:
            collection: The collection to get the data from.
            id: The unique identifier of the data.

        Returns:
            Optional[dict]: The stored data or None if not found.
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """
        Delete a value from the database by collection and ID.

        Args:
            collection: The collection to delete the data from.
            id: The unique identifier of the data to delete.
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """
        List all items in a collection.

        Args:
            collection: The collection to list items from.

        Returns:
            List[Tuple[str, dict]]: List of (id, data) pairs from the collection.
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """Reload all data from storage, discarding any in-memory changes."""
        self._load()

    def _persist(self) -> None:
        """
        Persist all in-memory data to storage.

        Saves all collections and items to the storage backend and removes any
        items that were deleted.
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}{os.sep}{id}"
                )

        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split(os.sep)[-2:]
            if not self._data.get(collection, id):
                self._storage.delete(f"{collection}{os.sep}{id}")

    def _load(self) -> None:
        """
        Load all data from storage into memory.

        Reads all collections and items from the storage backend and populates
        the in-memory data structure.
        """
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{id}")
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
