from typing import List, Optional

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage


class ArtifactRegistry:
    """
    A registry for managing artifacts.
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initialize the ArtifactRegistry instance.

        Args:
            database (Database): The database to use.
            storage (Storage): The storage backend to use.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact in the registry.

        Args:
            artifact (Artifact): The artifact to register.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: Optional[str] = None) -> List[Artifact]:
        """
        List all artifacts in the registry.

        Args:
            type (Optional[str]): The type of artifacts to list.

        Returns:
            List[Artifact]: A list of artifacts.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Get an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact.

        Returns:
            Artifact: The requested artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class for managing the AutoML system.
    """

    _instance: Optional["AutoMLSystem"] = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the AutoMLSystem instance.

        Args:
            storage (LocalStorage): The storage backend to use.
            database (Database): The database to use.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Get the singleton instance of the AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Get the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry.
        """
        return self._registry
