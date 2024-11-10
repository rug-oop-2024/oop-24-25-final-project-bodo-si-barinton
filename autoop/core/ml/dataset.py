import io
from typing import Any, Dict, List, Optional

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """
    A class representing a dataset artifact.
    """

    def __init__(
        self,
        asset_path: str,
        version: str = "1.0.0",
        name: str = "Unnamed",
        data: Optional[bytes] = None,
        artifact_type: str = "dataset",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """
        Initialize a Dataset artifact with the specified attributes.

        Args:
            asset_path (str): Path to the asset.
            version (str): Version of the dataset.
            name (str): Name of the dataset.
            data (Optional[bytes]): Data in bytes.
            artifact_type (str): Type of the artifact.
            tags (Optional[List[str]]): List of tags for the dataset.
            metadata (Optional[Dict[Any, Any]])
        """
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b"",
            type=artifact_type,
            tags=tags,
            metadata=metadata,
        )

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """
        Create a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to convert to a Dataset.
            name (str): Name of the dataset.
            asset_path (str): Path to the asset.
            version (str): Version of the dataset. Default is "1.0.0".

        Returns:
            Dataset: A new Dataset instance.
        """
        encoded_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=encoded_data,
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Read the dataset from bytes and return it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        bytes_data = super().read()
        csv = bytes_data.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> None:
        """
        Save the dataset from a pandas DataFrame to bytes.

        Args:
            data (pd.DataFrame): The DataFrame to save.
        """
        bytes_data = data.to_csv(index=False).encode()
        super().save(bytes_data)
