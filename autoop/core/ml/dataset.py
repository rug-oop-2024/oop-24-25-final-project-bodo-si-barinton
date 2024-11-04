from autoop.core.ml.artifact import Artifact
from abc import ABC
import pandas as pd
import io
from typing import Optional, List, Dict, Any


class Dataset(Artifact):
    def __init__(self, asset_path: str, version: str = "1.0.0", name: str = "Unnamed", data: Optional[bytes] = None, artifact_type: str ="dataset", 
                 tags: Optional[List[str]] = None, metadata: Optional[Dict[Any, Any]] = None):
        """
        Initialize a Dataset artifact with the specified attributes.
        
        Args:
            asset_path (str): Path to the asset.
            version (str): Version of the dataset.
            name (str): Name of the dataset.
            data (Optional[bytes]): Data in bytes.
            tags (Optional[List[str]]): List of tags for the dataset.
            metadata (Optional[Dict[Any, Any]]): Additional metadata for the dataset.
        """
        super().__init__(
            asset_path=asset_path,
            version=version,
            name=name,
            data=data if data is not None else b'',
            type=artifact_type,
            tags=tags,
            metadata=metadata
        )

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"):
        encoded_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=encoded_data,
            version=version,
        )
        
    def read(self) -> pd.DataFrame:
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)

