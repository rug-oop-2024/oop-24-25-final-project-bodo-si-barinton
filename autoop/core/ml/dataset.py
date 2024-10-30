from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

class Dataset(Artifact):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._asset_path = kwargs.get('asset_path', '')
        self._version = kwargs.get('version', '1.0.0')
        self._name = kwargs.get('name', 'Unnamed')
        self._data = kwargs.get('data', None)
        self._type = 'dataset'
        self._tags = None
        self._metadata = None

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
        if self._data is None:
            raise ValueError("Data is not initialized.")
        csv = self._data.decode()
        return pd.read_csv(io.StringIO(csv))
    
    def save(self, data: pd.DataFrame) -> None:
        self._data = data.to_csv(index=False).encode()

