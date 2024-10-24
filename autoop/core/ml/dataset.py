from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"):
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        
    def read(self) -> pd.DataFrame:
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))
    
    def save(self, data: pd.DataFrame) -> None:
        self.data = data.to_csv(index=False).encode()

    def split_data(self, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the tabular data into training and testing sets.
        """
        df = self.read()
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df