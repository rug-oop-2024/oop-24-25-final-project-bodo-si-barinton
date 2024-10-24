from pydantic import BaseModel
import base64
from typing import Optional

class Artifact(BaseModel):
    _asset_path: str
    _version: str
    name: Optional[str] = None
    data: Optional[bytes] = None
    type: Optional[str] = None

    @property
    def id(self) -> str:
        encoded_path = base64.b64encode(self._asset_path.encode()).decode()
        return f"{encoded_path}:{self._version}"

    def read(self) -> Optional[bytes]:
        """Returns the stored data in bytes if available."""
        if self.data is None:
            raise ValueError("No data available in this artifact.")
        return self.data

    def save(self, data: bytes) -> None:
        """Saves the provided data in the artifact."""
        self.data = data

    class Config:
        underscore_attrs_are_private = True
