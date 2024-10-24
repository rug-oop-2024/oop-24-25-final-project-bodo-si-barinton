from pydantic import BaseModel
import base64
from typing import Optional

class Artifact(BaseModel):
    asset_path: str
    version: str
    name: Optional[str] = None
    data: Optional[bytes] = None
    type: Optional[str] = None

    @property
    def id(self) -> str:
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    class Config:
        underscore_attrs_are_private = True
