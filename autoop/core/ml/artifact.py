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
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"


