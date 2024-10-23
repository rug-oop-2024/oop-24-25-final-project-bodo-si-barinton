from pydantic import BaseModel, Field
import base64
from typing import Dict, Any, List

class Artifact(BaseModel):
    _name : str
    _asset_path : str
    _version : str
    _data : bytes
    _metadata: Dict[str, Any] = None
    _ttype : str
    _tags : List[str] = None
