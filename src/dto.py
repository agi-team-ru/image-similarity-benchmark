from typing import Any, Dict, List
from pydantic import BaseModel


class ImgSimilarityRequest(BaseModel):
    images: List[str]
    options: Dict[str, Any]


class ImgSimilarityResponse(BaseModel):
    score: float

