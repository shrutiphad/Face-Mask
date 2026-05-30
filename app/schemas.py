from pydantic import BaseModel, Field
from typing import Optional


class EnrollResponse(BaseModel):
    
    identity_id: str
    enrolled_count: int
    message: str = "enrolled successfully"


class SearchMatch(BaseModel):
   
    identity_id: str
    score: float = Field(..., ge=-1.0, le=1.0)

class SearchResponse(BaseModel):
    
    query_file: str
    top_match: Optional[SearchMatch]
    is_match: bool
    threshold_used: float
    latency_ms: float
    enrolled_count: int


class ErrorResponse(BaseModel):
    
    detail: str