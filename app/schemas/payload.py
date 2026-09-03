from pydantic import BaseModel, Field
from typing import Optional

class AnalyzeRequest(BaseModel):
    content_text: str = Field(..., description="The text content to analyze")
    media_url: Optional[str] = Field(None, description="Optional URL to media (image/video)")
    source_platform: str = Field(..., description="Source platform (e.g., instagram_reel, twitter, web)")

class AnalysisData(BaseModel):
    is_misleading: bool
    confidence_score: float
    rating: str
    platform: str
    extracted_text: Optional[str] = None

class AnalyzeResponse(BaseModel):
    status: str = "success"
    data: AnalysisData
