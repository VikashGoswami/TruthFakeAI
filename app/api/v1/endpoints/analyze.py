import requests
from fastapi import APIRouter, HTTPException
from app.schemas.payload import AnalyzeRequest, AnalyzeResponse, AnalysisData
from app.services.ai_pipeline import ai_pipeline
from app.config import settings

router = APIRouter()

@router.post("/", response_model=AnalyzeResponse)
async def analyze_content(payload: AnalyzeRequest):
    try:
        # 1. API GATEWAY MODE (Forward to Colab if configured)
        if settings.COLAB_API_URL:
            # Send the request to the Colab GPU server
            response = requests.post(
                f"{settings.COLAB_API_URL.rstrip('/')}/analyze", 
                json=payload.model_dump()
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                response_data = AnalysisData(
                    is_misleading=data.get("is_misleading", True),
                    confidence_score=data.get("confidence_score", 0.0),
                    rating=data.get("rating", "Error"),
                    platform=payload.source_platform,
                    extracted_text=data.get("extracted_text", "")
                )
                return AnalyzeResponse(status="success", data=response_data)
            else:
                raise HTTPException(status_code=502, detail="Colab GPU Server returned an error")
                
        # 2. LOCAL FALLBACK MODE (If Colab is off, run local text analysis)
        if payload.media_url:
            raise HTTPException(
                status_code=400, 
                detail="Media processing requires the Colab GPU server to be running. Please set COLAB_API_URL."
            )
            
        result = ai_pipeline.analyze(payload.content_text)
        
        response_data = AnalysisData(
            is_misleading=result["is_misleading"],
            confidence_score=result["confidence_score"],
            rating=result["rating"],
            platform=payload.source_platform,
            extracted_text=payload.content_text
        )
        
        return AnalyzeResponse(status="success", data=response_data)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Colab Server: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
