from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Misinformation & Fake News Detector API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Model
    MODEL_PATH: str = "models/"
    
    # Microservice / Gateway
    COLAB_API_URL: str = "" # Leave blank if running locally without Colab
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
