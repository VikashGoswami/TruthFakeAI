# Google Colab GPU Inference Server

This notebook runs your heavy AI workloads (Video processing, Image OCR, and Fake News Detection) on a free NVIDIA GPU.

## Instructions:
1. Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Go to **Runtime -> Change runtime type** and select **T4 GPU**.
3. Create a free account on [Ngrok](https://ngrok.com/) and get your Auth Token.
4. Copy and paste the code below into a single Colab cell and run it.
5. Copy the `ngrok.io` public URL it prints out, and add it to your Hugging Face Space environment variables as `COLAB_API_URL`.

## The Code

```python
!pip install fastapi uvicorn pyngrok transformers torch yt-dlp openai-whisper pytesseract pillow nest-asyncio
!apt-get install tesseract-ocr -y

import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import whisper
from transformers import pipeline
import yt_dlp
import os
import pytesseract
from PIL import Image
import requests
from io import BytesIO

# 1. Initialize AI Models (This takes a minute to download weights)
print("Loading Whisper Model...")
whisper_model = whisper.load_model("base") # Use 'small' or 'base' for speed

print("Loading Fake News Model...")
# Using a public model that doesn't require authentication
fake_news_model = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")

app = FastAPI()

class AnalyzeRequest(BaseModel):
    content_text: Optional[str] = None
    media_url: Optional[str] = None
    source_platform: str

def extract_text_from_video(video_url: str) -> str:
    """Downloads audio from a video URL and transcribes it."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    # Transcribe
    result = whisper_model.transcribe("temp_audio.mp3")
    os.remove("temp_audio.mp3")
    return result["text"]

def extract_text_from_image(image_url: str) -> str:
    """Downloads an image and performs OCR to extract text."""
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    return text

@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    text_to_analyze = payload.content_text or ""
    
    # Handle Media
    if payload.media_url:
        try:
            if payload.source_platform in ['instagram_reel', 'youtube', 'tiktok', 'video']:
                print(f"Extracting audio from {payload.media_url}")
                extracted = extract_text_from_video(payload.media_url)
                text_to_analyze += " " + extracted
            elif payload.source_platform in ['image', 'meme']:
                print(f"Running OCR on {payload.media_url}")
                extracted = extract_text_from_image(payload.media_url)
                text_to_analyze += " " + extracted
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Media processing failed: {str(e)}")

    if not text_to_analyze.strip():
        raise HTTPException(status_code=400, detail="No text or analyzable media found.")

    # Run Fake News Inference
    result = fake_news_model(text_to_analyze[:512], truncation=True, max_length=512)
    label = result[0]['label'].upper()
    score = result[0]['score']
    
    is_misleading = label in ['LABEL_1', 'FAKE', 'MISLEADING', '0'] # Adjust based on model output
    
    return {
        "status": "success",
        "data": {
            "is_misleading": is_misleading,
            "confidence_score": round(score, 4),
            "rating": "Warning: Potential Misinformation" if is_misleading else "Likely Authentic",
            "extracted_text": text_to_analyze.strip()
        }
    }

# 2. Start Ngrok Tunnel
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN_HERE") # REPLACE THIS!
public_url = ngrok.connect(8000).public_url
print(f"\n\n🌟 YOUR COLAB API URL IS: {public_url} 🌟\n\n")

# 3. Run Server
nest_asyncio.apply()
uvicorn.run(app, port=8000)
```
