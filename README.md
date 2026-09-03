# Misinformation & Fake News Detector API

A robust, scalable, and stateless FastAPI backend for detecting misinformation and fake news.

## Features
- FastAPI based asynchronous architecture
- Local RoBERTa model integration for text classification
- Fallback mock responses when model weights are missing
- Clean, modular directory structure
- Pydantic v2 data validation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your fine-tuned `roberta-base` weights to the `models/` directory.

3. Run the application:
```bash
uvicorn app.main:app --reload
```

## Docker

Build and run with Docker:

```bash
docker build -t misinformation-api .
docker run -p 8000:8000 misinformation-api
```
