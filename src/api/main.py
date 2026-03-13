
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # ← import here
from pydantic import BaseModel
from typing import Dict, Any
from src.api.predictor import PhishingPredictor
# FIRST define the app
app = FastAPI(title="Phishing URL Detector API")
predictor = PhishingPredictor()

# THEN add middleware (after app exists)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                    # for local testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class URLRequest(BaseModel):
    url: str


@app.post("/api/v1/detect", response_model=Dict[str, Any])
async def detect_phishing(request: URLRequest):
    url = request.url.strip()
    if len(url) < 8:
        raise HTTPException(status_code=400, detail="URL too short or invalid")

    if not any(url.startswith(s) for s in ["http://", "https://"]):
        url = "https://" + url

    try:
        result = predictor.predict(url)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    try:
        predictor._load_if_needed()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "reason": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )