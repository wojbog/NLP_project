from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="NLP Classification Service", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_class: str
    probabilities: dict[str, float]

class PredictionDistributionResponse(BaseModel):
    text: str
    probabilities: dict[str, float]

from model_utils import  get_model

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint to classify text using the NLP model
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        model = get_model()
        

        result = model.predict(request.text)
        predicted_class = result['predicted_class']
        probabilities = result['distribution']
        
        return PredictionResponse(
            text=request.text,
            predicted_class=predicted_class,
            probabilities=probabilities,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/distribution", response_model=PredictionDistributionResponse)
async def predict_distribution(request: PredictionRequest):
    """
    Endpoint to classify text with confidence scores
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        model = get_model()
        probabilities = model.predict_distribution(request.text)
        
        return PredictionDistributionResponse(
            text=request.text,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model = get_model()
        return {
            "model_type": "BERT",
            "classes": list(model.id2label.values()),
            "num_classes": len(model.id2label)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
