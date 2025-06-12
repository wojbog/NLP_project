from model_utils import get_model
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


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


class ImportanceResponse(BaseModel):
    text: str
    tokens: list[str]
    importances: list[float]


@app.post("/predict/importance", response_model=ImportanceResponse)
async def predict_importance(request: PredictionRequest):
    try:
        logging.info(f"Received text: {request.text}")

        model = get_model()
        pairs = model.compute_importance(request.text)

        if not pairs:
            logging.error("Importance output is empty.")
            raise ValueError("No importance values returned.")

        tokens, scores = zip(*pairs)
        import numpy as np
        arr = np.array(scores)
        norm = arr
        norm[norm > 0] /= (np.abs(arr[arr > 0]).max() + 1e-12)
        norm[norm < 0] /= (np.abs(arr[arr < 0]).max() + 1e-12)

        logging.debug(f"Returning tokens: {tokens}")
        logging.debug(f"Returning importances: {norm.tolist()}")

        return ImportanceResponse(
            text=request.text,
            tokens=list(tokens),
            importances=norm.tolist()
        )
    except Exception as e:
        logging.exception("Exception in /predict/importance")
        raise HTTPException(
            status_code=500,
            detail=f"Importance calculation failed: {str(e)}")


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
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}")


@app.post("/predict/distribution",
          response_model=PredictionDistributionResponse)
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
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}")


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
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
