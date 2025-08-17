"""FastAPI application for OChem Helper."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OChem Helper API",
    description="Neural network API for molecular discovery and organic chemistry",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class GenerateRequest(BaseModel):
    n_molecules: int = 10
    target_properties: Optional[Dict[str, float]] = None


class PredictRequest(BaseModel):
    molecules: List[str]
    properties: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/api/generate")
async def generate_molecules(request: GenerateRequest):
    """Generate new molecules with desired properties."""
    try:
        # Placeholder for generation logic
        return {
            "molecules": [],
            "message": f"Generated {request.n_molecules} molecules"
        }
    except Exception as e:
        logger.error(f"Error in molecule generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict_properties(request: PredictRequest):
    """Predict properties for given molecules."""
    try:
        # Placeholder for prediction logic
        return {
            "predictions": {},
            "message": f"Predicted properties for {len(request.molecules)} molecules"
        }
    except Exception as e:
        logger.error(f"Error in property prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()