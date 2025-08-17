"""FastAPI application for OChem Helper."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import uuid
import json
import os

# Try to import PyTorch, fall back to mock if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module
    class MockTorch:
        def randn(*args, **kwargs):
            import numpy as np
            return np.random.randn(*args)
        def randn_like(x):
            import numpy as np
            return np.random.randn(*x.shape)
        def load(*args, **kwargs):
            return {}
    torch = MockTorch()

# Import our models - use mock if requested or torch not available
if os.environ.get('USE_MOCK_VAE') or not TORCH_AVAILABLE:
    from src.models.generative.mock_vae import MolecularVAE, SMILESTokenizer
else:
    from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer

from src.core.molecular_graph import MolecularGraph
from src.api.structure_converter import converter

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

# Global model storage
MODELS = {}
JOBS = {}


# Request/Response models
class GenerateRequest(BaseModel):
    n_molecules: int = Field(default=10, ge=1, le=1000)
    target_properties: Optional[Dict[str, float]] = None
    mode: str = Field(default="random", pattern="^(random|similar|interpolate)$")
    reference_smiles: Optional[str] = None
    variance: float = Field(default=0.1, ge=0, le=1)


class PredictRequest(BaseModel):
    molecules: List[str]
    properties: Optional[List[str]] = None


class RetrosynthesisRequest(BaseModel):
    target_smiles: str
    max_steps: int = Field(default=3, ge=1, le=10)
    

class OptimizeRequest(BaseModel):
    smiles: str
    objective: str = Field(default="QED", pattern="^(QED|logP|MW|SA)$")
    constraint: Optional[Dict[str, float]] = None
    n_steps: int = Field(default=10, ge=1, le=100)


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]


class GenerateResponse(BaseModel):
    molecules: List[str]
    properties: Optional[List[Dict[str, float]]]
    job_id: str
    message: str


class PredictResponse(BaseModel):
    predictions: Dict[str, List[float]]
    molecules: List[str]
    message: str


class StructureRequest(BaseModel):
    smiles: str
    format: str = Field(default="pdb", pattern="^(pdb|sdf|mol2|xyz)$")


class Structure3DResponse(BaseModel):
    structure: str
    format: str
    smiles: str
    
    
class StructureInfoResponse(BaseModel):
    atoms: List[Dict]
    bonds: List[Dict]
    smiles: str
    formula: str
    molecular_weight: float


# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load pretrained models on startup."""
    global MODELS
    
    # If using mock VAE or torch not available, use mock model
    if os.environ.get('USE_MOCK_VAE') or not TORCH_AVAILABLE:
        tokenizer = SMILESTokenizer()
        MODELS['vae'] = MolecularVAE(vocab_size=tokenizer.vocab_size)
        logger.info("Initialized mock VAE model (PyTorch not available or USE_MOCK_VAE set)")
        return
    
    # Check for VAE checkpoint
    vae_path = Path("models/checkpoints/best_model.pt")
    if vae_path.exists():
        try:
            checkpoint = torch.load(vae_path, map_location="cpu")
            config = checkpoint['model_config']
            
            model = MolecularVAE(
                vocab_size=config['vocab_size'],
                embedding_dim=config.get('embedding_dim', 128),
                hidden_dim=config['hidden_dim'],
                latent_dim=config['latent_dim'],
                num_layers=config.get('num_layers', 2),
                max_length=config['max_length'],
                beta=config.get('beta', 1.0)
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            MODELS['vae'] = model
            logger.info("Loaded VAE model")
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            # Fall back to mock
            tokenizer = SMILESTokenizer()
            MODELS['vae'] = MolecularVAE(vocab_size=tokenizer.vocab_size)
            logger.info("Initialized mock VAE model (failed to load real model)")
    else:
        # Initialize default model
        tokenizer = SMILESTokenizer()
        MODELS['vae'] = MolecularVAE(vocab_size=tokenizer.vocab_size)
        logger.info("Initialized default VAE model")


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=list(MODELS.keys())
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_molecules(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate new molecules with desired properties."""
    try:
        if 'vae' not in MODELS:
            raise HTTPException(status_code=503, detail="VAE model not loaded")
        
        model = MODELS['vae']
        job_id = str(uuid.uuid4())
        
        # Generate molecules
        if request.mode == "random":
            molecules = model.generate(request.n_molecules)
        elif request.mode == "similar" and request.reference_smiles:
            # Encode reference and generate similar
            z_ref = model.encode_smiles([request.reference_smiles])
            noise = torch.randn(request.n_molecules, z_ref.shape[1]) * request.variance
            z_samples = z_ref + noise
            molecules = model.generate(request.n_molecules, z=z_samples)
        else:
            molecules = model.generate(request.n_molecules)
        
        # Filter by target properties if specified
        if request.target_properties:
            molecules = filter_by_properties(molecules, request.target_properties)
        
        # Calculate properties for generated molecules
        properties = []
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                props = {
                    'MW': Descriptors.MolWt(mol),
                    'logP': Crippen.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'QED': Descriptors.qed(mol)
                }
                properties.append(props)
        
        # Store job result
        JOBS[job_id] = {
            'molecules': molecules,
            'properties': properties,
            'status': 'completed'
        }
        
        return GenerateResponse(
            molecules=molecules,
            properties=properties,
            job_id=job_id,
            message=f"Generated {len(molecules)} valid molecules"
        )
    
    except Exception as e:
        logger.error(f"Error in molecule generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictResponse)
async def predict_properties(request: PredictRequest):
    """Predict properties for given molecules."""
    try:
        predictions = {}
        valid_molecules = []
        
        # Basic property prediction using RDKit
        for smiles in request.molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            valid_molecules.append(smiles)
            
            # Calculate properties
            props = {
                'MW': [Descriptors.MolWt(mol)],
                'logP': [Crippen.MolLogP(mol)],
                'TPSA': [Descriptors.TPSA(mol)],
                'QED': [Descriptors.qed(mol)],
                'HBA': [Descriptors.NumHAcceptors(mol)],
                'HBD': [Descriptors.NumHDonors(mol)],
                'Rotatable': [Descriptors.NumRotatableBonds(mol)],
                'Rings': [Descriptors.RingCount(mol)]
            }
            
            # Add to predictions
            for key, value in props.items():
                if key not in predictions:
                    predictions[key] = []
                predictions[key].extend(value)
        
        return PredictResponse(
            predictions=predictions,
            molecules=valid_molecules,
            message=f"Predicted properties for {len(valid_molecules)} molecules"
        )
    
    except Exception as e:
        logger.error(f"Error in property prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def optimize_molecule(request: OptimizeRequest):
    """Optimize a molecule for a given objective."""
    try:
        if 'vae' not in MODELS:
            raise HTTPException(status_code=503, detail="VAE model not loaded")
        
        model = MODELS['vae']
        
        # Encode starting molecule
        z = model.encode_smiles([request.smiles])
        
        best_molecule = request.smiles
        best_score = calculate_objective(request.smiles, request.objective)
        
        # Optimization loop
        for step in range(request.n_steps):
            # Perturb latent vector
            z_perturbed = z + torch.randn_like(z) * 0.1
            
            # Generate new molecule
            candidates = model.generate(10, z=z_perturbed)
            
            for candidate in candidates:
                # Check constraints
                if request.constraint and not check_constraints(candidate, request.constraint):
                    continue
                
                # Calculate objective
                score = calculate_objective(candidate, request.objective)
                
                if score > best_score:
                    best_score = score
                    best_molecule = candidate
                    z = model.encode_smiles([best_molecule])
        
        return {
            "optimized_molecule": best_molecule,
            "score": best_score,
            "original_molecule": request.smiles,
            "objective": request.objective
        }
    
    except Exception as e:
        logger.error(f"Error in molecule optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get results for a specific job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JOBS[job_id]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=list(MODELS.keys())
    )


@app.post("/api/structure/convert", response_model=Structure3DResponse)
async def convert_structure(request: StructureRequest):
    """Convert SMILES to 3D structure in specified format."""
    try:
        structure = converter.smiles_to_3d(request.smiles, request.format)
        if structure is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES or conversion failed")
        
        return Structure3DResponse(
            structure=structure,
            format=request.format,
            smiles=request.smiles
        )
    except Exception as e:
        logger.error(f"Error in structure conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/structure/info", response_model=StructureInfoResponse)
async def get_structure_info(request: StructureRequest):
    """Get 3D structure information including atoms and bonds."""
    try:
        info = converter.get_3d_info(request.smiles)
        if "error" in info:
            raise HTTPException(status_code=400, detail=info["error"])
        
        return StructureInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting structure info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def filter_by_properties(molecules: List[str], target_properties: Dict[str, float]) -> List[str]:
    """Filter molecules by target properties."""
    filtered = []
    
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        match = True
        
        if 'MW' in target_properties:
            mw = Descriptors.MolWt(mol)
            if abs(mw - target_properties['MW']) / target_properties['MW'] > 0.2:
                match = False
        
        if 'logP' in target_properties:
            logp = Crippen.MolLogP(mol)
            if abs(logp - target_properties['logP']) > 1.0:
                match = False
        
        if 'TPSA' in target_properties:
            tpsa = Descriptors.TPSA(mol)
            if abs(tpsa - target_properties['TPSA']) > 20:
                match = False
        
        if match:
            filtered.append(smiles)
    
    return filtered


def calculate_objective(smiles: str, objective: str) -> float:
    """Calculate objective function value for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -float('inf')
    
    if objective == 'QED':
        return Descriptors.qed(mol)
    elif objective == 'logP':
        return Crippen.MolLogP(mol)
    elif objective == 'MW':
        return -abs(Descriptors.MolWt(mol) - 400)  # Target MW of 400
    elif objective == 'SA':
        # Simplified SA score
        n_rings = Descriptors.RingCount(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol))
        complexity = mol.GetNumAtoms() + n_rings * 2 + n_stereo * 3
        return -complexity / 10  # Lower is better
    else:
        return 0


def check_constraints(smiles: str, constraints: Dict[str, float]) -> bool:
    """Check if molecule satisfies constraints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    for prop, value in constraints.items():
        if prop == 'MW':
            if Descriptors.MolWt(mol) > value:
                return False
        elif prop == 'logP':
            if Crippen.MolLogP(mol) > value:
                return False
        elif prop == 'HBA':
            if Descriptors.NumHAcceptors(mol) > value:
                return False
        elif prop == 'HBD':
            if Descriptors.NumHDonors(mol) > value:
                return False
    
    return True


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()