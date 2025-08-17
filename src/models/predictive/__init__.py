"""Enhanced predictive models for molecular properties with ensemble methods."""

from .molecular_ensemble import (
    MolecularPropertyEnsemble,
    MolecularEnsembleConfig, 
    MolecularDescriptorCalculator,
    MolecularNeuralNetwork,
    create_molecular_ensemble
)

import logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Some ensemble methods will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network models will be limited.")

class PropertyPredictor:
    """Enhanced property predictor with ensemble methods"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ensemble = None
        self.descriptor_calculator = None
        self.fitted = False
        
        # Initialize components
        try:
            ensemble_config = MolecularEnsembleConfig(
                use_xgboost=True,
                use_lightgbm=True,
                use_catboost=True,
                use_neural_net=TORCH_AVAILABLE,
                use_random_forest=SKLEARN_AVAILABLE,
                uncertainty_estimation=True
            )
            self.ensemble = MolecularPropertyEnsemble(ensemble_config)
            self.descriptor_calculator = MolecularDescriptorCalculator()
            logger.info("Property predictor ensemble initialized")
            
        except Exception as e:
            logger.error(f"Error initializing property predictor: {e}")
    
    @classmethod
    def from_pretrained(cls, model_name: str = 'default') -> "PropertyPredictor":
        """Load a pretrained property predictor"""
        instance = cls()
        
        from pathlib import Path
        model_path = Path(f'models/pretrained/{model_name}_ensemble.pkl')
        
        if model_path.exists():
            try:
                instance.ensemble.load_model(str(model_path))
                instance.fitted = True
                logger.info(f"Loaded pretrained ensemble: {model_name}")
            except Exception as e:
                logger.error(f"Error loading pretrained ensemble: {e}")
        
        return instance
    
    def fit(self, 
            smiles_list: list,
            property_values: list,
            property_name: str = 'property') -> dict:
        """Train the ensemble on molecular data"""
        if not self.ensemble:
            raise RuntimeError("Ensemble not initialized")
        
        try:
            performances = self.ensemble.fit(smiles_list, property_values, property_name)
            self.fitted = True
            logger.info(f"Ensemble trained for {property_name}")
            return performances
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {}
    
    def predict(self, molecules: list) -> dict:
        """Predict properties for molecules"""
        if not self.fitted:
            return self._fallback_prediction(molecules)
        
        try:
            predictions, uncertainties = self.ensemble.predict(molecules)
            
            return {
                'molecules': molecules,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'uncertainties': uncertainties.tolist() if hasattr(uncertainties, 'tolist') else uncertainties,
                'model_type': 'ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return self._fallback_prediction(molecules)
    
    def _fallback_prediction(self, molecules: list) -> dict:
        """Fallback prediction using simple descriptors"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen
            
            predictions = []
            uncertainties = []
            
            for smiles in molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Simple property estimation
                    mw = Descriptors.MolWt(mol)
                    logp = Crippen.MolLogP(mol)
                    
                    # Placeholder prediction based on descriptors
                    pred = (mw / 500.0) * (logp + 5) / 10.0
                    predictions.append(pred)
                    uncertainties.append(0.5)  # High uncertainty for fallback
                else:
                    predictions.append(0.0)
                    uncertainties.append(1.0)
            
            return {
                'molecules': molecules,
                'predictions': predictions,
                'uncertainties': uncertainties,
                'model_type': 'fallback'
            }
            
        except ImportError:
            # Ultimate fallback
            return {
                'molecules': molecules,
                'predictions': [0.5] * len(molecules),
                'uncertainties': [1.0] * len(molecules),
                'model_type': 'random'
            }
    
    def get_feature_importance(self, top_n: int = 20) -> dict:
        """Get feature importance from trained models"""
        if not self.fitted or not self.ensemble:
            return {}
        
        try:
            return self.ensemble.get_feature_importance(top_n)
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.fitted or not self.ensemble:
            raise RuntimeError("No trained model to save")
        
        try:
            self.ensemble.save_model(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not self.ensemble:
            raise RuntimeError("Ensemble not initialized")
        
        try:
            self.ensemble.load_model(filepath)
            self.fitted = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

__all__ = [
    "MolecularPropertyEnsemble",
    "MolecularEnsembleConfig",
    "MolecularDescriptorCalculator", 
    "MolecularNeuralNetwork",
    "PropertyPredictor",
    "create_molecular_ensemble"
]
