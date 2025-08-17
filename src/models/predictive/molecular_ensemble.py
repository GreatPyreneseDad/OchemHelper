"""
Molecular Property Ensemble Predictor
Advanced ensemble methods for molecular property prediction

Adapted from TraderAI's ML enhancements for chemical property prediction
using XGBoost, LightGBM, CatBoost, and neural networks.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

logger = logging.getLogger(__name__)

@dataclass
class MolecularEnsembleConfig:
    """Configuration for molecular property ensemble"""
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    use_neural_net: bool = True
    use_random_forest: bool = True
    
    # Model hyperparameters
    xgb_params: Dict = None
    lgb_params: Dict = None
    cb_params: Dict = None
    rf_params: Dict = None
    nn_params: Dict = None
    
    # Ensemble parameters
    ensemble_method: str = 'weighted_average'  # 'simple_average', 'weighted_average', 'stacking'
    uncertainty_estimation: bool = True
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if self.lgb_params is None:
            self.lgb_params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        
        if self.cb_params is None:
            self.cb_params = {
                'iterations': 1000,
                'depth': 8,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbose': False
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 500,
                'max_depth': 15,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if self.nn_params is None:
            self.nn_params = {
                'hidden_sizes': [512, 256, 128],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'epochs': 200,
                'batch_size': 64
            }

class MolecularDescriptorCalculator:
    """Calculate molecular descriptors for property prediction"""
    
    def __init__(self):
        self.descriptor_names = []
        self.scaler = StandardScaler()
        self.fitted = False
    
    def calculate_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """Calculate molecular descriptors from SMILES"""
        descriptors = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Fill with NaN for invalid SMILES
                descriptors.append([np.nan] * len(self.get_descriptor_names()))
                continue
            
            desc_values = self._calculate_single_molecule(mol)
            descriptors.append(desc_values)
        
        df = pd.DataFrame(descriptors, columns=self.get_descriptor_names())
        return df
    
    def _calculate_single_molecule(self, mol: Chem.Mol) -> List[float]:
        """Calculate descriptors for a single molecule"""
        descriptors = []
        
        # Basic molecular properties
        descriptors.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCsp3(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.NumValenceElectrons(mol)
        ])
        
        # Lipinski descriptors
        descriptors.extend([
            Descriptors.qed(mol),
            1 if self._passes_lipinski(mol) else 0,
            1 if self._passes_veber(mol) else 0,
            1 if self._passes_ghose(mol) else 0
        ])
        
        # Connectivity and shape descriptors
        descriptors.extend([
            Descriptors.BalabanJ(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            Descriptors.HallKierAlpha(mol)
        ])
        
        # Electronic descriptors
        descriptors.extend([
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol)
        ])
        
        # Surface area and volume
        descriptors.extend([
            Descriptors.LabuteASA(mol),
            Descriptors.PEOE_VSA1(mol),
            Descriptors.PEOE_VSA2(mol),
            Descriptors.PEOE_VSA3(mol),
            Descriptors.SMR_VSA1(mol),
            Descriptors.SMR_VSA2(mol),
            Descriptors.SMR_VSA3(mol)
        ])
        
        return descriptors
    
    def get_descriptor_names(self) -> List[str]:
        """Get list of descriptor names"""
        if not self.descriptor_names:
            self.descriptor_names = [
                'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
                'RingCount', 'FractionCsp3', 'NumHeteroatoms', 'NumRadicalElectrons',
                'NumValenceElectrons', 'QED', 'Lipinski', 'Veber', 'Ghose',
                'BalabanJ', 'Kappa1', 'Kappa2', 'Kappa3', 'Chi0', 'Chi1',
                'HallKierAlpha', 'MaxPartialCharge', 'MinPartialCharge',
                'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'LabuteASA',
                'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3'
            ]
        return self.descriptor_names
    
    def _passes_lipinski(self, mol: Chem.Mol) -> bool:
        """Check Lipinski's Rule of Five"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
        
        return violations <= 1
    
    def _passes_veber(self, mol: Chem.Mol) -> bool:
        """Check Veber's rules"""
        rotb = Descriptors.NumRotatableBonds(mol)
        psa = Descriptors.TPSA(mol)
        return rotb <= 10 and psa <= 140
    
    def _passes_ghose(self, mol: Chem.Mol) -> bool:
        """Check Ghose filter"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        atoms = mol.GetNumHeavyAtoms()
        mr = Descriptors.MolMR(mol)
        
        return (160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 
                20 <= atoms <= 70 and 40 <= mr <= 130)
    
    def fit_scaler(self, X: pd.DataFrame):
        """Fit scaler on training data"""
        X_clean = X.fillna(X.median())
        self.scaler.fit(X_clean)
        self.fitted = True
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        X_clean = X.fillna(X.median())
        return self.scaler.transform(X_clean)

class MolecularNeuralNetwork(nn.Module):
    """Neural network for molecular property prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MolecularPropertyEnsemble:
    """Ensemble predictor for molecular properties"""
    
    def __init__(self, config: MolecularEnsembleConfig = None):
        self.config = config or MolecularEnsembleConfig()
        self.models = {}
        self.model_weights = {}
        self.descriptor_calculator = MolecularDescriptorCalculator()
        self.fitted = False
        
        # Performance tracking
        self.model_performances = {}
        self.feature_importances = {}
        
        logger.info("Initialized Molecular Property Ensemble")
    
    def fit(self, 
            smiles_list: List[str], 
            target_values: List[float],
            property_name: str = 'property') -> Dict[str, float]:
        """
        Fit ensemble models on molecular data
        
        Args:
            smiles_list: List of SMILES strings
            target_values: Target property values
            property_name: Name of the property being predicted
            
        Returns:
            Dictionary of model performances
        """
        logger.info(f"Training ensemble for {property_name} with {len(smiles_list)} molecules")
        
        # Calculate molecular descriptors
        X = self.descriptor_calculator.calculate_descriptors(smiles_list)
        y = np.array(target_values)
        
        # Remove rows with invalid SMILES (all NaN)
        valid_mask = ~X.isnull().all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid molecules found")
        
        # Fit scaler
        self.descriptor_calculator.fit_scaler(X)
        X_scaled = self.descriptor_calculator.transform(X)
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        performances = {}
        
        # Train XGBoost
        if self.config.use_xgboost and XGBOOST_AVAILABLE:
            self.models['xgboost'] = self._train_xgboost(X_train, y_train, X_val, y_val)
            performances['xgboost'] = self._evaluate_model('xgboost', X_val, y_val)
        
        # Train LightGBM
        if self.config.use_lightgbm and LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = self._train_lightgbm(X_train, y_train, X_val, y_val)
            performances['lightgbm'] = self._evaluate_model('lightgbm', X_val, y_val)
        
        # Train CatBoost
        if self.config.use_catboost and CATBOOST_AVAILABLE:
            self.models['catboost'] = self._train_catboost(X_train, y_train, X_val, y_val)
            performances['catboost'] = self._evaluate_model('catboost', X_val, y_val)
        
        # Train Random Forest
        if self.config.use_random_forest:
            self.models['random_forest'] = self._train_random_forest(X_train, y_train)
            performances['random_forest'] = self._evaluate_model('random_forest', X_val, y_val)
        
        # Train Neural Network
        if self.config.use_neural_net:
            self.models['neural_net'] = self._train_neural_network(X_train, y_train, X_val, y_val)
            performances['neural_net'] = self._evaluate_model('neural_net', X_val, y_val)
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(performances)
        
        self.model_performances = performances
        self.fitted = True
        
        logger.info(f"Ensemble training completed. Best model: {max(performances, key=lambda k: performances[k]['r2'])}")
        
        return performances
    
    def predict(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict properties for molecules
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Calculate descriptors
        X = self.descriptor_calculator.calculate_descriptors(smiles_list)
        
        # Handle invalid SMILES
        valid_mask = ~X.isnull().all(axis=1)
        X_scaled = np.full((len(smiles_list), len(self.descriptor_calculator.get_descriptor_names())), np.nan)
        
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_scaled_valid = self.descriptor_calculator.transform(X_valid)
            X_scaled[valid_mask] = X_scaled_valid
        
        # Get predictions from each model
        model_predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'neural_net':
                preds = self._predict_neural_network(model, X_scaled)
            else:
                preds = np.full(len(smiles_list), np.nan)
                if valid_mask.sum() > 0:
                    preds[valid_mask] = model.predict(X_scaled_valid)
            
            model_predictions[model_name] = preds
        
        # Ensemble predictions
        ensemble_pred, ensemble_uncertainty = self._ensemble_predict(model_predictions, valid_mask)
        
        return ensemble_pred, ensemble_uncertainty
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(**self.config.xgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importances['xgboost'] = model.feature_importances_
        
        return model
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        model = lgb.LGBMRegressor(**self.config.lgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Store feature importance
        self.feature_importances['lightgbm'] = model.feature_importances_
        
        return model
    
    def _train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        model = cb.CatBoostRegressor(**self.config.cb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importances['catboost'] = model.feature_importances_
        
        return model
    
    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        model = RandomForestRegressor(**self.config.rf_params)
        model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importances['random_forest'] = model.feature_importances_
        
        return model
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network model"""
        input_size = X_train.shape[1]
        model = MolecularNeuralNetwork(
            input_size=input_size,
            hidden_sizes=self.config.nn_params['hidden_sizes'],
            dropout=self.config.nn_params['dropout']
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.nn_params['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.config.nn_params['epochs']):
            model.train()
            optimizer.zero_grad()
            
            predictions = model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return model
    
    def _predict_neural_network(self, model, X):
        """Predict using neural network"""
        model.eval()
        predictions = np.full(len(X), np.nan)
        
        valid_mask = ~np.isnan(X).any(axis=1)
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_tensor = torch.FloatTensor(X_valid)
            
            with torch.no_grad():
                pred_tensor = model(X_tensor)
                predictions[valid_mask] = pred_tensor.squeeze().numpy()
        
        return predictions
    
    def _evaluate_model(self, model_name: str, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        model = self.models[model_name]
        
        if model_name == 'neural_net':
            y_pred = self._predict_neural_network(model, X_val)
        else:
            y_pred = model.predict(X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def _calculate_ensemble_weights(self, performances: Dict[str, Dict[str, float]]):
        """Calculate ensemble weights based on model performance"""
        if self.config.ensemble_method == 'simple_average':
            # Equal weights
            n_models = len(performances)
            self.model_weights = {name: 1.0 / n_models for name in performances.keys()}
        
        elif self.config.ensemble_method == 'weighted_average':
            # Weight by R² score
            r2_scores = {name: perf['r2'] for name, perf in performances.items()}
            
            # Ensure non-negative weights
            min_r2 = min(r2_scores.values())
            if min_r2 < 0:
                r2_scores = {name: score - min_r2 + 0.01 for name, score in r2_scores.items()}
            
            # Normalize weights
            total_weight = sum(r2_scores.values())
            self.model_weights = {name: score / total_weight for name, score in r2_scores.items()}
        
        logger.info(f"Ensemble weights: {self.model_weights}")
    
    def _ensemble_predict(self, model_predictions: Dict[str, np.ndarray], valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Combine model predictions into ensemble prediction"""
        n_samples = len(next(iter(model_predictions.values())))
        ensemble_pred = np.full(n_samples, np.nan)
        ensemble_uncertainty = np.full(n_samples, np.nan)
        
        if valid_mask.sum() == 0:
            return ensemble_pred, ensemble_uncertainty
        
        # Get predictions for valid samples
        valid_predictions = {}
        for model_name, preds in model_predictions.items():
            if model_name in self.model_weights:
                valid_predictions[model_name] = preds[valid_mask]
        
        if not valid_predictions:
            return ensemble_pred, ensemble_uncertainty
        
        # Weighted average
        weighted_preds = np.zeros(valid_mask.sum())
        total_weight = 0
        
        for model_name, preds in valid_predictions.items():
            weight = self.model_weights[model_name]
            weighted_preds += weight * preds
            total_weight += weight
        
        if total_weight > 0:
            weighted_preds /= total_weight
            ensemble_pred[valid_mask] = weighted_preds
        
        # Calculate uncertainty as std of model predictions
        if self.config.uncertainty_estimation and len(valid_predictions) > 1:
            pred_matrix = np.array(list(valid_predictions.values()))
            uncertainty = np.std(pred_matrix, axis=0)
            ensemble_uncertainty[valid_mask] = uncertainty
        
        return ensemble_pred, ensemble_uncertainty
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Get feature importance for each model"""
        if not self.fitted:
            raise ValueError("Ensemble not fitted")
        
        importances = {}
        descriptor_names = self.descriptor_calculator.get_descriptor_names()
        
        for model_name, importance_scores in self.feature_importances.items():
            # Create list of (feature_name, importance) tuples
            feature_importance = list(zip(descriptor_names, importance_scores))
            # Sort by importance descending
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            # Take top N
            importances[model_name] = feature_importance[:top_n]
        
        return importances
    
    def save_model(self, filepath: str):
        """Save trained ensemble model"""
        model_data = {
            'config': self.config,
            'model_weights': self.model_weights,
            'model_performances': self.model_performances,
            'feature_importances': self.feature_importances,
            'descriptor_calculator': self.descriptor_calculator,
            'fitted': self.fitted
        }
        
        # Save sklearn-compatible models
        sklearn_models = {}
        for name, model in self.models.items():
            if name != 'neural_net':
                sklearn_models[name] = model
        
        model_data['sklearn_models'] = sklearn_models
        
        # Save neural network separately if exists
        if 'neural_net' in self.models:
            torch.save(self.models['neural_net'].state_dict(), f"{filepath}_neural_net.pt")
            model_data['neural_net_architecture'] = {
                'input_size': len(self.descriptor_calculator.get_descriptor_names()),
                'hidden_sizes': self.config.nn_params['hidden_sizes'],
                'dropout': self.config.nn_params['dropout']
            }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained ensemble model"""
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.model_weights = model_data['model_weights']
        self.model_performances = model_data['model_performances']
        self.feature_importances = model_data['feature_importances']
        self.descriptor_calculator = model_data['descriptor_calculator']
        self.fitted = model_data['fitted']
        
        # Load sklearn models
        self.models = model_data['sklearn_models']
        
        # Load neural network if exists
        if 'neural_net_architecture' in model_data:
            arch = model_data['neural_net_architecture']
            neural_net = MolecularNeuralNetwork(
                arch['input_size'], arch['hidden_sizes'], arch['dropout']
            )
            neural_net.load_state_dict(torch.load(f"{filepath}_neural_net.pt"))
            neural_net.eval()
            self.models['neural_net'] = neural_net
        
        logger.info(f"Ensemble model loaded from {filepath}")

def create_molecular_ensemble(config: Optional[MolecularEnsembleConfig] = None) -> MolecularPropertyEnsemble:
    """Factory function to create molecular property ensemble"""
    if config is None:
        config = MolecularEnsembleConfig()
    
    return MolecularPropertyEnsemble(config)
