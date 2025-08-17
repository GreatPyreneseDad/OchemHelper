"""Unit tests for FastAPI application."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import json

# Import the app
import sys
sys.path.insert(0, '/Users/chris/ochem-helper/src')

from api.app import app


class TestHealthEndpoints(unittest.TestCase):
    """Test health check endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models", data)
        self.assertIn("version", data)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)


class TestGenerationEndpoints(unittest.TestCase):
    """Test molecule generation endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('api.app.generator')
    def test_generate_molecules_basic(self, mock_generator):
        """Test basic molecule generation."""
        # Mock the generator
        mock_generator.generate.return_value = ["CCO", "CC(=O)O", "c1ccccc1"]
        
        response = self.client.post(
            "/api/v1/generate",
            json={"n_molecules": 3}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("molecules", data)
        self.assertEqual(len(data["molecules"]), 3)
        mock_generator.generate.assert_called_once()
    
    @patch('api.app.generator')
    def test_generate_molecules_with_properties(self, mock_generator):
        """Test generation with target properties."""
        mock_generator.generate.return_value = ["CCO"]
        
        response = self.client.post(
            "/api/v1/generate",
            json={
                "n_molecules": 1,
                "target_properties": {
                    "logP": [2.0, 4.0],
                    "MW": [200, 400]
                }
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("molecules", data)
        
        # Check that target properties were passed
        call_args = mock_generator.generate.call_args
        self.assertIn("target_properties", call_args[1])
    
    def test_generate_molecules_invalid_request(self):
        """Test generation with invalid request."""
        response = self.client.post(
            "/api/v1/generate",
            json={"n_molecules": -1}
        )
        
        self.assertEqual(response.status_code, 422)
    
    @patch('api.app.generator')
    def test_generate_molecules_error_handling(self, mock_generator):
        """Test error handling in generation."""
        mock_generator.generate.side_effect = Exception("Generation failed")
        
        response = self.client.post(
            "/api/v1/generate",
            json={"n_molecules": 5}
        )
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)


class TestPredictionEndpoints(unittest.TestCase):
    """Test property prediction endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('api.app.predictor')
    def test_predict_properties_single(self, mock_predictor):
        """Test single molecule property prediction."""
        mock_predictor.predict.return_value = {
            "molecules": ["CCO"],
            "predictions": [2.5],
            "uncertainties": [0.1]
        }
        
        response = self.client.post(
            "/api/v1/predict/properties",
            json={
                "molecules": ["CCO"],
                "properties": ["logP"]
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 1)
    
    @patch('api.app.predictor')
    def test_predict_properties_batch(self, mock_predictor):
        """Test batch property prediction."""
        molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
        mock_predictor.predict.return_value = {
            "molecules": molecules,
            "predictions": [2.5, 1.8, 3.2],
            "uncertainties": [0.1, 0.2, 0.15]
        }
        
        response = self.client.post(
            "/api/v1/predict/properties",
            json={
                "molecules": molecules,
                "properties": ["logP"]
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 3)
    
    def test_predict_properties_invalid_smiles(self):
        """Test prediction with invalid SMILES."""
        response = self.client.post(
            "/api/v1/predict/properties",
            json={
                "molecules": ["invalid_smiles"],
                "properties": ["logP"]
            }
        )
        
        # Should still return 200 but with error in results
        self.assertEqual(response.status_code, 200)
    
    @patch('api.app.predictor')
    def test_predict_admet(self, mock_predictor):
        """Test ADMET prediction endpoint."""
        mock_predictor.predict.return_value = {
            "molecules": ["CCO"],
            "predictions": {
                "absorption": 0.9,
                "distribution": 0.7,
                "metabolism": 0.5,
                "excretion": 0.8,
                "toxicity": 0.2
            }
        }
        
        response = self.client.post(
            "/api/v1/predict/admet",
            json={"molecule": "CCO"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("admet_properties", data)


class TestOptimizationEndpoints(unittest.TestCase):
    """Test molecule optimization endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('api.app.optimize_lead')
    async def test_optimize_molecule(self, mock_optimize):
        """Test molecule optimization."""
        mock_optimize.return_value = {
            "lead": "CCO",
            "optimized_molecules": [
                {
                    "smiles": "CCCO",
                    "score": 0.9,
                    "properties": {"logP": 2.5}
                }
            ]
        }
        
        response = self.client.post(
            "/api/v1/optimize",
            json={
                "lead_smiles": "CCO",
                "optimization_goals": {"logP": [2.0, 4.0]}
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("lead", data)
        self.assertIn("optimized_molecules", data)
    
    def test_optimize_molecule_invalid_smiles(self):
        """Test optimization with invalid SMILES."""
        response = self.client.post(
            "/api/v1/optimize",
            json={
                "lead_smiles": "invalid",
                "optimization_goals": {"logP": [2.0, 4.0]}
            }
        )
        
        self.assertEqual(response.status_code, 500)


class TestSynthesisEndpoints(unittest.TestCase):
    """Test synthesis endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('api.app.suggest_synthesis')
    async def test_synthesis_routes(self, mock_synthesis):
        """Test synthesis route suggestion."""
        mock_synthesis.return_value = {
            "target": "CCO",
            "routes": [
                {
                    "score": 0.8,
                    "steps": 2,
                    "starting_materials": ["C", "CO"]
                }
            ]
        }
        
        response = self.client.post(
            "/api/v1/synthesis/routes",
            json={"target_smiles": "CCO"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("target", data)
        self.assertIn("routes", data)
    
    @patch('api.app.predict_reactions')
    async def test_predict_reactions(self, mock_predict):
        """Test reaction prediction."""
        mock_predict.return_value = {
            "reactants": ["CCO", "CC(=O)Cl"],
            "products": ["CC(=O)OCC"],
            "feasible": True,
            "confidence": 0.9
        }
        
        response = self.client.post(
            "/api/v1/synthesis/predict-reaction",
            json={
                "reactants": ["CCO", "CC(=O)Cl"],
                "conditions": {"solvent": "DCM"}
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("feasible", data)
        self.assertIn("confidence", data)


class TestAnalysisEndpoints(unittest.TestCase):
    """Test analysis endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('api.app.analyze_scaffold')
    def test_scaffold_analysis(self, mock_analyze):
        """Test scaffold analysis."""
        mock_analyze.return_value = {
            "scaffold": "c1ccccc1",
            "frequency": 0.3,
            "molecules": ["c1ccccc1CCO"]
        }
        
        response = self.client.post(
            "/api/v1/analyze/scaffold",
            json={"molecules": ["c1ccccc1CCO", "c1ccccc1C"]}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("scaffolds", data)
    
    @patch('api.app.calculate_similarity')
    def test_similarity_calculation(self, mock_similarity):
        """Test similarity calculation."""
        mock_similarity.return_value = 0.85
        
        response = self.client.post(
            "/api/v1/analyze/similarity",
            json={
                "molecule1": "CCO",
                "molecule2": "CCCO"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("similarity", data)
        self.assertIsInstance(data["similarity"], float)


class TestBatchEndpoints(unittest.TestCase):
    """Test batch processing endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('api.app.batch_processor')
    def test_batch_process(self, mock_processor):
        """Test batch processing."""
        mock_processor.process.return_value = {
            "job_id": "12345",
            "status": "completed",
            "results": []
        }
        
        response = self.client.post(
            "/api/v1/batch/process",
            json={
                "molecules": ["CCO", "CC(=O)O"],
                "operations": ["optimize", "predict"]
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
    
    def test_batch_status(self):
        """Test batch job status."""
        # This would need a real job_id in practice
        response = self.client.get("/api/v1/batch/status/12345")
        
        # Should return 404 for non-existent job
        self.assertEqual(response.status_code, 404)


class TestErrorHandling(unittest.TestCase):
    """Test error handling across endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_404_error(self):
        """Test 404 error for non-existent endpoint."""
        response = self.client.get("/api/v1/nonexistent")
        self.assertEqual(response.status_code, 404)
    
    def test_method_not_allowed(self):
        """Test 405 error for wrong method."""
        response = self.client.get("/api/v1/generate")  # Should be POST
        self.assertEqual(response.status_code, 405)
    
    def test_validation_error(self):
        """Test validation error response."""
        response = self.client.post(
            "/api/v1/generate",
            json={"wrong_field": "value"}
        )
        
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)


class TestWebSocketEndpoints(unittest.TestCase):
    """Test WebSocket endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_websocket_connection(self):
        """Test WebSocket connection."""
        with self.client.websocket_connect("/ws") as websocket:
            # Send a message
            websocket.send_json({
                "type": "generate",
                "data": {"n_molecules": 1}
            })
            
            # Should receive a response
            data = websocket.receive_json()
            self.assertIn("type", data)
    
    def test_websocket_invalid_message(self):
        """Test WebSocket with invalid message."""
        with self.client.websocket_connect("/ws") as websocket:
            # Send invalid message
            websocket.send_json({"invalid": "message"})
            
            # Should receive error response
            data = websocket.receive_json()
            self.assertEqual(data.get("type"), "error")


if __name__ == '__main__':
    unittest.main()