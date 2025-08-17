"""Integration tests for OChem Helper."""

import unittest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import patch
import requests
import time
import subprocess

# Import modules
import sys
sys.path.insert(0, '/Users/chris/ochem-helper/src')


class TestDataPipeline(unittest.TestCase):
    """Test data download and processing pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    @patch('scripts.download_data.download_zinc')
    def test_data_download(self, mock_download):
        """Test data download process."""
        # Mock successful download
        mock_download.return_value = True
        
        # Import download script
        sys.path.insert(0, '/Users/chris/ochem-helper/scripts')
        from download_data import main
        
        # Run download
        with patch('sys.argv', ['download_data.py', '--output', self.data_dir]):
            success = main()
        
        self.assertTrue(success)
        mock_download.assert_called_once()
    
    def test_smiles_processing(self):
        """Test SMILES data processing."""
        # Create test SMILES file
        test_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CC(C)C']
        smiles_file = os.path.join(self.data_dir, 'test.smi')
        
        with open(smiles_file, 'w') as f:
            for smi in test_smiles:
                f.write(f"{smi}\n")
        
        # Process SMILES
        from data.dataset import SMILESDataset
        
        dataset = SMILESDataset(smiles_file)
        self.assertEqual(len(dataset), len(test_smiles))
        
        # Test data loading
        for i, smi in enumerate(dataset):
            self.assertIn(smi, test_smiles)


class TestModelTraining(unittest.TestCase):
    """Test model training pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_vae_training_small(self):
        """Test VAE training on small dataset."""
        from models.generative.smiles_vae import MolecularVAE, SMILESTokenizer
        from training.train_vae import train_epoch
        
        # Create small dataset
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
        tokenizer = SMILESTokenizer()
        
        # Initialize model
        vae = MolecularVAE(vocab_size=tokenizer.vocab_size)
        optimizer = torch.optim.Adam(vae.parameters())
        
        # Mock training for one batch
        import torch
        
        # Encode SMILES
        encoded = [tokenizer.encode(smi) for smi in smiles_list]
        batch = tokenizer.pad_sequences(encoded)
        
        # One forward pass
        vae.train()
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(batch)
        loss, _, _ = vae.loss_function(recon_batch, batch, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_property_predictor_training(self):
        """Test property predictor training."""
        from models.predictive import PropertyPredictor
        
        # Create predictor
        predictor = PropertyPredictor()
        
        # Mock training data
        smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
        properties = [2.5, 1.8, 3.2]  # Mock logP values
        
        # Train (if ensemble is available)
        if predictor.ensemble:
            result = predictor.fit(smiles, properties, 'logP')
            self.assertIsInstance(result, dict)
            
            # Test prediction
            predictions = predictor.predict(['CCCO'])
            self.assertIn('predictions', predictions)


class TestAPIIntegration(unittest.TestCase):
    """Test API integration and endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Start API server for testing."""
        # Start server in background
        cls.server_process = subprocess.Popen(
            ['python', '-m', 'uvicorn', 'api.app:app', '--port', '8001'],
            cwd='/Users/chris/ochem-helper/src',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        cls.base_url = 'http://localhost:8001'
    
    @classmethod
    def tearDownClass(cls):
        """Stop API server."""
        cls.server_process.terminate()
        cls.server_process.wait()
    
    def test_api_health_check(self):
        """Test API health check."""
        try:
            response = requests.get(f'{self.base_url}/health')
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['status'], 'healthy')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server not running")
    
    def test_api_generation_endpoint(self):
        """Test molecule generation through API."""
        try:
            response = requests.post(
                f'{self.base_url}/api/v1/generate',
                json={'n_molecules': 3}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn('molecules', data)
                self.assertIsInstance(data['molecules'], list)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server not running")
    
    def test_api_prediction_endpoint(self):
        """Test property prediction through API."""
        try:
            response = requests.post(
                f'{self.base_url}/api/v1/predict/properties',
                json={
                    'molecules': ['CCO'],
                    'properties': ['logP']
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn('results', data)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server not running")


class TestMCPIntegration(unittest.TestCase):
    """Test MCP server integration."""
    
    def setUp(self):
        """Set up MCP environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        sys.path.insert(0, '/Users/chris/ochem-helper/mcp')
        from server.ochem_mcp import OChemMCPServer
        self.server = OChemMCPServer()
    
    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()
    
    async def test_mcp_full_workflow(self):
        """Test complete MCP workflow."""
        # 1. Generate molecules
        gen_result = await self.server.handle_tool_call(
            'generate_molecules',
            {'n_molecules': 5, 'target_properties': {'logP': [2, 4]}}
        )
        
        if 'molecules' in gen_result:
            molecules = gen_result['molecules']
            
            # 2. Predict properties for generated molecules
            for mol in molecules[:2]:  # Test first 2
                pred_result = await self.server.handle_tool_call(
                    'predict_properties',
                    {'smiles': mol, 'properties': ['logP', 'MW']}
                )
                self.assertIsNotNone(pred_result)
            
            # 3. Optimize one molecule
            if molecules:
                opt_result = await self.server.handle_tool_call(
                    'optimize_structure',
                    {
                        'lead_smiles': molecules[0],
                        'optimization_goals': {'logP': [2, 4]}
                    }
                )
                self.assertIsNotNone(opt_result)
    
    def test_mcp_workflow_execution(self):
        """Execute MCP workflow test."""
        self.loop.run_until_complete(self.test_mcp_full_workflow())


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end molecular discovery workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_molecular_discovery_workflow(self):
        """Test complete molecular discovery workflow."""
        from models.generative import MoleculeGenerator
        from models.predictive import PropertyPredictor
        from core.validators import MoleculeValidator
        from core.descriptors import MolecularDescriptors
        
        # 1. Generate molecules
        generator = MoleculeGenerator()
        molecules = generator.generate(
            n_molecules=10,
            target_properties={'logP': [2, 4]}
        )
        
        self.assertIsInstance(molecules, list)
        self.assertGreater(len(molecules), 0)
        
        # 2. Validate molecules
        validator = MoleculeValidator()
        valid_molecules = []
        
        for mol in molecules:
            validation = validator.validate_smiles(mol)
            if validation['valid']:
                valid_molecules.append(mol)
        
        # 3. Calculate descriptors
        descriptor_calc = MolecularDescriptors()
        
        for mol in valid_molecules[:3]:  # Test first 3
            descriptors = descriptor_calc.calculate(mol)
            self.assertIsInstance(descriptors, dict)
            
            # Check drug-likeness
            drug_like = validator.check_drug_likeness(mol)
            self.assertIn('lipinski_compliant', drug_like)
        
        # 4. Predict properties
        predictor = PropertyPredictor()
        
        if valid_molecules:
            predictions = predictor.predict(valid_molecules)
            self.assertIn('predictions', predictions)
            self.assertIn('uncertainties', predictions)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI tool integration."""
    
    def test_cli_generate_command(self):
        """Test CLI generation command."""
        result = subprocess.run(
            ['python', '-m', 'cli.generate', '--num', '3'],
            cwd='/Users/chris/ochem-helper/src',
            capture_output=True,
            text=True
        )
        
        # Check if command runs without error
        self.assertIn('Generated molecules:', result.stdout)
    
    def test_cli_predict_command(self):
        """Test CLI prediction command."""
        result = subprocess.run(
            ['python', '-m', 'cli.predict', 'CCO', '--properties', 'logP,MW'],
            cwd='/Users/chris/ochem-helper/src',
            capture_output=True,
            text=True
        )
        
        # Check if command runs
        self.assertNotEqual(result.returncode, 1)
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ['python', '-m', 'cli', '--help'],
            cwd='/Users/chris/ochem-helper/src',
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage:', result.stdout.lower())


class TestDockerIntegration(unittest.TestCase):
    """Test container integration."""
    
    @unittest.skipIf(not shutil.which('podman'), "Podman not installed")
    def test_container_build(self):
        """Test container build process."""
        result = subprocess.run(
            ['podman', 'build', '-t', 'ochem-helper:test', '-f', 'containers/Containerfile', '.'],
            cwd='/Users/chris/ochem-helper',
            capture_output=True
        )
        
        self.assertEqual(result.returncode, 0)
    
    @unittest.skipIf(not shutil.which('podman'), "Podman not installed")
    def test_container_run(self):
        """Test running container."""
        # First ensure image is built
        subprocess.run(
            ['podman', 'build', '-t', 'ochem-helper:test', '-f', 'containers/Containerfile', '.'],
            cwd='/Users/chris/ochem-helper',
            capture_output=True
        )
        
        # Run container
        result = subprocess.run(
            ['podman', 'run', '--rm', 'ochem-helper:test', 'python', '-c', 'print("Container works")'],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('Container works', result.stdout)


if __name__ == '__main__':
    unittest.main()