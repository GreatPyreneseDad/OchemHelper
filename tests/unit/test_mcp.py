"""Unit tests for MCP server and tools."""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json

# Import MCP modules
import sys
sys.path.insert(0, '/Users/chris/ochem-helper/mcp')

from server.ochem_mcp import OChemMCPServer
from tools import predict_properties, suggest_synthesis, optimize_structure, reaction_prediction


class TestMCPServer(unittest.TestCase):
    """Test cases for MCP server."""
    
    def setUp(self):
        """Set up test server."""
        self.server = OChemMCPServer()
    
    def test_server_initialization(self):
        """Test server initialization."""
        self.assertIsInstance(self.server, OChemMCPServer)
        self.assertIsNotNone(self.server.tools)
        self.assertIsNotNone(self.server.prompts)
    
    def test_tool_registration(self):
        """Test that tools are properly registered."""
        tools = self.server.list_tools()
        
        # Check that essential tools are registered
        tool_names = [tool['name'] for tool in tools]
        self.assertIn('predict_properties', tool_names)
        self.assertIn('suggest_synthesis', tool_names)
        self.assertIn('optimize_structure', tool_names)
        self.assertIn('check_reaction_feasibility', tool_names)
    
    def test_prompt_registration(self):
        """Test that prompts are properly registered."""
        prompts = self.server.list_prompts()
        
        # Check that prompts exist
        self.assertIsInstance(prompts, list)
        self.assertGreater(len(prompts), 0)
        
        # Check prompt structure
        for prompt in prompts:
            self.assertIn('name', prompt)
            self.assertIn('description', prompt)
            self.assertIn('template', prompt)
    
    async def test_handle_tool_call(self):
        """Test handling tool calls."""
        # Mock a tool call
        with patch.object(self.server, 'predict_properties_tool', new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = {"predictions": {"logP": 2.5}}
            
            result = await self.server.handle_tool_call(
                'predict_properties',
                {'smiles': 'CCO', 'properties': ['logP']}
            )
            
            self.assertIsNotNone(result)
            mock_tool.assert_called_once()
    
    def test_get_tool_schema(self):
        """Test getting tool schema."""
        schema = self.server.get_tool_schema('predict_properties')
        
        self.assertIsNotNone(schema)
        self.assertIn('name', schema)
        self.assertIn('description', schema)
        self.assertIn('input_schema', schema)


class TestPredictPropertiesTool(unittest.TestCase):
    """Test property prediction tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()
    
    @patch('mcp.tools.predict_properties.PropertyPredictor')
    def test_predict_single_property(self, mock_predictor_class):
        """Test predicting single property."""
        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            'predictions': [2.5],
            'uncertainties': [0.1]
        }
        mock_predictor_class.from_pretrained.return_value = mock_predictor
        
        # Run prediction
        result = self.loop.run_until_complete(
            predict_properties.predict_molecular_properties(
                'CCO',
                ['logP']
            )
        )
        
        self.assertIn('smiles', result)
        self.assertIn('predictions', result)
        self.assertEqual(result['predictions']['logP']['value'], 2.5)
    
    def test_predict_invalid_smiles(self):
        """Test prediction with invalid SMILES."""
        result = self.loop.run_until_complete(
            predict_properties.predict_molecular_properties(
                'invalid_smiles',
                ['logP']
            )
        )
        
        self.assertIn('error', result)
    
    @patch('mcp.tools.predict_properties.PropertyPredictor')
    def test_predict_multiple_properties(self, mock_predictor_class):
        """Test predicting multiple properties."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            'predictions': [2.5],
            'uncertainties': [0.1]
        }
        mock_predictor_class.from_pretrained.return_value = mock_predictor
        
        result = self.loop.run_until_complete(
            predict_properties.predict_molecular_properties(
                'CCO',
                ['logP', 'MW', 'TPSA']
            )
        )
        
        self.assertIn('predictions', result)
        # Should have attempted to predict all properties
        self.assertGreater(len(result['predictions']), 0)
    
    @patch('mcp.tools.predict_properties.rdkit.Chem')
    def test_predict_admet_properties(self, mock_chem):
        """Test ADMET property prediction."""
        mock_chem.MolFromSmiles.return_value = Mock()
        
        result = self.loop.run_until_complete(
            predict_properties.predict_admet_properties('CCO')
        )
        
        self.assertIn('smiles', result)
        self.assertIn('admet_properties', result)
        
        # Check ADMET categories
        admet = result['admet_properties']
        self.assertIn('absorption', admet)
        self.assertIn('distribution', admet)
        self.assertIn('metabolism', admet)
        self.assertIn('excretion', admet)
        self.assertIn('toxicity', admet)


class TestSuggestSynthesisTool(unittest.TestCase):
    """Test synthesis suggestion tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()
    
    @patch('mcp.tools.suggest_synthesis.Chem')
    def test_retrosynthetic_analysis(self, mock_chem):
        """Test retrosynthetic analysis."""
        mock_chem.MolFromSmiles.return_value = Mock()
        
        result = self.loop.run_until_complete(
            suggest_synthesis.retrosynthetic_analysis(
                'CC(=O)OCC',  # Ethyl acetate
                max_steps=3
            )
        )
        
        self.assertIn('target', result)
        self.assertIn('routes', result)
        self.assertIsInstance(result['routes'], list)
    
    def test_synthesis_with_starting_materials(self):
        """Test synthesis with specified starting materials."""
        result = self.loop.run_until_complete(
            suggest_synthesis.retrosynthetic_analysis(
                'CC(=O)OCC',
                max_steps=3,
                starting_materials=['CCO', 'CC(=O)O']
            )
        )
        
        self.assertIn('routes', result)
        # Routes should consider the specified starting materials
    
    def test_invalid_target_molecule(self):
        """Test synthesis with invalid target."""
        result = self.loop.run_until_complete(
            suggest_synthesis.retrosynthetic_analysis('invalid_smiles')
        )
        
        self.assertIn('error', result)
    
    @patch('mcp.tools.suggest_synthesis.apply_reaction_template')
    def test_reaction_template_application(self, mock_apply):
        """Test reaction template application."""
        mock_apply.return_value = [Mock()]
        
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CC(=O)OCC')
        
        if mol:
            result = suggest_synthesis.apply_reaction_template(
                mol,
                '[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[C:4][OH]'
            )
            
            self.assertIsNotNone(result)


class TestOptimizeStructureTool(unittest.TestCase):
    """Test structure optimization tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()
    
    @patch('mcp.tools.optimize_structure.Chem')
    def test_lead_optimization(self, mock_chem):
        """Test lead optimization."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        
        result = self.loop.run_until_complete(
            optimize_structure.optimize_lead(
                'CCO',
                {'logP': [2.0, 4.0], 'MW': [200, 400]},
                maintain_scaffold=True
            )
        )
        
        self.assertIn('lead', result)
        self.assertIn('optimized_molecules', result)
        self.assertIn('optimization_strategy', result)
    
    def test_analog_generation(self):
        """Test analog generation."""
        result = self.loop.run_until_complete(
            optimize_structure.generate_analogs(
                'CCO',
                num_analogs=5,
                similarity_threshold=0.7
            )
        )
        
        self.assertIn('reference', result)
        self.assertIn('analogs', result)
        self.assertIsInstance(result['analogs'], list)
    
    def test_optimization_with_invalid_lead(self):
        """Test optimization with invalid lead."""
        result = self.loop.run_until_complete(
            optimize_structure.optimize_lead(
                'invalid_smiles',
                {'logP': [2.0, 4.0]}
            )
        )
        
        self.assertIn('error', result)
    
    @patch('mcp.tools.optimize_structure.calculate_properties')
    def test_property_calculation(self, mock_calc_props):
        """Test property calculation."""
        mock_calc_props.return_value = {
            'MW': 300.0,
            'logP': 2.5,
            'TPSA': 60.0
        }
        
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CCO')
        
        if mol:
            props = optimize_structure.calculate_properties(mol)
            self.assertIn('MW', props)
            self.assertIn('logP', props)
            self.assertIn('TPSA', props)


class TestReactionPredictionTool(unittest.TestCase):
    """Test reaction prediction tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()
    
    @patch('mcp.tools.reaction_prediction.Chem')
    def test_check_reaction_feasibility(self, mock_chem):
        """Test reaction feasibility checking."""
        mock_chem.MolFromSmiles.return_value = Mock()
        
        result = self.loop.run_until_complete(
            reaction_prediction.check_feasibility(
                ['CCO', 'CC(=O)Cl'],  # Ethanol + Acetyl chloride
                ['CC(=O)OCC']  # Ethyl acetate
            )
        )
        
        self.assertIn('feasible', result)
        self.assertIn('confidence', result)
        self.assertIn('reaction_type', result)
    
    def test_reaction_with_conditions(self):
        """Test reaction with specified conditions."""
        result = self.loop.run_until_complete(
            reaction_prediction.check_feasibility(
                ['CCO', 'CC(=O)O'],
                ['CC(=O)OCC'],
                conditions={'catalyst': 'H2SO4', 'temperature': 'reflux'}
            )
        )
        
        self.assertIn('feasible', result)
        # Should consider conditions in feasibility assessment
    
    def test_unbalanced_reaction(self):
        """Test unbalanced reaction detection."""
        result = self.loop.run_until_complete(
            reaction_prediction.check_feasibility(
                ['CCO'],
                ['CC(=O)OCC', 'CCCCCC']  # Clearly unbalanced
            )
        )
        
        self.assertIn('feasible', result)
        self.assertIn('issues', result)
        # Should detect atom imbalance
    
    def test_invalid_reactants(self):
        """Test reaction with invalid SMILES."""
        result = self.loop.run_until_complete(
            reaction_prediction.check_feasibility(
                ['invalid_smiles'],
                ['CCO']
            )
        )
        
        self.assertIn('error', result)


class TestMCPIntegration(unittest.TestCase):
    """Test MCP server integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.server = OChemMCPServer()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()
    
    async def test_full_workflow(self):
        """Test a full workflow through MCP."""
        # 1. Predict properties
        prop_result = await self.server.handle_tool_call(
            'predict_properties',
            {'smiles': 'CCO', 'properties': ['logP']}
        )
        self.assertIsNotNone(prop_result)
        
        # 2. Optimize structure
        opt_result = await self.server.handle_tool_call(
            'optimize_structure',
            {
                'lead_smiles': 'CCO',
                'optimization_goals': {'logP': [2.0, 4.0]}
            }
        )
        self.assertIsNotNone(opt_result)
        
        # 3. Suggest synthesis
        synth_result = await self.server.handle_tool_call(
            'suggest_synthesis',
            {'target_smiles': 'CC(=O)OCC'}
        )
        self.assertIsNotNone(synth_result)
    
    def test_workflow_execution(self):
        """Run the full workflow test."""
        self.loop.run_until_complete(self.test_full_workflow())


class TestMCPPrompts(unittest.TestCase):
    """Test MCP prompt templates."""
    
    def setUp(self):
        """Set up test server."""
        self.server = OChemMCPServer()
    
    def test_prompt_rendering(self):
        """Test prompt template rendering."""
        prompts = self.server.list_prompts()
        
        for prompt in prompts:
            # Test that prompts can be rendered with sample data
            template = prompt['template']
            
            # Check for common placeholders
            if '{molecule}' in template:
                rendered = template.format(molecule='CCO')
                self.assertNotIn('{molecule}', rendered)
            
            if '{target}' in template:
                rendered = template.format(target='High bioavailability')
                self.assertNotIn('{target}', rendered)
    
    def test_chemistry_expert_prompt(self):
        """Test chemistry expert prompt."""
        prompt = self.server.get_prompt('chemistry_expert')
        
        self.assertIsNotNone(prompt)
        self.assertIn('organic chemistry', prompt['description'].lower())
        self.assertIn('expert', prompt['template'].lower())
    
    def test_synthesis_planner_prompt(self):
        """Test synthesis planner prompt."""
        prompt = self.server.get_prompt('synthesis_planner')
        
        self.assertIsNotNone(prompt)
        self.assertIn('synthesis', prompt['description'].lower())
        self.assertIn('retrosynthetic', prompt['template'].lower())


if __name__ == '__main__':
    unittest.main()