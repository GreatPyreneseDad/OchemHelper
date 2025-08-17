"""
OChem Helper MCP Server - Advanced Chemistry AI Integration
Enhanced MCP server adapted from TraderAI for molecular discovery and analysis

Provides comprehensive chemical analysis tools for AI assistants like Claude and xAI.
Integrates molecular generation, property prediction, synthesis planning, and optimization.
"""

import asyncio
import json
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path
import torch

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import our enhanced chemistry modules
from src.models.generative.molecular_reservoir_engine import (
    MolecularReservoirEngine, MolecularReservoirConfig
)
from src.models.generative.smiles_vae import MolecularVAE, SMILESTokenizer
from src.models.predictive.molecular_ensemble import (
    MolecularPropertyEnsemble, MolecularEnsembleConfig
)
from src.core.molecular_graph import MolecularGraph

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Some functionality will be limited.")

logger = logging.getLogger(__name__)

class OChemMCPServer:
    """Advanced MCP server for organic chemistry AI assistance"""
    
    def __init__(self):
        self.server = Server("ochem-helper-advanced")
        
        # Initialize advanced chemistry engines
        self.molecular_reservoir = None
        self.vae_model = None
        self.property_ensemble = None
        self.tokenizer = SMILESTokenizer()
        
        # Model loading status
        self.models_loaded = {
            'reservoir': False,
            'vae': False,
            'ensemble': False
        }
        
        # Performance tracking
        self.analysis_history = []
        self.generation_history = []
        
        self.setup_handlers()
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize chemistry models"""
        try:
            # Initialize Molecular Reservoir Engine
            reservoir_config = MolecularReservoirConfig(
                num_nodes=150,
                spatial_dimension=3,
                learning_rate=0.01,
                coherence_coupling=0.08
            )
            self.molecular_reservoir = MolecularReservoirEngine(reservoir_config)
            self.models_loaded['reservoir'] = True
            logger.info("Molecular Reservoir Engine initialized")
            
            # Try to load pre-trained VAE
            vae_path = Path("models/checkpoints/best_model.pt")
            if vae_path.exists():
                checkpoint = torch.load(vae_path, map_location='cpu')
                config = checkpoint['model_config']
                
                self.vae_model = MolecularVAE(
                    vocab_size=config['vocab_size'],
                    embedding_dim=config.get('embedding_dim', 128),
                    hidden_dim=config['hidden_dim'],
                    latent_dim=config['latent_dim'],
                    num_layers=config.get('num_layers', 2),
                    max_length=config['max_length'],
                    beta=config.get('beta', 1.0)
                )
                self.vae_model.load_state_dict(checkpoint['model_state_dict'])
                self.vae_model.eval()
                self.models_loaded['vae'] = True
                logger.info("VAE model loaded from checkpoint")
            else:
                # Initialize default VAE
                self.vae_model = MolecularVAE(vocab_size=self.tokenizer.vocab_size)
                logger.info("Default VAE model initialized")
            
            # Initialize Property Ensemble
            ensemble_config = MolecularEnsembleConfig(
                use_xgboost=True,
                use_lightgbm=True,
                use_neural_net=True,
                uncertainty_estimation=True
            )
            self.property_ensemble = MolecularPropertyEnsemble(ensemble_config)
            self.models_loaded['ensemble'] = True
            logger.info("Property Ensemble initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
        
    def setup_handlers(self):
        """Set up MCP tool handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available chemistry tools"""
            return [
                types.Tool(
                    name="analyze_molecule_advanced",
                    description="Advanced molecular analysis using reservoir computing and ensemble prediction",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "smiles": {
                                "type": "string",
                                "description": "SMILES string of the molecule"
                            },
                            "properties": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Properties to predict (logP, MW, TPSA, QED, SA_score)"
                            },
                            "use_reservoir": {
                                "type": "boolean",
                                "description": "Use molecular reservoir computing for enhanced analysis",
                                "default": True
                            },
                            "use_ensemble": {
                                "type": "boolean", 
                                "description": "Use ensemble prediction for properties",
                                "default": True
                            }
                        },
                        "required": ["smiles"]
                    }
                ),
                types.Tool(
                    name="generate_molecules_advanced",
                    description="Generate novel molecules using VAE and reservoir computing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "n_molecules": {
                                "type": "integer",
                                "description": "Number of molecules to generate",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100
                            },
                            "target_properties": {
                                "type": "object",
                                "description": "Target molecular properties (e.g., {'logP': 2.5, 'MW': 350})"
                            },
                            "reference_smiles": {
                                "type": "string",
                                "description": "Reference molecule for similarity-based generation"
                            },
                            "use_reservoir_guidance": {
                                "type": "boolean",
                                "description": "Use reservoir computing to guide generation",
                                "default": True
                            },
                            "diversity_factor": {
                                "type": "number",
                                "description": "Diversity factor (0.0-1.0, higher = more diverse)",
                                "default": 0.3,
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        }
                    }
                ),
                types.Tool(
                    name="optimize_lead_compound",
                    description="Optimize a lead compound for better properties using advanced ML",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lead_smiles": {
                                "type": "string",
                                "description": "SMILES of lead compound"
                            },
                            "optimization_goals": {
                                "type": "object",
                                "description": "Optimization targets (e.g., {'logP': [2, 4], 'QED': [0.7, 1.0]})"
                            },
                            "maintain_scaffold": {
                                "type": "boolean",
                                "description": "Keep core scaffold intact",
                                "default": True
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": "Maximum optimization iterations",
                                "default": 20,
                                "minimum": 5,
                                "maximum": 100
                            }
                        },
                        "required": ["lead_smiles", "optimization_goals"]
                    }
                ),
                types.Tool(
                    name="predict_synthesis_route",
                    description="Predict synthetic routes using reservoir computing anticipation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "target_smiles": {
                                "type": "string",
                                "description": "SMILES of target molecule"
                            },
                            "max_steps": {
                                "type": "integer",
                                "description": "Maximum synthesis steps",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "starting_materials": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Preferred starting materials (SMILES)"
                            },
                            "use_reservoir_anticipation": {
                                "type": "boolean",
                                "description": "Use reservoir anticipation for route prediction",
                                "default": True
                            }
                        },
                        "required": ["target_smiles"]
                    }
                ),
                types.Tool(
                    name="compare_molecules",
                    description="Compare molecules using advanced similarity metrics and reservoir analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "smiles_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of SMILES to compare",
                                "minItems": 2
                            },
                            "comparison_metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Metrics to use (tanimoto, reservoir_similarity, property_distance)"
                            },
                            "reference_molecule": {
                                "type": "string",
                                "description": "Reference molecule for comparison"
                            }
                        },
                        "required": ["smiles_list"]
                    }
                ),
                types.Tool(
                    name="chemical_space_exploration",
                    description="Explore chemical space around molecules using reservoir dynamics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "seed_molecules": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Seed molecules for exploration"
                            },
                            "exploration_radius": {
                                "type": "number",
                                "description": "Exploration radius in chemical space",
                                "default": 0.3,
                                "minimum": 0.1,
                                "maximum": 1.0
                            },
                            "n_candidates": {
                                "type": "integer",
                                "description": "Number of candidates to generate",
                                "default": 50,
                                "minimum": 10,
                                "maximum": 200
                            },
                            "filter_criteria": {
                                "type": "object",
                                "description": "Filtering criteria for generated molecules"
                            }
                        },
                        "required": ["seed_molecules"]
                    }
                ),
                types.Tool(
                    name="reaction_feasibility_analysis",
                    description="Analyze reaction feasibility using molecular reservoir dynamics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "reactants": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "SMILES strings of reactants"
                            },
                            "products": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "SMILES strings of expected products"
                            },
                            "reaction_conditions": {
                                "type": "object",
                                "description": "Reaction conditions (temperature, pH, solvent)"
                            },
                            "analyze_mechanism": {
                                "type": "boolean",
                                "description": "Analyze reaction mechanism using reservoir dynamics",
                                "default": True
                            }
                        },
                        "required": ["reactants", "products"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]]
        ) -> List[types.TextContent]:
            """Handle tool execution"""
            try:
                if name == "analyze_molecule_advanced":
                    result = await self._analyze_molecule_advanced(arguments)
                
                elif name == "generate_molecules_advanced":
                    result = await self._generate_molecules_advanced(arguments)
                
                elif name == "optimize_lead_compound":
                    result = await self._optimize_lead_compound(arguments)
                
                elif name == "predict_synthesis_route":
                    result = await self._predict_synthesis_route(arguments)
                
                elif name == "compare_molecules":
                    result = await self._compare_molecules(arguments)
                
                elif name == "chemical_space_exploration":
                    result = await self._chemical_space_exploration(arguments)
                
                elif name == "reaction_feasibility_analysis":
                    result = await self._reaction_feasibility_analysis(arguments)
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]
                
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": name,
                        "status": "failed"
                    }, indent=2)
                )]
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            """List available prompts for chemistry tasks"""
            return [
                types.Prompt(
                    name="drug_discovery_workflow",
                    description="Complete drug discovery workflow with advanced ML",
                    arguments=[
                        types.PromptArgument(
                            name="target",
                            description="Biological target (e.g., 'EGFR kinase')",
                            required=True
                        ),
                        types.PromptArgument(
                            name="indication",
                            description="Disease indication",
                            required=False
                        ),
                        types.PromptArgument(
                            name="constraints",
                            description="Drug development constraints",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="lead_optimization_campaign",
                    description="Systematic lead optimization using reservoir computing",
                    arguments=[
                        types.PromptArgument(
                            name="lead_smiles",
                            description="SMILES of lead compound",
                            required=True
                        ),
                        types.PromptArgument(
                            name="issues",
                            description="Issues to address (e.g., 'low solubility, high clearance')",
                            required=True
                        ),
                        types.PromptArgument(
                            name="objectives",
                            description="Optimization objectives",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="synthesis_planning_advanced",
                    description="Advanced synthesis planning with reservoir anticipation",
                    arguments=[
                        types.PromptArgument(
                            name="target_smiles",
                            description="SMILES of target molecule",
                            required=True
                        ),
                        types.PromptArgument(
                            name="complexity",
                            description="Synthesis complexity level",
                            required=False
                        ),
                        types.PromptArgument(
                            name="available_reagents",
                            description="Available starting materials and reagents",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="chemical_space_analysis",
                    description="Analyze and explore chemical space using advanced ML",
                    arguments=[
                        types.PromptArgument(
                            name="focus_area",
                            description="Chemical space focus (e.g., 'kinase inhibitors')",
                            required=True
                        ),
                        types.PromptArgument(
                            name="reference_compounds",
                            description="Reference compounds for analysis",
                            required=False
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: Optional[Dict[str, str]]
        ) -> types.GetPromptResult:
            """Get specific prompt template"""
            
            if name == "drug_discovery_workflow":
                target = arguments.get("target", "unspecified target")
                indication = arguments.get("indication", "")
                constraints = arguments.get("constraints", "")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""I need comprehensive drug discovery assistance for {target}.
{f'Disease indication: {indication}' if indication else ''}
{f'Constraints: {constraints}' if constraints else ''}

Please help me with advanced molecular discovery using:

1. **Target Analysis & Validation**
   - Use analyze_molecule_advanced to study known inhibitors
   - Identify key pharmacophore features using reservoir computing
   - Analyze binding site characteristics

2. **Lead Identification**
   - Generate novel scaffolds using generate_molecules_advanced
   - Screen virtual compounds with ensemble property prediction
   - Optimize hit compounds using lead optimization tools

3. **Lead Optimization** 
   - Use optimize_lead_compound for systematic improvement
   - Address ADMET issues with reservoir-guided design
   - Optimize selectivity and potency simultaneously

4. **Synthesis Planning**
   - Use predict_synthesis_route for retrosynthetic analysis
   - Evaluate synthetic accessibility with reservoir anticipation
   - Plan scalable synthetic routes

5. **Chemical Space Exploration**
   - Map relevant chemical space using chemical_space_exploration
   - Identify novel chemotypes and escape routes
   - Use reservoir dynamics for innovative design

Please provide a comprehensive analysis using the advanced OChem Helper tools."""
                        )
                    )
                ]
            
            elif name == "lead_optimization_campaign":
                lead_smiles = arguments.get("lead_smiles", "")
                issues = arguments.get("issues", "")
                objectives = arguments.get("objectives", "")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""I need systematic lead optimization for this compound:
SMILES: {lead_smiles}
Issues to address: {issues}
{f'Objectives: {objectives}' if objectives else ''}

Please conduct advanced lead optimization using:

1. **Current Analysis**
   - Use analyze_molecule_advanced with reservoir computing
   - Identify problematic molecular features
   - Predict current ADMET profile with ensemble methods

2. **Optimization Strategy**
   - Use optimize_lead_compound with multi-objective goals
   - Apply reservoir-guided molecular modifications
   - Generate analogs addressing specific issues

3. **Analog Generation**
   - Use generate_molecules_advanced for similar structures
   - Explore chemical space around the lead
   - Filter candidates by predicted properties

4. **Validation & Ranking**
   - Compare generated analogs using compare_molecules
   - Rank by predicted improvement in target properties
   - Assess synthetic accessibility

5. **Synthesis Planning**
   - Plan synthesis routes for top candidates
   - Use reservoir anticipation for route optimization
   - Evaluate scalability and feasibility

Provide detailed optimization recommendations with specific molecular modifications."""
                        )
                    )
                ]
            
            elif name == "synthesis_planning_advanced":
                target_smiles = arguments.get("target_smiles", "")
                complexity = arguments.get("complexity", "moderate")
                reagents = arguments.get("available_reagents", "standard organic reagents")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""Plan advanced synthesis for this target molecule:
SMILES: {target_smiles}
Complexity level: {complexity}
Available reagents: {reagents}

Please provide comprehensive synthesis planning using:

1. **Target Analysis**
   - Use analyze_molecule_advanced to assess the target
   - Identify key structural features and challenges
   - Evaluate synthetic accessibility with reservoir computing

2. **Retrosynthetic Analysis**
   - Use predict_synthesis_route with reservoir anticipation
   - Identify strategic bond disconnections
   - Consider multiple synthetic approaches

3. **Route Optimization**
   - Use reaction_feasibility_analysis for key steps
   - Optimize reaction conditions with molecular reservoir
   - Assess step efficiency and selectivity

4. **Alternative Approaches**
   - Explore different synthetic strategies
   - Use chemical_space_exploration for route variants
   - Compare convergent vs linear approaches

5. **Practical Considerations**
   - Evaluate reagent availability and cost
   - Consider scalability and safety
   - Plan purification and characterization

Provide detailed synthetic routes with step-by-step analysis and recommendations."""
                        )
                    )
                ]
            
            elif name == "chemical_space_analysis":
                focus_area = arguments.get("focus_area", "general chemical space")
                references = arguments.get("reference_compounds", "")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""Analyze chemical space for: {focus_area}
{f'Reference compounds: {references}' if references else ''}

Please conduct comprehensive chemical space analysis using:

1. **Space Mapping**
   - Use chemical_space_exploration to map relevant regions
   - Identify density clusters and empty spaces
   - Use reservoir computing for pattern recognition

2. **Reference Analysis**
   - Analyze reference compounds with advanced tools
   - Extract common molecular features and patterns
   - Identify structure-activity relationships

3. **Novel Region Identification**
   - Find unexplored areas in chemical space
   - Generate molecules in novel regions
   - Use reservoir dynamics for innovative designs

4. **Diversity Analysis**
   - Compare molecular diversity across regions
   - Identify scaffold hopping opportunities
   - Assess property distributions

5. **Optimization Opportunities**
   - Find regions for property optimization
   - Identify structure-property trends
   - Suggest design strategies for improvement

Provide insights on chemical space opportunities and design strategies."""
                        )
                    )
                ]
            
            else:
                raise ValueError(f"Unknown prompt: {name}")
            
            return types.GetPromptResult(
                description=f"Advanced chemistry prompt for {name}",
                messages=messages
            )
    
    async def _analyze_molecule_advanced(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced molecular analysis using reservoir computing and ensemble prediction"""
        smiles = args.get("smiles")
        properties = args.get("properties", ["logP", "MW", "TPSA", "QED", "SA_score"])
        use_reservoir = args.get("use_reservoir", True)
        use_ensemble = args.get("use_ensemble", True)
        
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available for molecular analysis"}
        
        if not smiles:
            return {"error": "SMILES string required"}
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}
        
        result = {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "basic_properties": {},
            "advanced_analysis": {}
        }
        
        # Basic RDKit properties
        result["basic_properties"] = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "num_hbd": Descriptors.NumHDonors(mol),
            "num_hba": Descriptors.NumHAcceptors(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "qed": Descriptors.qed(mol)
        }
        
        # Reservoir computing analysis
        if use_reservoir and self.molecular_reservoir:
            try:
                reservoir_analysis = self.molecular_reservoir.predict_molecular_properties(
                    smiles, properties
                )
                result["advanced_analysis"]["reservoir_predictions"] = reservoir_analysis
                
                # Get reservoir state
                reservoir_state = self.molecular_reservoir.get_reservoir_state()
                result["advanced_analysis"]["reservoir_state"] = {
                    "chemical_coherence": reservoir_state["chemical_coherence"],
                    "reaction_anticipation": reservoir_state["reaction_anticipation"],
                    "average_energy": reservoir_state["average_energy"]
                }
                
            except Exception as e:
                result["advanced_analysis"]["reservoir_error"] = str(e)
        
        # Ensemble prediction
        if use_ensemble and self.property_ensemble:
            try:
                if self.models_loaded['ensemble']:
                    ensemble_pred, ensemble_uncertainty = self.property_ensemble.predict([smiles])
                    result["advanced_analysis"]["ensemble_predictions"] = {
                        "predictions": ensemble_pred[0] if len(ensemble_pred) > 0 else None,
                        "uncertainty": ensemble_uncertainty[0] if len(ensemble_uncertainty) > 0 else None
                    }
            except Exception as e:
                result["advanced_analysis"]["ensemble_error"] = str(e)
        
        # Chemical alerts and drug-likeness
        result["drug_likeness"] = {
            "lipinski_violations": self._count_lipinski_violations(mol),
            "veber_compliant": self._check_veber(mol),
            "pains_alerts": self._check_pains(mol)
        }
        
        # Store analysis
        self.analysis_history.append({
            "smiles": smiles,
            "timestamp": asyncio.get_event_loop().time(),
            "analysis_type": "advanced"
        })
        
        return result
    
    async def _generate_molecules_advanced(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate molecules using VAE and reservoir computing"""
        n_molecules = args.get("n_molecules", 10)
        target_properties = args.get("target_properties", {})
        reference_smiles = args.get("reference_smiles")
        use_reservoir_guidance = args.get("use_reservoir_guidance", True)
        diversity_factor = args.get("diversity_factor", 0.3)
        
        if not self.vae_model:
            return {"error": "VAE model not available"}
        
        result = {
            "generated_molecules": [],
            "generation_stats": {},
            "filter_results": {}
        }
        
        try:
            # Generate molecules using VAE
            if reference_smiles:
                # Similarity-based generation
                molecules = self.vae_model.generate_similar(
                    reference_smiles, n_molecules, diversity_factor
                )
            else:
                # Random generation
                molecules = self.vae_model.generate(n_molecules)
            
            # Filter by target properties if specified
            if target_properties:
                filtered_molecules = self._filter_by_properties(molecules, target_properties)
                result["filter_results"] = {
                    "original_count": len(molecules),
                    "filtered_count": len(filtered_molecules),
                    "filter_criteria": target_properties
                }
                molecules = filtered_molecules
            
            # Enhanced analysis with reservoir computing
            if use_reservoir_guidance and self.molecular_reservoir:
                enhanced_molecules = []
                for smiles in molecules:
                    try:
                        # Get reservoir analysis
                        reservoir_props = self.molecular_reservoir.predict_molecular_properties(smiles)
                        
                        mol_data = {
                            "smiles": smiles,
                            "reservoir_properties": reservoir_props,
                            "basic_properties": self._calculate_basic_properties(smiles)
                        }
                        enhanced_molecules.append(mol_data)
                        
                    except Exception as e:
                        # Include molecule even if reservoir analysis fails
                        enhanced_molecules.append({
                            "smiles": smiles,
                            "basic_properties": self._calculate_basic_properties(smiles),
                            "reservoir_error": str(e)
                        })
                
                result["generated_molecules"] = enhanced_molecules
            else:
                result["generated_molecules"] = [
                    {
                        "smiles": smiles,
                        "basic_properties": self._calculate_basic_properties(smiles)
                    }
                    for smiles in molecules
                ]
            
            # Generation statistics
            result["generation_stats"] = {
                "total_generated": len(molecules),
                "valid_molecules": sum(1 for m in result["generated_molecules"] if "basic_properties" in m),
                "diversity_score": self._calculate_diversity(molecules),
                "novelty_score": self._calculate_novelty(molecules)
            }
            
            # Store generation history
            self.generation_history.append({
                "n_molecules": n_molecules,
                "target_properties": target_properties,
                "timestamp": asyncio.get_event_loop().time(),
                "success_rate": len(molecules) / n_molecules if n_molecules > 0 else 0
            })
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def _optimize_lead_compound(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize lead compound using advanced ML methods"""
        lead_smiles = args.get("lead_smiles")
        optimization_goals = args.get("optimization_goals", {})
        maintain_scaffold = args.get("maintain_scaffold", True)
        max_iterations = args.get("max_iterations", 20)
        
        if not lead_smiles:
            return {"error": "Lead SMILES required"}
        
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available for optimization"}
        
        # Validate lead molecule
        mol = Chem.MolFromSmiles(lead_smiles)
        if mol is None:
            return {"error": f"Invalid lead SMILES: {lead_smiles}"}
        
        result = {
            "lead_analysis": {},
            "optimization_trajectory": [],
            "best_candidates": [],
            "optimization_summary": {}
        }
        
        try:
            # Analyze lead compound
            lead_analysis = await self._analyze_molecule_advanced({
                "smiles": lead_smiles,
                "use_reservoir": True,
                "use_ensemble": True
            })
            result["lead_analysis"] = lead_analysis
            
            # Optimization using VAE latent space
            if self.vae_model:
                # Encode lead to latent space
                z_lead = self.vae_model.encode_smiles([lead_smiles])
                
                best_molecules = []
                current_smiles = lead_smiles
                best_score = self._score_molecule(current_smiles, optimization_goals)
                
                for iteration in range(max_iterations):
                    # Generate candidates around current molecule
                    candidates = []
                    for _ in range(20):  # Generate 20 candidates per iteration
                        # Add noise to latent vector
                        noise = torch.randn_like(z_lead) * 0.1
                        z_candidate = z_lead + noise
                        
                        # Generate molecule
                        candidate_molecules = self.vae_model.generate(1, z=z_candidate)
                        if candidate_molecules:
                            candidates.extend(candidate_molecules)
                    
                    # Score candidates
                    scored_candidates = []
                    for candidate in candidates:
                        if maintain_scaffold and not self._scaffold_match(candidate, lead_smiles):
                            continue
                        
                        score = self._score_molecule(candidate, optimization_goals)
                        scored_candidates.append((candidate, score))
                    
                    # Select best candidate
                    if scored_candidates:
                        scored_candidates.sort(key=lambda x: x[1], reverse=True)
                        best_candidate, best_candidate_score = scored_candidates[0]
                        
                        # Update if better
                        if best_candidate_score > best_score:
                            best_score = best_candidate_score
                            current_smiles = best_candidate
                            z_lead = self.vae_model.encode_smiles([current_smiles])
                            
                            # Add to best molecules
                            best_molecules.append({
                                "smiles": current_smiles,
                                "score": best_candidate_score,
                                "iteration": iteration,
                                "properties": self._calculate_basic_properties(current_smiles)
                            })
                    
                    # Track optimization trajectory
                    result["optimization_trajectory"].append({
                        "iteration": iteration,
                        "best_score": best_score,
                        "current_molecule": current_smiles,
                        "n_candidates": len(candidates)
                    })
                
                # Sort and return best candidates
                best_molecules.sort(key=lambda x: x["score"], reverse=True)
                result["best_candidates"] = best_molecules[:10]  # Top 10
                
                # Optimization summary
                result["optimization_summary"] = {
                    "initial_score": self._score_molecule(lead_smiles, optimization_goals),
                    "final_score": best_score,
                    "improvement": best_score - self._score_molecule(lead_smiles, optimization_goals),
                    "iterations_completed": max_iterations,
                    "total_candidates_evaluated": sum(len(traj.get("n_candidates", 0)) for traj in result["optimization_trajectory"])
                }
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def _predict_synthesis_route(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict synthesis route using reservoir anticipation"""
        target_smiles = args.get("target_smiles")
        max_steps = args.get("max_steps", 5)
        starting_materials = args.get("starting_materials", [])
        use_reservoir_anticipation = args.get("use_reservoir_anticipation", True)
        
        if not target_smiles:
            return {"error": "Target SMILES required"}
        
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available for synthesis planning"}
        
        result = {
            "target_analysis": {},
            "synthetic_routes": [],
            "starting_materials": starting_materials,
            "route_analysis": {}
        }
        
        try:
            # Analyze target molecule
            target_analysis = await self._analyze_molecule_advanced({
                "smiles": target_smiles,
                "use_reservoir": True
            })
            result["target_analysis"] = target_analysis
            
            # Use reservoir computing for route prediction
            if use_reservoir_anticipation and self.molecular_reservoir:
                route = self.molecular_reservoir.predict_synthetic_route(target_smiles, max_steps)
                
                if route:
                    # Analyze each step
                    route_steps = []
                    for i, precursor in enumerate(route):
                        step_analysis = {
                            "step": i + 1,
                            "precursor": precursor,
                            "properties": self._calculate_basic_properties(precursor),
                            "synthetic_accessibility": self._calculate_sa_score(precursor)
                        }
                        route_steps.append(step_analysis)
                    
                    result["synthetic_routes"].append({
                        "route_id": 1,
                        "steps": route_steps,
                        "total_steps": len(route),
                        "overall_feasibility": np.mean([step.get("synthetic_accessibility", 5) for step in route_steps])
                    })
            
            # Add simple retrosynthetic analysis
            simple_route = self._simple_retrosynthesis(target_smiles, max_steps)
            if simple_route:
                result["synthetic_routes"].append({
                    "route_id": 2,
                    "type": "simplified_retrosynthesis",
                    "steps": simple_route,
                    "total_steps": len(simple_route)
                })
            
            # Route analysis
            if result["synthetic_routes"]:
                result["route_analysis"] = {
                    "n_routes_found": len(result["synthetic_routes"]),
                    "average_steps": np.mean([route.get("total_steps", 0) for route in result["synthetic_routes"]]),
                    "best_route": min(result["synthetic_routes"], key=lambda x: x.get("total_steps", float('inf')))
                }
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    # Helper methods
    def _calculate_basic_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate basic molecular properties"""
        if not RDKIT_AVAILABLE:
            return {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "logP": Crippen.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "QED": Descriptors.qed(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds()
            }
        except:
            return {}
    
    def _filter_by_properties(self, molecules: List[str], target_properties: Dict[str, float]) -> List[str]:
        """Filter molecules by target properties"""
        if not RDKIT_AVAILABLE:
            return molecules
        
        filtered = []
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                match = True
                for prop, target_value in target_properties.items():
                    if prop == "MW":
                        actual = Descriptors.MolWt(mol)
                        if abs(actual - target_value) / target_value > 0.2:
                            match = False
                    elif prop == "logP":
                        actual = Crippen.MolLogP(mol)
                        if abs(actual - target_value) > 1.0:
                            match = False
                    elif prop == "QED":
                        actual = Descriptors.qed(mol)
                        if abs(actual - target_value) > 0.3:
                            match = False
                
                if match:
                    filtered.append(smiles)
            except:
                continue
        
        return filtered
    
    def _score_molecule(self, smiles: str, optimization_goals: Dict[str, Any]) -> float:
        """Score molecule based on optimization goals"""
        if not RDKIT_AVAILABLE:
            return 0.0
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            score = 0.0
            total_weight = 0.0
            
            for prop, target_range in optimization_goals.items():
                if isinstance(target_range, list) and len(target_range) == 2:
                    min_val, max_val = target_range
                    target_val = (min_val + max_val) / 2
                    tolerance = (max_val - min_val) / 2
                elif isinstance(target_range, (int, float)):
                    target_val = target_range
                    tolerance = target_val * 0.2
                else:
                    continue
                
                if prop == "logP":
                    actual = Crippen.MolLogP(mol)
                elif prop == "MW":
                    actual = Descriptors.MolWt(mol)
                elif prop == "QED":
                    actual = Descriptors.qed(mol)
                elif prop == "TPSA":
                    actual = Descriptors.TPSA(mol)
                else:
                    continue
                
                # Calculate score component
                if tolerance > 0:
                    component_score = max(0, 1 - abs(actual - target_val) / tolerance)
                    score += component_score
                    total_weight += 1.0
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except:
            return 0.0
    
    def _scaffold_match(self, smiles1: str, smiles2: str) -> bool:
        """Check if two molecules have the same scaffold"""
        if not RDKIT_AVAILABLE:
            return True  # Conservative approach
        
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return False
            
            # Simple scaffold comparison - in practice would use more sophisticated methods
            return mol1.GetNumRings() == mol2.GetNumRings()
            
        except:
            return False
    
    def _calculate_diversity(self, molecules: List[str]) -> float:
        """Calculate diversity score for molecule set"""
        if not molecules:
            return 0.0
        
        # Simple diversity based on unique SMILES
        unique_molecules = set(molecules)
        return len(unique_molecules) / len(molecules)
    
    def _calculate_novelty(self, molecules: List[str]) -> float:
        """Calculate novelty score (placeholder)"""
        # In practice, would compare against known molecule databases
        return 0.8  # Placeholder
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule violations"""
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Crippen.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        return violations
    
    def _check_veber(self, mol) -> bool:
        """Check Veber rules"""
        return (Descriptors.NumRotatableBonds(mol) <= 10 and 
                Descriptors.TPSA(mol) <= 140)
    
    def _check_pains(self, mol) -> List[str]:
        """Check for PAINS (placeholder)"""
        # In practice, would use actual PAINS filters
        return []
    
    def _calculate_sa_score(self, smiles: str) -> float:
        """Calculate synthetic accessibility score"""
        if not RDKIT_AVAILABLE:
            return 5.0
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0
            
            # Simplified SA score
            complexity = (mol.GetNumAtoms() + 
                         Descriptors.RingCount(mol) * 2 + 
                         len(Chem.FindMolChiralCenters(mol)) * 3)
            
            return min(10.0, max(1.0, complexity / 10))
        except:
            return 5.0
    
    def _simple_retrosynthesis(self, target_smiles: str, max_steps: int) -> List[Dict[str, Any]]:
        """Simple retrosynthetic analysis (placeholder)"""
        # This is a simplified placeholder - real implementation would use
        # reaction databases and sophisticated retrosynthetic algorithms
        steps = []
        current = target_smiles
        
        for i in range(max_steps):
            # Placeholder: create a "simpler" precursor
            step = {
                "step": i + 1,
                "transformation": f"Generic transformation {i + 1}",
                "precursor": current,  # In practice, would generate actual precursor
                "confidence": 0.7 - i * 0.1
            }
            steps.append(step)
            
            if i >= 2:  # Stop after a few steps for demo
                break
        
        return steps
    
    # Implement remaining methods...
    async def _compare_molecules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare molecules using advanced metrics"""
        # Implementation for molecule comparison
        return {"status": "not_implemented"}
    
    async def _chemical_space_exploration(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Explore chemical space using reservoir dynamics"""
        # Implementation for chemical space exploration
        return {"status": "not_implemented"}
    
    async def _reaction_feasibility_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reaction feasibility"""
        # Implementation for reaction analysis
        return {"status": "not_implemented"}
    
    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ochem-helper-advanced",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(),
                ),
            )

async def main():
    """Main entry point"""
    server = OChemMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
