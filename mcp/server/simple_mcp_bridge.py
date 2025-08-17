#!/usr/bin/env python3
"""
Simple MCP HTTP Bridge for OChem Helper

This provides a basic HTTP API for the dashboard to interact with chemistry tools.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import chemistry tools directly
try:
    from src.core.descriptors import MolecularDescriptors
    from src.core.validators import SMILESValidator
    print("Successfully imported chemistry modules")
except ImportError as e:
    print(f"Warning: Could not import chemistry modules: {e}")
    # Create mock classes for testing
    class MolecularDescriptors:
        def calculate_all(self, smiles):
            return {
                "MW": 180.16,
                "logP": 2.3,
                "TPSA": 40.5,
                "QED": 0.82,
                "SA": 2.5
            }
    
    class SMILESValidator:
        def validate(self, smiles):
            return len(smiles) > 0 and smiles.count('(') == smiles.count(')')


class ToolCallRequest(BaseModel):
    """Request for calling a specific tool"""
    tool_name: str
    arguments: Dict[str, Any]


class SimpleMCPBridge:
    """Simple HTTP bridge for MCP-like functionality"""
    
    def __init__(self):
        self.app = FastAPI(
            title="OChem MCP HTTP Bridge",
            description="HTTP API for chemistry tools",
            version="1.0.0"
        )
        self.descriptors = MolecularDescriptors()
        self.validator = SMILESValidator()
        self.setup_routes()
        self.setup_cors()
    
    def setup_cors(self):
        """Configure CORS for dashboard access"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Set up HTTP routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "OChem MCP HTTP Bridge",
                "status": "running",
                "endpoints": [
                    "/list_tools",
                    "/call_tool",
                    "/health"
                ]
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "tools_available": len(self.get_available_tools())
            }
        
        @self.app.post("/list_tools")
        async def list_tools():
            """List all available MCP tools"""
            tools = []
            for tool_name, tool_info in self.get_available_tools().items():
                tools.append({
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": tool_info.get("parameters", {})
                })
            return {"tools": tools}
        
        @self.app.post("/call_tool")
        async def call_tool(request: ToolCallRequest):
            """Call a specific MCP tool"""
            try:
                tool_name = request.tool_name
                args = request.arguments
                
                if tool_name == "predict_properties":
                    result = await self.predict_properties(**args)
                elif tool_name == "suggest_synthesis":
                    result = await self.suggest_synthesis(**args)
                elif tool_name == "optimize_structure":
                    result = await self.optimize_structure(**args)
                elif tool_name == "reaction_prediction":
                    result = await self.reaction_prediction(**args)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool '{tool_name}' not found"
                    )
                
                return {"result": result}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get available chemistry tools"""
        return {
            "predict_properties": {
                "description": "Predict molecular properties and ADMET profiles",
                "parameters": {
                    "smiles": "SMILES string of the molecule",
                    "properties": "List of properties to predict (optional)"
                }
            },
            "suggest_synthesis": {
                "description": "Suggest retrosynthetic routes for a target molecule",
                "parameters": {
                    "target_smiles": "SMILES string of the target molecule",
                    "max_steps": "Maximum number of synthesis steps (default: 5)"
                }
            },
            "optimize_structure": {
                "description": "Generate optimized molecular structures",
                "parameters": {
                    "lead_smiles": "SMILES string of the lead compound",
                    "optimization_goals": "Dictionary of property targets",
                    "num_molecules": "Number of molecules to generate (default: 10)"
                }
            },
            "reaction_prediction": {
                "description": "Predict reaction feasibility and products",
                "parameters": {
                    "reactants": "List of reactant SMILES",
                    "products": "List of product SMILES"
                }
            }
        }
    
    async def predict_properties(self, smiles: str, properties: Optional[List[str]] = None):
        """Predict molecular properties"""
        # Validate SMILES
        if not self.validator.validate(smiles):
            return {"error": "Invalid SMILES string"}
        
        # Calculate properties
        all_props = self.descriptors.calculate_all(smiles)
        
        # Filter if specific properties requested
        if properties:
            filtered = {k: v for k, v in all_props.items() if k in properties}
            return {
                "smiles": smiles,
                "properties": filtered
            }
        
        # Add ADMET predictions (mock for now)
        admet = {
            "absorption": {"probability": 0.85, "category": "high"},
            "distribution": {"probability": 0.72, "category": "moderate"},
            "metabolism": {"probability": 0.91, "category": "high"},
            "excretion": {"probability": 0.68, "category": "moderate"},
            "toxicity": {"probability": 0.12, "category": "low"}
        }
        
        return {
            "smiles": smiles,
            "properties": all_props,
            "admet_properties": admet
        }
    
    async def suggest_synthesis(self, target_smiles: str, max_steps: int = 5):
        """Suggest synthesis routes (mock implementation)"""
        # Validate SMILES
        if not self.validator.validate(target_smiles):
            return {"error": "Invalid target SMILES"}
        
        # Mock retrosynthetic analysis
        routes = [
            {
                "route_id": 1,
                "confidence": 0.85,
                "steps": [
                    {
                        "step": 1,
                        "reaction": "Friedel-Crafts acylation",
                        "reactants": ["c1ccccc1", "CC(=O)Cl"],
                        "product": "CC(=O)c1ccccc1",
                        "conditions": "AlCl3, DCM, 0°C"
                    },
                    {
                        "step": 2,
                        "reaction": "Reduction",
                        "reactants": ["CC(=O)c1ccccc1"],
                        "product": target_smiles,
                        "conditions": "NaBH4, MeOH, RT"
                    }
                ],
                "total_yield": 0.72
            }
        ]
        
        return {
            "target_smiles": target_smiles,
            "synthesis_routes": routes,
            "num_routes": len(routes)
        }
    
    async def optimize_structure(self, lead_smiles: str, optimization_goals: Dict[str, Any], num_molecules: int = 10):
        """Optimize molecular structure (mock implementation)"""
        # Validate SMILES
        if not self.validator.validate(lead_smiles):
            return {"error": "Invalid lead SMILES"}
        
        # Mock optimization results
        optimized = []
        for i in range(min(num_molecules, 5)):
            # Generate variations (mock)
            if i == 0:
                smiles = lead_smiles
            else:
                # Add simple modifications
                smiles = lead_smiles.replace("c1ccccc1", f"c1ccc(C)cc1") if "c1ccccc1" in lead_smiles else lead_smiles + "C"
            
            props = self.descriptors.calculate_all(smiles)
            optimized.append({
                "smiles": smiles,
                "properties": props,
                "similarity": 1.0 - (i * 0.1)
            })
        
        return {
            "lead_smiles": lead_smiles,
            "optimization_goals": optimization_goals,
            "optimized_molecules": optimized
        }
    
    async def reaction_prediction(self, reactants: List[str], products: List[str]):
        """Predict reaction feasibility (mock implementation)"""
        # Validate all SMILES
        for smiles in reactants + products:
            if not self.validator.validate(smiles):
                return {"error": f"Invalid SMILES: {smiles}"}
        
        # Mock prediction
        return {
            "reactants": reactants,
            "products": products,
            "feasibility": 0.82,
            "reaction_type": "Nucleophilic substitution",
            "conditions": {
                "temperature": "25-50°C",
                "solvent": "DMF",
                "catalyst": "None required"
            },
            "mechanism": "SN2",
            "yield_estimate": 0.75
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the HTTP bridge server"""
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OChem MCP HTTP Bridge")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    
    args = parser.parse_args()
    
    print(f"Starting Simple MCP HTTP Bridge on {args.host}:{args.port}")
    print(f"Dashboard can connect to: http://localhost:{args.port}")
    print("\nAvailable endpoints:")
    print("  POST /list_tools - List available MCP tools")
    print("  POST /call_tool - Call a specific tool")
    print("  GET  /health - Health check")
    print("\nPress Ctrl+C to stop the server")
    
    bridge = SimpleMCPBridge()
    bridge.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()