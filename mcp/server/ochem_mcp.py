#!/usr/bin/env python3
"""OChem Helper MCP Server for AI integration."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import our chemistry tools
from ..tools import (
    analyze_molecule,
    predict_properties,
    suggest_synthesis,
    optimize_structure,
    retrosynthesis,
    reaction_prediction
)


class OChemMCPServer:
    """MCP server for organic chemistry AI assistance."""
    
    def __init__(self):
        self.server = Server("ochem-helper")
        self.setup_handlers()
        
    def setup_handlers(self):
        """Set up MCP tool handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available chemistry tools."""
            return [
                types.Tool(
                    name="analyze_molecule",
                    description="Analyze a molecule's structure and properties from SMILES",
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
                                "description": "Properties to calculate (e.g., logP, MW, TPSA)"
                            }
                        },
                        "required": ["smiles"]
                    }
                ),
                types.Tool(
                    name="predict_activity",
                    description="Predict biological activity and ADMET properties",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "smiles": {
                                "type": "string",
                                "description": "SMILES string of the molecule"
                            },
                            "targets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Biological targets or ADMET endpoints"
                            }
                        },
                        "required": ["smiles"]
                    }
                ),
                types.Tool(
                    name="suggest_synthesis",
                    description="Suggest synthetic routes for a target molecule",
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
                                "default": 5
                            },
                            "starting_materials": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Preferred starting materials (SMILES)"
                            }
                        },
                        "required": ["target_smiles"]
                    }
                ),
                types.Tool(
                    name="optimize_lead",
                    description="Optimize a lead compound for better properties",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lead_smiles": {
                                "type": "string",
                                "description": "SMILES of lead compound"
                            },
                            "optimization_goals": {
                                "type": "object",
                                "description": "Target property ranges (e.g., {'logP': [2, 4]})"
                            },
                            "maintain_scaffold": {
                                "type": "boolean",
                                "description": "Keep core scaffold intact",
                                "default": True
                            }
                        },
                        "required": ["lead_smiles", "optimization_goals"]
                    }
                ),
                types.Tool(
                    name="reaction_feasibility",
                    description="Check if a chemical reaction is feasible",
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
                            "conditions": {
                                "type": "object",
                                "description": "Reaction conditions (solvent, temperature, etc.)"
                            }
                        },
                        "required": ["reactants", "products"]
                    }
                ),
                types.Tool(
                    name="generate_analogs",
                    description="Generate analog molecules with similar properties",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "reference_smiles": {
                                "type": "string",
                                "description": "SMILES of reference molecule"
                            },
                            "num_analogs": {
                                "type": "integer",
                                "description": "Number of analogs to generate",
                                "default": 10
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Tanimoto similarity threshold (0-1)",
                                "default": 0.7
                            }
                        },
                        "required": ["reference_smiles"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]]
        ) -> List[types.TextContent]:
            """Handle tool execution."""
            try:
                if name == "analyze_molecule":
                    result = await analyze_molecule.analyze(
                        arguments.get("smiles"),
                        arguments.get("properties", ["MW", "logP", "TPSA", "HBD", "HBA"])
                    )
                
                elif name == "predict_activity":
                    result = await predict_properties.predict_activity(
                        arguments.get("smiles"),
                        arguments.get("targets", ["bioavailability", "toxicity"])
                    )
                
                elif name == "suggest_synthesis":
                    result = await suggest_synthesis.retrosynthetic_analysis(
                        arguments.get("target_smiles"),
                        max_steps=arguments.get("max_steps", 5),
                        starting_materials=arguments.get("starting_materials", [])
                    )
                
                elif name == "optimize_lead":
                    result = await optimize_structure.optimize_lead(
                        arguments.get("lead_smiles"),
                        arguments.get("optimization_goals"),
                        maintain_scaffold=arguments.get("maintain_scaffold", True)
                    )
                
                elif name == "reaction_feasibility":
                    result = await reaction_prediction.check_feasibility(
                        arguments.get("reactants"),
                        arguments.get("products"),
                        conditions=arguments.get("conditions", {})
                    )
                
                elif name == "generate_analogs":
                    result = await optimize_structure.generate_analogs(
                        arguments.get("reference_smiles"),
                        num_analogs=arguments.get("num_analogs", 10),
                        similarity_threshold=arguments.get("similarity_threshold", 0.7)
                    )
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
                
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            """List available prompts for chemistry tasks."""
            return [
                types.Prompt(
                    name="drug_discovery",
                    description="Guide for drug discovery workflow",
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
                        )
                    ]
                ),
                types.Prompt(
                    name="lead_optimization",
                    description="Optimize a lead compound systematically",
                    arguments=[
                        types.PromptArgument(
                            name="lead_smiles",
                            description="SMILES of lead compound",
                            required=True
                        ),
                        types.PromptArgument(
                            name="issues",
                            description="Issues to address (e.g., 'low solubility')",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="synthesis_planning",
                    description="Plan synthesis route for complex molecule",
                    arguments=[
                        types.PromptArgument(
                            name="target_smiles",
                            description="SMILES of target molecule",
                            required=True
                        ),
                        types.PromptArgument(
                            name="constraints",
                            description="Synthesis constraints",
                            required=False
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str, arguments: Optional[Dict[str, str]]
        ) -> types.GetPromptResult:
            """Get specific prompt template."""
            if name == "drug_discovery":
                target = arguments.get("target", "unspecified target")
                indication = arguments.get("indication", "")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""I need help with drug discovery for {target}.
{f'The indication is {indication}.' if indication else ''}

Please help me:
1. Analyze known inhibitors/modulators
2. Identify key pharmacophore features
3. Suggest novel scaffolds
4. Optimize for drug-like properties
5. Check for potential off-targets"""
                        )
                    )
                ]
            
            elif name == "lead_optimization":
                lead_smiles = arguments.get("lead_smiles", "")
                issues = arguments.get("issues", "")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""I have a lead compound that needs optimization:
SMILES: {lead_smiles}
Issues: {issues}

Please:
1. Analyze the current structure and properties
2. Identify problematic features
3. Suggest specific modifications
4. Generate optimized analogs
5. Predict improved properties"""
                        )
                    )
                ]
            
            elif name == "synthesis_planning":
                target_smiles = arguments.get("target_smiles", "")
                constraints = arguments.get("constraints", "None specified")
                
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"""Plan synthesis for this target molecule:
SMILES: {target_smiles}
Constraints: {constraints}

Please provide:
1. Retrosynthetic analysis
2. Key disconnections
3. Suggested synthetic routes
4. Starting materials
5. Key transformations and conditions"""
                        )
                    )
                ]
            
            else:
                raise ValueError(f"Unknown prompt: {name}")
            
            return types.GetPromptResult(
                description=f"Prompt for {name}",
                messages=messages
            )
    
    async def run(self):
        """Run the MCP server."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ochem-helper",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Main entry point."""
    server = OChemMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())