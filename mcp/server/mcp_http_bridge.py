#!/usr/bin/env python3
"""
HTTP Bridge for MCP Server

This provides an HTTP API interface to the MCP server, allowing
web-based clients (like the dashboard) to interact with MCP tools.
"""

import json
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the MCP server
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ochem_mcp import OChemMCPServer


class ToolListRequest(BaseModel):
    """Request for listing available tools"""
    pass


class ToolCallRequest(BaseModel):
    """Request for calling a specific tool"""
    tool_name: str
    arguments: Dict[str, Any]


class MCPHTTPBridge:
    """HTTP bridge for MCP server"""
    
    def __init__(self):
        self.mcp_server = OChemMCPServer()
        self.app = FastAPI(
            title="OChem MCP HTTP Bridge",
            description="HTTP API for MCP chemistry tools",
            version="1.0.0"
        )
        self.setup_routes()
        self.setup_cors()
    
    def setup_cors(self):
        """Configure CORS for dashboard access"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
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
        
        @self.app.post("/list_tools")
        async def list_tools(request: ToolListRequest):
            """List all available MCP tools"""
            try:
                tools = []
                for tool_name, tool_info in self.mcp_server.tools.items():
                    tools.append({
                        "name": tool_name,
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("input_schema", {})
                    })
                
                return {"tools": tools}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/call_tool")
        async def call_tool(request: ToolCallRequest):
            """Call a specific MCP tool"""
            try:
                # Get the tool
                if request.tool_name not in self.mcp_server.tools:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool '{request.tool_name}' not found"
                    )
                
                # Call the tool
                result = await self.mcp_server.call_tool(
                    request.tool_name,
                    request.arguments
                )
                
                return {"result": result}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "mcp_server": "connected",
                "tools_available": len(self.mcp_server.tools)
            }
        
        @self.app.post("/predict_properties")
        async def predict_properties(smiles: str, properties: List[str] = None):
            """Shortcut endpoint for property prediction"""
            try:
                result = await self.mcp_server.call_tool(
                    "predict_properties",
                    {
                        "smiles": smiles,
                        "properties": properties
                    }
                )
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/suggest_synthesis")
        async def suggest_synthesis(target_smiles: str, max_steps: int = 5):
            """Shortcut endpoint for synthesis suggestion"""
            try:
                result = await self.mcp_server.call_tool(
                    "suggest_synthesis",
                    {
                        "target_smiles": target_smiles,
                        "max_steps": max_steps
                    }
                )
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/optimize_structure")
        async def optimize_structure(
            lead_smiles: str,
            optimization_goals: Dict[str, Any],
            num_molecules: int = 10
        ):
            """Shortcut endpoint for structure optimization"""
            try:
                result = await self.mcp_server.call_tool(
                    "optimize_structure",
                    {
                        "lead_smiles": lead_smiles,
                        "optimization_goals": optimization_goals,
                        "num_molecules": num_molecules
                    }
                )
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the HTTP bridge server"""
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OChem MCP HTTP Bridge")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting OChem MCP HTTP Bridge on {args.host}:{args.port}")
    print(f"Dashboard can connect to: http://localhost:{args.port}")
    print("\nAvailable endpoints:")
    print("  POST /list_tools - List available MCP tools")
    print("  POST /call_tool - Call a specific tool")
    print("  GET  /health - Health check")
    print("\nPress Ctrl+C to stop the server")
    
    bridge = MCPHTTPBridge()
    bridge.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()