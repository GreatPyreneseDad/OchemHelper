#!/bin/bash

echo "Testing OChem Helper Services..."
echo

# Test OChem API
echo "1. Testing OChem API (http://localhost:8000)..."
curl -s http://localhost:8000/ | jq . || echo "OChem API not responding"
echo

# Test molecule generation
echo "2. Testing molecule generation..."
curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"n_molecules": 3, "mode": "random"}' | jq . || echo "Generation failed"
echo

# Test MCP server
echo "3. Testing MCP Server (http://localhost:8001)..."
curl -s http://localhost:8001/health | jq . || echo "MCP Server not responding"
echo

# Test MCP tool
echo "4. Testing MCP predict_properties tool..."
curl -s -X POST http://localhost:8001/call_tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "predict_properties", "arguments": {"smiles": "CCO"}}' | jq . || echo "MCP tool call failed"
echo

echo "All tests complete!"