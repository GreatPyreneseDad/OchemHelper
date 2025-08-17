#!/bin/bash

# Start OChem Helper Backend Services

echo "Starting OChem Helper Backend..."

# Activate virtual environment
source venv/bin/activate

# Install any missing dependencies
echo "Checking dependencies..."
pip install -q torch rdkit-pypi fastapi uvicorn pydantic numpy pandas scikit-learn

# Start the API server
echo "Starting OChem API on port 8000..."
cd src
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

cd ..

# Start the MCP server
echo "Starting MCP Server on port 8001..."
cd mcp/server
python3 simple_mcp_bridge.py --port 8001 &
MCP_PID=$!

cd ../..

echo ""
echo "Services started:"
echo "  OChem API: http://localhost:8000 (PID: $API_PID)"
echo "  MCP Server: http://localhost:8001 (PID: $MCP_PID)"
echo ""
echo "To test the system, open: http://localhost:8080/test-full-system.html"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $API_PID $MCP_PID 2>/dev/null; exit" INT
wait