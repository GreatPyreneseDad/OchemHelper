#!/bin/bash

# Quick run script for OChem Helper

echo "🚀 Starting OChem Helper..."

# Kill any existing processes
pkill -f "uvicorn" 2>/dev/null
pkill -f "simple_mcp_bridge" 2>/dev/null
pkill -f "http.server 8080" 2>/dev/null

# Wait a moment
sleep 1

# Set up environment
export PYTHONPATH=/Users/chris/ochem-helper:$PYTHONPATH

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Start services
echo "Starting services..."

# 1. Start API
echo "1️⃣  OChem API on port 8000..."
cd /Users/chris/ochem-helper
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!

# 2. Start MCP
echo "2️⃣  MCP Server on port 8001..."
cd mcp/server
python simple_mcp_bridge.py --port 8001 > ../../logs/mcp.log 2>&1 &
MCP_PID=$!
cd ../..

# 3. Start Dashboard
echo "3️⃣  Dashboard on port 8080..."
cd dashboard
python -m http.server 8080 > ../logs/dashboard.log 2>&1 &
DASH_PID=$!
cd ..

# Wait and open browser
sleep 3
echo
echo "✅ Services started!"
echo
echo "Opening http://localhost:8080"
open http://localhost:8080 2>/dev/null || xdg-open http://localhost:8080 2>/dev/null || echo "Please open http://localhost:8080"

echo
echo "PIDs: API=$API_PID, MCP=$MCP_PID, Dashboard=$DASH_PID"
echo "Press Ctrl+C to stop"

# Trap and cleanup
trap "echo 'Stopping...'; kill $API_PID $MCP_PID $DASH_PID 2>/dev/null; exit" INT

# Keep running
wait