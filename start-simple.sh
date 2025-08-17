#!/bin/bash

# Simple startup script that handles missing dependencies

echo "🚀 Starting OChem Helper..."
echo

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to check if a Python module is installed
check_module() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Check critical dependencies
echo "Checking dependencies..."
MISSING_DEPS=()

if ! check_module "fastapi"; then
    MISSING_DEPS+=("fastapi uvicorn")
fi

if ! check_module "numpy"; then
    MISSING_DEPS+=("numpy pandas scikit-learn")
fi

if ! check_module "rdkit"; then
    MISSING_DEPS+=("rdkit")
fi

if ! check_module "torch"; then
    echo "⚠️  PyTorch not found - using mock VAE model"
    export USE_MOCK_VAE=1
fi

# Install missing dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Installing missing dependencies: ${MISSING_DEPS[@]}"
    pip install ${MISSING_DEPS[@]}
fi

# Start services
echo
echo "Starting services..."

# Start OChem API
echo "1️⃣  Starting OChem API on port 8000..."
cd src
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000 > ../logs/api.log 2>&1 &
API_PID=$!
cd ..

sleep 2

# Start MCP Server
echo "2️⃣  Starting MCP Server on port 8001..."
cd mcp/server
python simple_mcp_bridge.py --port 8001 > ../../logs/mcp.log 2>&1 &
MCP_PID=$!
cd ../..

sleep 2

# Start Dashboard
echo "3️⃣  Starting Dashboard on port 8080..."
cd dashboard
python -m http.server 8080 > ../logs/dashboard.log 2>&1 &
DASH_PID=$!
cd ..

sleep 2

# Open browser
echo
echo "✅ All services started!"
echo
echo "Opening dashboard in your browser..."
open http://localhost:8080 || xdg-open http://localhost:8080 || echo "Please open http://localhost:8080 in your browser"

echo
echo "Services:"
echo "  📊 Dashboard: http://localhost:8080"
echo "  🧪 OChem API: http://localhost:8000 (PID: $API_PID)"
echo "  🔧 MCP Server: http://localhost:8001 (PID: $MCP_PID)"
echo
echo "Press Ctrl+C to stop all services"

# Cleanup function
cleanup() {
    echo
    echo "Stopping services..."
    kill $API_PID $MCP_PID $DASH_PID 2>/dev/null
    exit
}

trap cleanup INT

# Keep running
wait