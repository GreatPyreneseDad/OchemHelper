#!/bin/bash

# Launch OChem Helper - All Services Together

echo "🚀 Launching OChem Helper Complete System..."
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing processes on our ports
echo "Cleaning up any existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 1

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install any missing dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
pip install -q torch rdkit-pypi fastapi uvicorn pydantic numpy pandas scikit-learn mcp 2>/dev/null

# Start the OChem API server
echo -e "${GREEN}Starting OChem API on port 8000...${NC}"
cd src
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000 > ../logs/api.log 2>&1 &
API_PID=$!
cd ..

# Wait for API to start
sleep 3

# Start the MCP server
echo -e "${GREEN}Starting MCP Server on port 8001...${NC}"
cd mcp/server
python3 simple_mcp_bridge.py --port 8001 > ../../logs/mcp.log 2>&1 &
MCP_PID=$!
cd ../..

# Wait for MCP to start
sleep 2

# Start a simple HTTP server for the dashboard
echo -e "${GREEN}Starting Dashboard Server on port 8080...${NC}"
cd dashboard
python3 -m http.server 8080 > ../logs/dashboard.log 2>&1 &
DASH_PID=$!
cd ..

# Function to check if a service is running
check_service() {
    local port=$1
    local name=$2
    if curl -s http://localhost:$port/health >/dev/null 2>&1 || curl -s http://localhost:$port/ >/dev/null 2>&1; then
        echo -e "${GREEN}✓ $name is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ $name may not be ready yet${NC}"
        return 1
    fi
}

# Wait a moment for services to start
sleep 3

# Check all services
echo
echo "Checking services..."
check_service 8000 "OChem API"
check_service 8001 "MCP Server"
check_service 8080 "Dashboard"

echo
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 OChem Helper is ready!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo
echo "Services running:"
echo -e "  ${BLUE}📊 Dashboard:${NC} http://localhost:8080"
echo -e "  ${BLUE}🧪 OChem API:${NC} http://localhost:8000 (PID: $API_PID)"
echo -e "  ${BLUE}🔧 MCP Server:${NC} http://localhost:8001 (PID: $MCP_PID)"
echo
echo "Quick links:"
echo -e "  ${YELLOW}Main Dashboard:${NC} http://localhost:8080/index.html"
echo -e "  ${YELLOW}System Test:${NC} http://localhost:8080/test-full-system.html"
echo -e "  ${YELLOW}3D Viewer Test:${NC} http://localhost:8080/test-3d-viewer.html"
echo -e "  ${YELLOW}API Docs:${NC} http://localhost:8000/docs"
echo
echo "Logs are being written to:"
echo "  - logs/api.log (OChem API)"
echo "  - logs/mcp.log (MCP Server)"
echo "  - logs/dashboard.log (Dashboard Server)"
echo
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo

# Function to cleanup on exit
cleanup() {
    echo
    echo "Stopping all services..."
    kill $API_PID $MCP_PID $DASH_PID 2>/dev/null
    echo "All services stopped."
    exit
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT

# Keep the script running and show logs
tail -f logs/api.log logs/mcp.log logs/dashboard.log 2>/dev/null || {
    # If logs don't exist yet, just wait
    while true; do
        sleep 1
    done
}