#!/bin/bash

# Startup script for OChem Helper with AI Integration

echo "Starting OChem Helper Services..."

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Start the main API server
if check_port 8000; then
    echo "Starting OChem API server on port 8000..."
    cd ../src
    uvicorn api.app:app --reload --port 8000 &
    API_PID=$!
    echo "API server PID: $API_PID"
else
    echo "OChem API server already running on port 8000"
fi

# Start the MCP HTTP bridge
if check_port 8001; then
    echo "Starting MCP HTTP Bridge on port 8001..."
    cd ../mcp/server
    source ../../venv/bin/activate
    python3 simple_mcp_bridge.py --port 8001 &
    MCP_PID=$!
    echo "MCP Bridge PID: $MCP_PID"
    deactivate
else
    echo "MCP HTTP Bridge already running on port 8001"
fi

# Start a simple HTTP server for the dashboard
if check_port 8080; then
    echo "Starting dashboard server on port 8080..."
    cd ../../dashboard
    python -m http.server 8080 &
    DASH_PID=$!
    echo "Dashboard server PID: $DASH_PID"
else
    echo "Dashboard server already running on port 8080"
fi

echo ""
echo "Services starting up..."
echo "Please wait a few seconds for all services to initialize."
echo ""
echo "Access the dashboard at: http://localhost:8080"
echo ""
echo "Service URLs:"
echo "  Dashboard: http://localhost:8080"
echo "  OChem API: http://localhost:8000/docs"
echo "  MCP Bridge: http://localhost:8001"
echo ""
echo "The AI chat will appear in the bottom right corner of the dashboard."
echo "It's connected to Claude API and the MCP server for advanced chemistry operations."
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $API_PID $MCP_PID $DASH_PID 2>/dev/null; exit" INT
wait