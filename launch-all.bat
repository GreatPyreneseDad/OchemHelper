@echo off
REM Launch OChem Helper - All Services Together (Windows)

echo Launching OChem Helper Complete System...
echo.

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Checking dependencies...
pip install -q torch rdkit-pypi fastapi uvicorn pydantic numpy pandas scikit-learn mcp 2>nul

REM Start the OChem API server
echo Starting OChem API on port 8000...
start /B cmd /c "cd src && uvicorn api.app:app --reload --host 0.0.0.0 --port 8000 > ..\logs\api.log 2>&1"

REM Wait for API to start
timeout /t 3 /nobreak >nul

REM Start the MCP server
echo Starting MCP Server on port 8001...
start /B cmd /c "cd mcp\server && python simple_mcp_bridge.py --port 8001 > ..\..\logs\mcp.log 2>&1"

REM Wait for MCP to start
timeout /t 2 /nobreak >nul

REM Start dashboard server
echo Starting Dashboard Server on port 8080...
start /B cmd /c "cd dashboard && python -m http.server 8080 > ..\logs\dashboard.log 2>&1"

REM Wait for services to start
timeout /t 3 /nobreak >nul

echo.
echo ===============================================
echo OChem Helper is ready!
echo ===============================================
echo.
echo Services running:
echo   Dashboard: http://localhost:8080
echo   OChem API: http://localhost:8000
echo   MCP Server: http://localhost:8001
echo.
echo Quick links:
echo   Main Dashboard: http://localhost:8080/index.html
echo   System Test: http://localhost:8080/test-full-system.html
echo   3D Viewer Test: http://localhost:8080/test-3d-viewer.html
echo   API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop all services
echo.

REM Open browser
start http://localhost:8080/index.html

REM Keep window open
pause