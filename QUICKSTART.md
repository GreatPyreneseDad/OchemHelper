# 🚀 OChem Helper Quick Start Guide

## One-Command Launch

### macOS/Linux:
```bash
./launch-all.sh
```

### Windows:
```cmd
launch-all.bat
```

This single command will:
1. ✅ Start the OChem API server (port 8000)
2. ✅ Start the MCP chemistry tools server (port 8001)  
3. ✅ Start the dashboard web server (port 8080)
4. ✅ Open your browser to the dashboard

## 🎯 What You'll See

Once launched, you'll have access to:

- **Main Dashboard**: http://localhost:8080
  - AI chat interface
  - Molecule generator
  - 3D structure viewer
  - Property predictions
  - Activity log

- **Test Pages**:
  - System Test: http://localhost:8080/test-full-system.html
  - 3D Viewer Test: http://localhost:8080/test-3d-viewer.html

- **API Documentation**: http://localhost:8000/docs

## 🧪 Try These Features

### 1. Generate Molecules
- Click "Generate Molecules" button
- Or tell the AI: "Generate 5 drug-like molecules"

### 2. View 3D Structures
- Enter a SMILES string (e.g., `CCO` for ethanol)
- Click "Update Viewer" to see the 3D structure

### 3. Use AI Chat
- "Set the input SMILES to c1ccccc1" (benzene)
- "Generate molecules similar to aspirin"
- "What are the properties of the current molecule?"
- "Optimize this structure for better solubility"

### 4. Test Chemistry Tools
- "Predict the synthesis route for the current molecule"
- "What are the ADMET properties?"
- "Find molecules similar to caffeine"

## 🔑 API Key Setup (Optional)

To enable Claude AI responses:
1. Get an API key from https://console.anthropic.com
2. Open http://localhost:8080/setup-api-key.html
3. Enter your API key and save

## 🛑 Stopping Services

Press `Ctrl+C` in the terminal where you ran the launch script.

## 📋 Requirements

- Python 3.8+
- Node.js (for npm packages)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🆘 Troubleshooting

If services don't start:
1. Check if ports 8000, 8001, 8080 are available
2. Ensure virtual environment is activated
3. Install missing dependencies: `pip install -r requirements.txt`
4. Check logs in the `logs/` directory

## 📝 Logs

Service logs are saved to:
- `logs/api.log` - OChem API server
- `logs/mcp.log` - MCP chemistry tools  
- `logs/dashboard.log` - Dashboard web server

---
Ready to explore? Run `./launch-all.sh` and start discovering molecules! 🧬