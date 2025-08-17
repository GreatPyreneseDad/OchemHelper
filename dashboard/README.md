# OChem Helper Dashboard

An advanced molecular discovery dashboard for visualizing and controlling the OChem Helper platform.

## Features

### Real-time Visualization
- **3D Molecular Structure Viewer**: Interactive 3D visualization using 3Dmol.js
- **Property Radar Charts**: Multi-parameter molecular property visualization
- **Chemical Space Explorer**: t-SNE visualization of molecular embeddings
- **ADMET Profile**: Real-time ADMET property predictions

### Performance Metrics
- Molecules generated counter
- Average QED score tracking
- Validity rate monitoring
- Synthesis feasibility scoring

### Control Panel
- SMILES input for molecular generation
- Generation mode selection (random, similar, optimize, interpolate)
- Target property configuration
- Batch size control

### Activity Feed
- Real-time activity logging
- Generation history
- Synthesis route calculations
- Property predictions

## Usage

### Standalone Mode
Simply open `index.html` in a modern web browser:
```bash
open dashboard/index.html
```

### With API Integration
To connect to the OChem Helper API:

1. Start the API server:
```bash
cd src
uvicorn api.app:app --reload
```

2. Update the dashboard configuration in the HTML:
```javascript
const API_URL = 'http://localhost:8000';
```

### Serving via Web Server
For production deployment:

```bash
# Using Python
python -m http.server 8080 --directory dashboard

# Using Node.js
npx http-server dashboard -p 8080

# Using Nginx
server {
    location /dashboard {
        root /path/to/ochem-helper;
        try_files $uri $uri/ /dashboard/index.html;
    }
}
```

## API Integration

To connect the dashboard to your OChem Helper API, add these functions:

```javascript
// Generate molecules via API
async function generateMoleculesAPI(params) {
    const response = await fetch(`${API_URL}/api/v1/generate`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    });
    return response.json();
}

// Predict properties
async function predictPropertiesAPI(smiles) {
    const response = await fetch(`${API_URL}/api/v1/predict/properties`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            molecules: [smiles],
            properties: ['logP', 'MW', 'TPSA', 'QED']
        })
    });
    return response.json();
}
```

## Customization

### Theme Colors
Modify the CSS variables in the `<style>` section:
```css
/* Primary colors */
--primary-gradient: linear-gradient(135deg, #63b3ed 0%, #4c1d95 100%);
--background-dark: #0a0e27;
--background-light: #151932;
--text-primary: #e0e6ed;
--text-secondary: #94a3b8;
```

### Chart Configuration
Customize Plotly charts in the `initCharts()` function:
```javascript
const customLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {color: '#e0e6ed'},
    // Add your customizations
};
```

### 3D Viewer Settings
Modify 3Dmol viewer settings:
```javascript
viewer.setStyle({}, {
    stick: {radius: 0.15},
    sphere: {radius: 0.3},
    cartoon: {color: 'spectrum'}
});
```

## Browser Compatibility

- Chrome 90+ ✓
- Firefox 88+ ✓
- Safari 14+ ✓
- Edge 90+ ✓

Requires:
- JavaScript ES6+
- WebGL support
- CSS Grid/Flexbox

## Performance Optimization

For large datasets:

1. **Implement pagination** for activity logs
2. **Use WebWorkers** for data processing
3. **Enable GPU acceleration** for 3D rendering
4. **Implement virtual scrolling** for large lists

## Security Considerations

For production deployment:

1. **Enable CORS** properly on API server
2. **Implement authentication** for API calls
3. **Sanitize user inputs** before display
4. **Use HTTPS** for all connections
5. **Add CSP headers** for XSS protection

## Extending the Dashboard

### Adding New Charts
```javascript
function addCustomChart() {
    const data = [{
        // Your data configuration
    }];
    
    const layout = {
        // Your layout configuration
    };
    
    Plotly.newPlot('chartId', data, layout);
}
```

### Adding New Metrics
```html
<div class="metric-card">
    <div class="metric-value" id="newMetric">0</div>
    <div class="metric-label">New Metric</div>
    <div class="metric-change positive">↑ 0%</div>
</div>
```

### WebSocket Integration
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateDashboard(data);
};
```

## Troubleshooting

### Charts not displaying
- Check browser console for errors
- Ensure Plotly.js is loaded
- Verify data format is correct

### 3D viewer issues
- Check WebGL support
- Ensure 3Dmol.js is loaded
- Try different molecular formats

### Performance issues
- Reduce update frequency
- Limit data points displayed
- Enable hardware acceleration

## License

Part of the OChem Helper project - MIT License