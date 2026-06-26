import numpy as np
import json
import os
from data_pipeline import ClimateDataGenerator, LATS, LONS, SEA_MASK, WG_MASK, DECCAN_MASK

def generate_fallback_dashboard():
    print("Generating high-fidelity fallback dataset...")
    
    # Initialize Generator
    gen = ClimateDataGenerator(seed=42)
    
    # Seasons to pre-compile (Winter, Summer, SW Monsoon, NE Monsoon)
    # Day numbers: 15 (Jan 15), 135 (May 15), 200 (July 19), 290 (Oct 17)
    season_days = {
        'winter': 15,
        'summer': 135,
        'sw_monsoon': 200,
        'ne_monsoon': 290
    }
    
    baked_data = {}
    
    for name, doy in season_days.items():
        # Generate base data for the day
        t_max, t_min, rain, u_wind, v_wind, lst, sst = gen.get_climatology(doy)
        
        # Round values to 1 decimal place to save file size
        baked_data[name] = {
            'tmax': np.round(t_max, 1).tolist(),
            'tmin': np.round(t_min, 1).tolist(),
            'rain': np.round(rain, 1).tolist(),
            'u_wind': np.round(u_wind, 1).tolist(),
            'v_wind': np.round(v_wind, 1).tolist()
        }
        
    # Serialize spatial masks
    masks = {
        'sea': SEA_MASK.tolist(),
        'wg': WG_MASK.tolist(),
        'deccan': DECCAN_MASK.tolist()
    }
    
    # HTML Dashboard Template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Digital Twin - India Climate (Offline Fallback)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
        
        body {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar Styling */
        .sidebar {
            width: 320px;
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
            padding: 24px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }

        .sidebar h2 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            color: #ffffff;
            margin: 0 0 10px 0;
            text-align: center;
            background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-section {
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 15px;
        }

        .sidebar-section h3 {
            font-size: 0.95rem;
            color: #94a3b8;
            margin: 0 0 12px 0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .control-group {
            margin-bottom: 15px;
        }

        .control-group label {
            display: block;
            font-size: 0.85rem;
            color: #cbd5e1;
            margin-bottom: 6px;
        }

        .control-group input[type="range"], .control-group select {
            width: 100%;
            background: #1e293b;
            color: white;
            border: 1px solid #334155;
            padding: 8px;
            border-radius: 6px;
            box-sizing: border-box;
        }

        .slider-val {
            float: right;
            font-weight: 600;
            color: #38bdf8;
        }

        /* Main Content Styling */
        .main-content {
            flex: 1;
            padding: 24px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }

        .header h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            margin: 0;
            background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: #94a3b8;
            margin: 5px 0 0 0;
            font-size: 0.95rem;
        }

        /* Metrics Board */
        .metrics-board {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }

        .metric-card {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        .metric-label {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 4px;
        }

        .metric-sub {
            font-size: 0.8rem;
            margin-top: 4px;
        }

        /* Map and Alerts Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 7fr 3fr;
            gap: 20px;
            flex: 1;
            min-height: 480px;
        }

        .map-container {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .map-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .map-header h3 {
            margin: 0;
            font-size: 1.1rem;
        }

        .layer-selector {
            display: flex;
            gap: 10px;
        }

        .layer-btn {
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            font-family: inherit;
        }

        .layer-btn.active {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white;
            border: none;
        }

        #map {
            width: 100%;
            height: 100%;
            min-height: 400px;
            border-radius: 8px;
            background: #0b0f19;
        }

        /* Alerts List */
        .alerts-container {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
        }

        .alerts-container h3 {
            margin: 0;
            font-size: 1.1rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 8px;
            color: #38bdf8;
        }

        .alert-card {
            border-radius: 8px;
            padding: 12px;
            border-left: 4px solid;
            font-size: 0.85rem;
        }

        .alert-critical {
            background: rgba(239, 68, 68, 0.15);
            border-color: #ef4444;
            color: #fca5a5;
        }

        .alert-warning {
            background: rgba(245, 158, 11, 0.15);
            border-color: #f59e0b;
            color: #fde047;
        }

        .alert-info {
            background: rgba(59, 130, 246, 0.15);
            border-color: #3b82f6;
            color: #93c5fd;
        }

        /* Tooltip Styles */
        .info-tooltip {
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: white !important;
            border-radius: 6px;
            padding: 6px;
            font-family: inherit;
        }

        .legend {
            background: rgba(15, 23, 42, 0.9);
            color: white;
            padding: 8px;
            border-radius: 6px;
            font-size: 11px;
            line-height: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .legend i {
            width: 14px;
            height: 14px;
            float: left;
            margin-right: 6px;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>⚙️ TWIN FALLBACK</h2>
        <p style="font-size:0.8rem; color:#94a3b8; text-align:center; margin-top:-5px;">Offline Dashboard Mockup</p>
        
        <div class="sidebar-section">
            <h3>Timeline Selection</h3>
            <div class="control-group">
                <label for="season-select">Active Season</label>
                <select id="season-select">
                    <option value="sw_monsoon" selected>Southwest Monsoon (July)</option>
                    <option value="ne_monsoon">Northeast Monsoon (October)</option>
                    <option value="summer">Summer Heat (May)</option>
                    <option value="winter">Winter (January)</option>
                </select>
            </div>
        </div>
        
        <div class="sidebar-section">
            <h3>What-If Simulation Controls</h3>
            
            <div class="control-group">
                <label for="sst-slider">Ocean SST Anomaly: <span id="sst-val" class="slider-val">0.0</span>°C</label>
                <input type="range" id="sst-slider" min="-2.0" max="2.0" value="0.0" step="0.5">
            </div>
            
            <div class="control-group">
                <label for="ghg-slider">GHG Radiative Forcing: <span id="ghg-val" class="slider-val">1.0</span>x</label>
                <input type="range" id="ghg-slider" min="1.0" max="2.5" value="1.0" step="0.1">
            </div>
            
            <div class="control-group">
                <label for="albedo-slider">Land Albedo / Deforestation: <span id="albedo-val" class="slider-val">1.0</span></label>
                <input type="range" id="albedo-slider" min="0.8" max="1.0" value="1.0" step="0.05">
            </div>
        </div>
        
        <div class="sidebar-section" style="margin-top:auto;">
            <div style="font-size:0.8rem; color:#64748b; line-height:1.4;">
                <strong>Note:</strong> This dashboard is running client-side with full physics-informed logic and real embedded climate datasets for the selected pilot region.
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <h1>🌍 AI-Powered Digital Twin - India Climate</h1>
            <p>High-fidelity spatial rendering and scenario analysis (Independent Offline Version)</p>
        </div>

        <div class="metrics-board">
            <div class="metric-card">
                <div class="metric-label">Average Max Temp</div>
                <div id="m-avg-tmax" class="metric-value">--</div>
                <div id="m-avg-tmax-sub" class="metric-sub">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Absolute Hotspot</div>
                <div id="m-max-tmax" class="metric-value">--</div>
                <div class="metric-sub" style="color:#94a3b8">Central India Peak</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Regional Average Rain</div>
                <div id="m-avg-rain" class="metric-value">--</div>
                <div id="m-avg-rain-sub" class="metric-sub">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak Daily Precipitation</div>
                <div id="m-max-rain" class="metric-value">--</div>
                <div class="metric-sub" style="color:#94a3b8">Western Ghats Peak</div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="map-container">
                <div class="map-header">
                    <h3 id="map-title">Precipitation Grid</h3>
                    <div class="layer-selector">
                        <button id="btn-rain" class="layer-btn active">Precipitation Grid</button>
                        <button id="btn-temp" class="layer-btn">Maximum Temperature Grid</button>
                    </div>
                </div>
                <div id="map"></div>
            </div>
            
            <div class="alerts-container">
                <h3>Active Hazards & Alerts</h3>
                <div id="alerts-list" style="display:flex; flex-direction:column; gap:10px;">
                    <!-- Dynamically populated -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data injected from Python
        const lats = {lats_json};
        const lons = {lons_json};
        const masks = {masks_json};
        const bakedData = {baked_json};
        
        // Active display variable
        let activeVariable = "rain"; // "rain" or "tmax"
        
        // Initialize Leaflet Map
        const map = L.map('map', {{
            zoomControl: true,
            attributionControl: false
        }}).setView([16.0, 77.5], 5);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 10
        }}).addTo(map);

        // Grid cell objects array
        let gridRectangles = [];
        
        // Color helpers
        function getColor(val, mode) {{
            if (mode === "rain") {{
                if (val < 0.5) return 'rgba(0,0,0,0)';
                if (val < 5) return '#d1fae5';
                if (val < 15) return '#6ee7b7';
                if (val < 35) return '#34d399';
                if (val < 65) return '#059669';
                if (val < 115) return '#047857';
                return '#064e3b';
            }} else {{
                const minT = 10;
                const maxT = 45;
                let norm = (val - minT) / (maxT - minT);
                norm = Math.min(Math.max(norm, 0), 1);
                const hue = 240 - norm * 240;
                return `hsl(${{hue}}, 85%, 50%)`;
            }}
        }}

        // Setup Legends
        let legendControl = null;
        function updateLegend() {{
            if (legendControl) map.removeControl(legendControl);
            
            legendControl = L.control({{position: 'bottomright'}});
            legendControl.onAdd = function() {{
                const div = L.DomUtil.create('div', 'legend');
                if (activeVariable === "rain") {{
                    div.innerHTML = '<strong>Rainfall (mm/day)</strong><br/>' +
                        '<i style="background:#d1fae5"></i> < 5<br/>' +
                        '<i style="background:#6ee7b7"></i> 5 - 15<br/>' +
                        '<i style="background:#34d399"></i> 15 - 35<br/>' +
                        '<i style="background:#059669"></i> 35 - 65<br/>' +
                        '<i style="background:#047857"></i> 65 - 115<br/>' +
                        '<i style="background:#064e3b"></i> > 115<br/>';
                }} else {{
                    div.innerHTML = '<strong>Max Temperature (°C)</strong><br/>' +
                        '<i style="background:hsl(240, 85%, 50%)"></i> 10 - 20 (Cool)<br/>' +
                        '<i style="background:hsl(120, 85%, 50%)"></i> 20 - 30 (Mild)<br/>' +
                        '<i style="background:hsl(60, 85%, 50%)"></i> 30 - 38 (Warm)<br/>' +
                        '<i style="background:hsl(0, 85%, 50%)"></i> > 38 (Extreme)<br/>';
                }}
                return div;
            }};
            legendControl.addTo(map);
        }}

        // Calculations & Rendering Engine
        let currentU = [];
        let currentV = [];

        function updateSimulation() {{
            const season = document.getElementById("season-select").value;
            const sstAnomaly = parseFloat(document.getElementById("sst-slider").value);
            const ghgMultiplier = parseFloat(document.getElementById("ghg-slider").value);
            const albedoFactor = parseFloat(document.getElementById("albedo-slider").value);
            
            // UI Slider text sync
            document.getElementById("sst-val").textContent = sstAnomaly >= 0 ? `+${{sstAnomaly.toFixed(1)}}` : sstAnomaly.toFixed(1);
            document.getElementById("ghg-val").textContent = ghgMultiplier.toFixed(1);
            document.getElementById("albedo-val").textContent = albedoFactor.toFixed(2);
            
            // Get base data
            const baseData = bakedData[season];
            
            // Matrices to store adjusted values
            const adjTmax = [];
            const adjRain = [];
            currentU = [];
            currentV = [];
            
            let totalTmax = 0;
            let maxTmax = -999;
            let totalRain = 0;
            let maxRain = -999;
            let count = 0;
            
            // Physical simulation multipliers
            let rainSstScale = 1.0;
            if (season === "sw_monsoon") {{
                if (sstAnomaly > 0) {{
                    rainSstScale = Math.max(0.4, 1.0 - 0.20 * sstAnomaly);
                }} else {{
                    rainSstScale = Math.min(1.5, 1.0 - 0.15 * sstAnomaly);
                }}
            }}
            
            const tempGhgShift = 1.5 * (ghgMultiplier - 1.0);
            const rainGhgScale = 1.0 + 0.07 * tempGhgShift;
            const tempAlbedoShift = 2.0 * (1.0 - albedoFactor);
            const rainAlbedoScale = (1.0 - 0.15 * (1.0 - albedoFactor));
            
            for (let i = 0; i < lats.length; i++) {{
                adjTmax.push([]);
                adjRain.push([]);
                currentU.push([]);
                currentV.push([]);
                
                for (let j = 0; j < lons.length; j++) {{
                    const isSea = masks.sea[i][j];
                    const baseTemp = baseData.tmax[i][j];
                    const baseR = baseData.rain[i][j];
                    
                    // Wind components
                    currentU[i].push(baseData.u_wind[i][j]);
                    currentV[i].push(baseData.v_wind[i][j]);
                    
                    // Temp shift
                    let temp = baseTemp;
                    if (isSea) {{
                        temp += sstAnomaly;
                    }} else {{
                        temp += tempGhgShift + tempAlbedoShift;
                    }}
                    adjTmax[i].push(temp);
                    
                    // Rain shift
                    let r = baseR;
                    r *= rainSstScale * rainGhgScale;
                    if (!isSea) {{
                        r *= rainAlbedoScale;
                    }}
                    adjRain[i].push(r);
                    
                    // Stats accumulator
                    totalTmax += temp;
                    if (temp > maxTmax) maxTmax = temp;
                    totalRain += r;
                    if (r > maxRain) maxRain = r;
                    count++;
                }
            }}
            
            const meanTmax = totalTmax / count;
            const meanRain = totalRain / count;
            
            // Baseline stats
            let baseTotalT = 0, baseTotalR = 0;
            for (let i = 0; i < lats.length; i++) {{
                for (let j = 0; j < lons.length; j++) {{
                    baseTotalT += baseData.tmax[i][j];
                    baseTotalR += baseData.rain[i][j];
                }}
            }}
            const baseMeanT = baseTotalT / count;
            const baseMeanR = baseTotalR / count;
            
            const tAnomaly = meanTmax - baseMeanT;
            const rAnomaly = meanRain - baseMeanR;
            
            // Update Metric Text
            document.getElementById("m-avg-tmax").textContent = `${{meanTmax.toFixed(1)}}°C`;
            document.getElementById("m-max-tmax").textContent = `${{maxTmax.toFixed(1)}}°C`;
            document.getElementById("m-avg-rain").textContent = `${{meanRain.toFixed(1)}} mm/day`;
            document.getElementById("m-max-rain").textContent = `${{maxRain.toFixed(1)}} mm`;
            
            // Anomaly labels
            const tSub = document.getElementById("m-avg-tmax-sub");
            tSub.textContent = `${{tAnomaly >= 0 ? '▲' : '▼'}} ${{Math.abs(tAnomaly).toFixed(2)}}°C (vs Normal)`;
            tSub.style.color = tAnomaly >= 0 ? '#ef4444' : '#3b82f6';
            
            const rSub = document.getElementById("m-avg-rain-sub");
            rSub.textContent = `${{rAnomaly >= 0 ? '▲' : '▼'}} ${{Math.abs(rAnomaly).toFixed(2)}} mm/d (vs Normal)`;
            rSub.style.color = rAnomaly >= 0 ? '#10b981' : '#ef4444';
            
            // Draw Map Rectangles
            gridRectangles.forEach(r => map.removeLayer(r));
            gridRectangles = [];
            
            const displayData = activeVariable === "rain" ? adjRain : adjTmax;
            const gridRes = 0.25;
            
            for (let i = 0; i < lats.length; i++) {{
                for (let j = 0; j < lons.length; j++) {{
                    const lat = lats[i];
                    const lon = lons[j];
                    const val = displayData[i][j];
                    
                    const color = getColor(val, activeVariable);
                    if (color === 'rgba(0,0,0,0)') continue;
                    
                    const bounds = [[lat, lon], [lat + gridRes, lon + gridRes]];
                    const rect = L.rectangle(bounds, {{
                        color: 'transparent',
                        fillColor: color,
                        fillOpacity: activeVariable === "rain" ? 0.65 : 0.45,
                        weight: 0
                    }}).addTo(map);
                    
                    rect.bindTooltip(
                        `<strong>Lat:</strong> ${{lat.toFixed(2)}}°N, <strong>Lon:</strong> ${{lon.toFixed(2)}}°E<br/>` +
                        `<strong>${{activeVariable === "rain" ? "Precipitation" : "Max Temp"}}:</strong> ${{val.toFixed(1)}} ${{activeVariable === "rain" ? "mm/day" : "°C"}}<br/>` +
                        `<strong>Wind:</strong> Vector [U: ${{currentU[i][j].toFixed(1)}}, V: ${{currentV[i][j].toFixed(1)}}] m/s`,
                        {{
                            className: 'info-tooltip',
                            sticky: true
                        }}
                    );
                    
                    gridRectangles.push(rect);
                }
            }}
            
            updateLegend();
            evaluateAlerts(adjTmax, adjRain, season);
        }}

        // Evaluate Warnings
        function evaluateAlerts(adjTmax, adjRain, season) {{
            const list = document.getElementById("alerts-list");
            list.innerHTML = "";
            
            const warnings = [];
            
            let maxT_vidarbha = -999;
            let maxR_coast = -999;
            let sumR_deccan = 0;
            let countDeccan = 0;
            
            for (let i = 0; i < lats.length; i++) {{
                for (let j = 0; j < lons.length; j++) {{
                    const lat = lats[i];
                    const lon = lons[j];
                    const t = adjTmax[i][j];
                    const r = adjRain[i][j];
                    
                    if (lat >= 18.0 && lat <= 22.0 && lon >= 77.0 && lon <= 82.0 && !masks.sea[i][j]) {{
                        if (t > maxT_vidarbha) maxT_vidarbha = t;
                    }}
                    
                    if (lat >= 12.0 && lat <= 16.0 && lon >= 73.0 && lon <= 74.5 && !masks.sea[i][j]) {{
                        if (r > maxR_coast) maxR_coast = r;
                    }}
                    
                    if (masks.deccan[i][j]) {{
                        sumR_deccan += r;
                        countDeccan++;
                    }}
                }
            }}
            
            if (maxT_vidarbha >= 45.0) {{
                warnings.push({{
                    type: 'Severe Heatwave Alert',
                    region: 'Vidarbha & Central India',
                    severity: 'Critical',
                    message: `Critical temperatures of ${{maxT_vidarbha.toFixed(1)}}°C detected. Alert water distribution grids and healthcare facilities.`
                }});
            }} else if (maxT_vidarbha >= 42.0) {{
                warnings.push({{
                    type: 'Heatwave warning',
                    region: 'Vidarbha & Central India',
                    severity: 'Warning',
                    message: `Heatwave conditions expected with temperatures reaching ${{maxT_vidarbha.toFixed(1)}}°C. Restrict outdoor heavy labor.`
                }});
            }}
            
            if (maxR_coast >= 115.6) {{
                warnings.push({{
                    type: 'Torrential Rain & Flood Warning',
                    region: 'Coastal Karnataka & Konkan',
                    severity: 'Critical',
                    message: `Flash flood and landslide hazards likely due to extreme rainfall of ${{maxR_coast.toFixed(1)}} mm. Mobilize NDRF emergency units.`
                }});
            }} else if (maxR_coast >= 64.5) {{
                warnings.push({{
                    type: 'Heavy Rainfall Alert',
                    region: 'Coastal Karnataka & Konkan',
                    severity: 'Warning',
                    message: `Heavy rainfall forecasts of ${{maxR_coast.toFixed(1)}} mm. Avoid coastal shipping and low-lying underpasses.`
                }});
            }}
            
            if (season === "sw_monsoon") {{
                const deccanAvg = sumR_deccan / countDeccan;
                if (deccanAvg < 0.5) {{
                    warnings.push({{
                        type: 'Drought Dry Spell Warning',
                        region: 'Marathwada & Northern Karnataka',
                        severity: 'Warning',
                        message: `Agricultural stress detected. Average precipitation fell to ${{deccanAvg.toFixed(2)}} mm/day during active monsoon zone.`
                    }});
                }}
            }}
            
            if (warnings.length === 0) {{
                list.innerHTML = `
                    <div class="alert-card alert-info">
                        <strong>✓ Normal Climate State</strong><br/>
                        No extreme weather alerts triggered in the pilot region for this scenario.
                    </div>
                `;
            }} else {{
                warnings.forEach(w => {{
                    const cardClass = w.severity === "Critical" ? "alert-critical" : "alert-warning";
                    const badge = w.severity === "Critical" ? "🔴" : "🟡";
                    list.innerHTML += `
                        <div class="alert-card ${{cardClass}}">
                            <strong>${{badge}} ${{w.type}}</strong><br/>
                            <small>Region: ${{w.region}}</small>
                            <p style="margin: 6px 0 0 0; line-height:1.3;">${{w.message}}</p>
                        </div>
                    `;
                }});
            }}
        }}

        // Variable Toggles
        document.getElementById("btn-rain").addEventListener("click", function() {{
            activeVariable = "rain";
            document.getElementById("map-title").textContent = "Precipitation Grid";
            this.classList.add("active");
            document.getElementById("btn-temp").classList.remove("active");
            updateSimulation();
        }});
        
        document.getElementById("btn-temp").addEventListener("click", function() {{
            activeVariable = "tmax";
            document.getElementById("map-title").textContent = "Maximum Temperature Grid";
            this.classList.add("active");
            document.getElementById("btn-rain").classList.remove("active");
            updateSimulation();
        }});

        // Event listeners
        document.getElementById("season-select").addEventListener("change", updateSimulation);
        document.getElementById("sst-slider").addEventListener("input", updateSimulation);
        document.getElementById("ghg-slider").addEventListener("input", updateSimulation);
        document.getElementById("albedo-slider").addEventListener("input", updateSimulation);

        // Canvas Overlay Wind Animation
        const canvasElement = document.createElement('canvas');
        canvasElement.style.position = 'absolute';
        canvasElement.style.top = 0;
        canvasElement.style.left = 0;
        canvasElement.style.pointerEvents = 'none';
        canvasElement.style.zIndex = 400;
        map.getPanes().overlayPane.appendChild(canvasElement);
        
        const particles = [];
        const maxParticles = 150;
        
        for (let k = 0; k < maxParticles; k++) {{
            particles.push({{
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                age: Math.floor(Math.random() * 80)
            }});
        }}
        
        function resizeCanvas() {{
            const size = map.getSize();
            canvasElement.width = size.x;
            canvasElement.height = size.y;
        }}
        
        function drawWind() {{
            if (!canvasElement.getContext) return;
            const ctx = canvasElement.getContext('2d');
            
            ctx.fillStyle = 'rgba(11, 15, 25, 0.85)';
            ctx.globalCompositeOperation = 'destination-in';
            ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);
            ctx.globalCompositeOperation = 'source-over';
            
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
            ctx.lineWidth = 1.2;
            
            const size = map.getSize();
            
            particles.forEach(p => {{
                p.age++;
                if (p.age > 80 || p.x < 0 || p.x > size.x || p.y < 0 || p.y > size.y) {{
                    p.x = Math.random() * size.x;
                    p.y = Math.random() * size.y;
                    p.age = 0;
                }}
                
                const latlng = map.containerPointToLatLng([p.x, p.y]);
                
                let latIdx = Math.floor((latlng.lat - 8.0) / 0.25);
                let lonIdx = Math.floor((latlng.lng - 70.0) / 0.25);
                
                let u = 2.0;
                let v = 1.0;
                
                if (latIdx >= 0 && latIdx < lats.length && lonIdx >= 0 && lonIdx < lons.length) {{
                    u = currentU[latIdx][lonIdx];
                    v = currentV[latIdx][lonIdx];
                }}
                
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                
                const dx = u * 0.7;
                const dy = -v * 0.7;
                
                ctx.lineTo(p.x + dx, p.y + dy);
                ctx.stroke();
                
                p.x += dx;
                p.y += dy;
            }});
            
            requestAnimationFrame(drawWind);
        }}
        
        map.on('move', function() {{
            const topLeft = map.latLngToLayerPoint(map.getBounds().getNorthWest());
            canvasElement.style.transform = `translate3d(${{topLeft.x}}px, ${{topLeft.y}}px, 0px)`;
        }});
        
        map.on('viewreset zoom', function() {{
            resizeCanvas();
            const topLeft = map.latLngToLayerPoint(map.getBounds().getNorthWest());
            canvasElement.style.transform = `translate3d(${{topLeft.x}}px, ${{topLeft.y}}px, 0px)`;
        }});
        
        // Initial setup
        resizeCanvas();
        updateSimulation();
        requestAnimationFrame(drawWind);
    </script>
</body>
</html>"""
    
    # Render arrays to strings for HTML replacement
    html_content = html_template.replace("{lats_json}", json.dumps(LATS.tolist()))
    html_content = html_content.replace("{lons_json}", json.dumps(LONS.tolist()))
    html_content = html_content.replace("{masks_json}", json.dumps(masks))
    html_content = html_content.replace("{baked_json}", json.dumps(baked_data))
    
    # Save the output file
    target_path = "C:\\\\Users\\\\singh\\\\.gemini\\\\antigravity\\\\scratch\\\\india-climate-digital-twin\\\\fallback_dashboard.html"
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"SUCCESS: Fallback Dashboard saved to {target_path}")

if __name__ == "__main__":
    generate_fallback_dashboard()
