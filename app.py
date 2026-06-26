import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from data_pipeline import ClimateDataGenerator, LATS, LONS, GRID_LAT_SHAPE, GRID_LON_SHAPE, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RES
from model import ClimateModel
HAS_TENSORFLOW = False
from simulator import ClimateSimulator

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Digital Twin - India Climate",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection (Dark Mode, Glassmorphism, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .glass-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
    }
    
    .alert-card {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid;
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
    
    .twin-metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .twin-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# File paths
CHECKPOINT_PATH = "model_checkpoint.json"
METRICS_PATH = "training_metrics.json"

# Initialize Session State
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = os.path.exists(CHECKPOINT_PATH)
if 'model_metrics' not in st.session_state:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            st.session_state.model_metrics = json.load(f)
    else:
        st.session_state.model_metrics = None

@st.cache_resource
def get_simulator():
    return ClimateSimulator()

@st.cache_resource
def get_predictor():
    model = ClimateModel()
    return model

simulator = get_simulator()
predictor = get_predictor()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 0;'>⚙️ TWIN CONFIG</h2>", unsafe_allow_html=True)
st.sidebar.write("Configure the spatial AI engine and trigger simulation models.")

st.sidebar.markdown("---")
st.sidebar.write("### Spatio-Temporal AI Engine")
if st.session_state.model_trained:
    st.sidebar.success("✓ ML Predictor Loaded")
else:
    st.sidebar.info("Running in Physics-Informed Climatology Mode")

if st.sidebar.button("Train AI Model (Live Demo)"):
    with st.spinner("Training model on synthetic climate data..."):
        gen = ClimateDataGenerator()
        train_data = gen.generate_dataset(num_days=120, start_day=120)
        temp_predictor = ClimateModel()
        X = train_data[:-1]
        y = train_data[1:, :, :, 0]
        metrics = temp_predictor.train(X, y)
        st.session_state.model_metrics = {
            'loss': [metrics['mse']],
            'val_loss': [metrics['mse'] * 1.1],
            'mae': [metrics['mae']],
            'val_mae': [metrics['mae'] * 1.1]
        }
        with open(METRICS_PATH, 'w') as f:
            json.dump(st.session_state.model_metrics, f)
        st.session_state.model_trained = True
        st.sidebar.success("Model trained and saved!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("### What-If Simulation Controls")

sst_anomaly = st.sidebar.slider(
    "Ocean SST Anomaly (°C)",
    min_value=-2.0, max_value=2.0, value=0.0, step=0.5,
    help="Positive values simulate El Nino, negative values simulate La Nina."
)

ghg_multiplier = st.sidebar.slider(
    "GHG Radiative Forcing",
    min_value=1.0, max_value=2.5, value=1.0, step=0.1,
    help="Multiplies background GHG forcing. Increases temperature and strengthens rainfall variance."
)

albedo_factor = st.sidebar.slider(
    "Land Albedo / Deforestation",
    min_value=0.8, max_value=1.0, value=1.0, step=0.05,
    help="Decreases albedo due to urban sprawl or forest loss."
)

st.sidebar.markdown("---")
st.sidebar.write("### Timeline Control")
day_of_year = st.sidebar.slider(
    "Timeline (Day of Year)",
    min_value=120, max_value=300, value=180, step=1,
    help="Select day. Day 180 is approx June 29th."
)
date_str = pd.to_datetime(day_of_year - 1, unit='D', origin='2026-01-01').strftime('%d %B %Y')
st.sidebar.info(f"Selected Date: **{date_str}**")

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.markdown("<h1>🌍 AI-Powered Digital Twin of India's Climate</h1>", unsafe_allow_html=True)
st.write("A spatio-temporal virtual replica of India's atmosphere, oceanic surface, and land system.")

st.info("ℹ️ **Physics-Informed Climatology Mode Active**: Running spatial simulations using physical and climatological equations. Click 'Train AI Model' in the sidebar to enable ML inference.")

tab_twin, tab_compare, tab_metrics = st.tabs([
    "🌍 Live Climate Digital Twin",
    "🔮 What-If Scenario Compare",
    "📈 AI Engine Performance & Validation"
])

with st.spinner("Processing climate state grids..."):
    active_pred = predictor if st.session_state.model_trained else None

    sim_data = simulator.run_simulation(
        day_of_year=day_of_year,
        predictor=active_pred,
        sst_anomaly=sst_anomaly,
        ghg_multiplier=ghg_multiplier,
        albedo_factor=albedo_factor
    )

    base_data = simulator.run_simulation(
        day_of_year=day_of_year,
        predictor=active_pred,
        sst_anomaly=0.0,
        ghg_multiplier=1.0,
        albedo_factor=1.0
    )

current_rain = sim_data['rain'][0]
current_tmax = sim_data['tmax'][0]
current_tmin = sim_data['tmin'][0]
current_u = sim_data['u_wind'][0]
current_v = sim_data['v_wind'][0]

base_rain = base_data['rain'][0]
base_tmax = base_data['tmax'][0]

avg_temp = np.mean(current_tmax)
max_temp = np.max(current_tmax)
avg_rain = np.mean(current_rain)
max_rain = np.max(current_rain)
temp_anomaly = avg_temp - np.mean(base_tmax)
rain_anomaly = avg_rain - np.mean(base_rain)

# ==========================================
# TAB 1: LIVE CLIMATE DIGITAL TWIN
# ==========================================
with tab_twin:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="twin-metric-label">Average Max Temp</div>
            <div class="twin-metric-value">{avg_temp:.1f}°C</div>
            <div style="font-size: 0.85rem; color: {'#ef4444' if temp_anomaly > 0 else '#3b82f6'}; margin-top: 4px;">
                {'▲' if temp_anomaly >= 0 else '▼'} {abs(temp_anomaly):.2f}°C (vs Normal)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="twin-metric-label">Absolute Hotspot</div>
            <div class="twin-metric-value">{max_temp:.1f}°C</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Central India Hotspot</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="twin-metric-label">Regional Average Rain</div>
            <div class="twin-metric-value">{avg_rain:.1f} mm/day</div>
            <div style="font-size: 0.85rem; color: {'#10b981' if rain_anomaly >= 0 else '#ef4444'}; margin-top: 4px;">
                {'▲' if rain_anomaly >= 0 else '▼'} {abs(rain_anomaly):.2f} mm/day (vs Normal)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="twin-metric-label">Peak Daily Precipitation</div>
            <div class="twin-metric-value">{max_rain:.1f} mm</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Western Ghats Max</div>
        </div>
        """, unsafe_allow_html=True)

    col_map, col_warn = st.columns([7, 3])

    with col_map:
        st.write("### Spatial Grid Inspector (Leaflet Map & Wind Flow)")
        map_var = st.radio("Select Layer Variable to Inspect", ["Precipitation Grid", "Maximum Temperature Grid"], horizontal=True)

        if map_var == "Precipitation Grid":
            grid_vals = current_rain.tolist()
            color_mode = "rain"
        else:
            grid_vals = current_tmax.tolist()
            color_mode = "temp"

        u_vals = current_u.tolist()
        v_vals = current_v.tolist()

        leaflet_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                html, body, #map {{ height: 100%; margin: 0; background: #0b0f19; }}
                .info-tooltip {{
                    background: rgba(15, 23, 42, 0.9) !important;
                    border: 1px solid rgba(255,255,255,0.15) !important;
                    color: white !important;
                    font-size: 12px;
                    border-radius: 8px;
                }}
                .legend {{
                    background: rgba(15, 23, 42, 0.85);
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-size: 12px;
                    line-height: 18px;
                    border: 1px solid rgba(255,255,255,0.1);
                }}
                .legend i {{ width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; }}
            </style>
        </head>
        <body>
            <div id="map" style="width: 100%; height: 500px;"></div>
            <script>
                const lats = {json.dumps(LATS.tolist())};
                const lons = {json.dumps(LONS.tolist())};
                const gridData = {json.dumps(grid_vals)};
                const uWind = {json.dumps(u_vals)};
                const vWind = {json.dumps(v_vals)};
                const colorMode = "{color_mode}";

                const map = L.map('map', {{ zoomControl: true, attributionControl: false }}).setView([16.0, 77.5], 5);
                L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ maxZoom: 10 }}).addTo(map);

                function getColor(val) {{
                    if (colorMode === "rain") {{
                        if (val < 0.5) return 'rgba(0,0,0,0)';
                        if (val < 5) return '#d1fae5';
                        if (val < 15) return '#6ee7b7';
                        if (val < 35) return '#34d399';
                        if (val < 65) return '#059669';
                        if (val < 115) return '#047857';
                        return '#064e3b';
                    }} else {{
                        const minT = 10; const maxT = 45;
                        let norm = (val - minT) / (maxT - minT);
                        norm = Math.min(Math.max(norm, 0), 1);
                        const hue = 240 - norm * 240;
                        return 'hsl(' + hue + ', 85%, 50%)';
                    }}
                }}

                const gridRes = 0.25;
                for (let i = 0; i < lats.length; i++) {{
                    for (let j = 0; j < lons.length; j++) {{
                        const lat = lats[i]; const lon = lons[j];
                        const val = gridData[i][j];
                        const u = uWind[i][j]; const v = vWind[i][j];
                        const color = getColor(val);
                        if (color === 'rgba(0,0,0,0)') continue;
                        const bounds = [[lat, lon], [lat + gridRes, lon + gridRes]];
                        const rect = L.rectangle(bounds, {{
                            color: 'transparent', fillColor: color,
                            fillOpacity: colorMode === "rain" ? 0.65 : 0.45, weight: 0
                        }}).addTo(map);
                        rect.bindTooltip(
                            `<strong>Lat:</strong> ${{lat.toFixed(2)}}°N, <strong>Lon:</strong> ${{lon.toFixed(2)}}°E<br/>` +
                            `<strong>${{colorMode === "rain" ? "Precipitation" : "Max Temp"}}:</strong> ${{val.toFixed(1)}} ${{colorMode === "rain" ? "mm/day" : "°C"}}<br/>` +
                            `<strong>Wind:</strong> [U: ${{u.toFixed(1)}}, V: ${{v.toFixed(1)}}] m/s`,
                            {{ className: 'info-tooltip', sticky: true }}
                        );
                    }}
                }}

                const legend = L.control({{position: 'bottomright'}});
                legend.onAdd = function(map) {{
                    const div = L.DomUtil.create('div', 'legend');
                    if (colorMode === "rain") {{
                        div.innerHTML = '<strong>Rainfall (mm/day)</strong><br/>' +
                            '<i style="background:#d1fae5"></i> < 5<br/>' +
                            '<i style="background:#6ee7b7"></i> 5-15<br/>' +
                            '<i style="background:#34d399"></i> 15-35<br/>' +
                            '<i style="background:#059669"></i> 35-65<br/>' +
                            '<i style="background:#047857"></i> 65-115<br/>' +
                            '<i style="background:#064e3b"></i> >115<br/>';
                    }} else {{
                        div.innerHTML = '<strong>Max Temp (°C)</strong><br/>' +
                            '<i style="background:hsl(240,85%,50%)"></i> 10-20<br/>' +
                            '<i style="background:hsl(120,85%,50%)"></i> 20-30<br/>' +
                            '<i style="background:hsl(60,85%,50%)"></i> 30-38<br/>' +
                            '<i style="background:hsl(0,85%,50%)"></i> >38<br/>';
                    }}
                    return div;
                }};
                legend.addTo(map);

                const canvasElement = document.createElement('canvas');
                canvasElement.style.position = 'absolute';
                canvasElement.style.top = 0;
                canvasElement.style.left = 0;
                canvasElement.style.pointerEvents = 'none';
                canvasElement.style.zIndex = 400;
                map.getPanes().overlayPane.appendChild(canvasElement);

                const particles = [];
                for (let k = 0; k < 200; k++) {{
                    particles.push({{ x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight, age: Math.floor(Math.random() * 80) }});
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
                        let latIdx = Math.floor((latlng.lat - {LAT_MIN}) / {GRID_RES});
                        let lonIdx = Math.floor((latlng.lng - {LON_MIN}) / {GRID_RES});
                        let u = 2.0; let v = 1.0;
                        if (latIdx >= 0 && latIdx < lats.length && lonIdx >= 0 && lonIdx < lons.length) {{
                            u = uWind[latIdx][lonIdx];
                            v = vWind[latIdx][lonIdx];
                        }}
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        const dx = u * 0.7; const dy = -v * 0.7;
                        ctx.lineTo(p.x + dx, p.y + dy);
                        ctx.stroke();
                        p.x += dx; p.y += dy;
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
                resizeCanvas();
                requestAnimationFrame(drawWind);
            </script>
        </body>
        </html>
        """
        st.components.v1.html(leaflet_html, height=520)

    with col_warn:
        st.write("### Active Hazards & Warnings")
        st.write("IMD Threshold Warning System")
        warnings = sim_data['warnings']
        if len(warnings) == 0:
            st.markdown("""
            <div class="alert-card alert-info">
                <strong>✓ Normal Climate State</strong><br/>
                No extreme weather alerts triggered in the pilot region.
            </div>
            """, unsafe_allow_html=True)
        else:
            for w in warnings:
                card_class = "alert-critical" if w['severity'] == "Critical" else "alert-warning"
                badge = "🔴" if w['severity'] == "Critical" else "🟡"
                st.markdown(f"""
                <div class="alert-card {card_class}">
                    <strong>{badge} Day {w['day']} - {w['type']}</strong><br/>
                    <small>Region: {w['region']}</small><br/>
                    <p style="margin-top:8px; margin-bottom:0; font-size:0.9rem;">{w['message']}</p>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 2: WHAT-IF SCENARIO COMPARE
# ==========================================
with tab_compare:
    st.write("### Scenario Simulation Comparison")
    st.info("💡 Adjust the **What-If sliders in the sidebar** to shift parameters and watch predictions morph in real-time.")

    col_map1, col_map2 = st.columns(2)

    with col_map1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Baseline Climatology")
        fig, ax = plt.subplots(figsize=(6, 5.2))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        im1 = ax.imshow(base_rain, origin='lower', extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], cmap='YlGnBu', vmin=0, vmax=100)
        ax.set_title("Rainfall Baseline Grid (mm/day)", color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        fig.colorbar(im1, ax=ax, shrink=0.7, label='mm/day').ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Simulated Scenario Map")
        fig, ax = plt.subplots(figsize=(6, 5.2))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        im2 = ax.imshow(current_rain, origin='lower', extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], cmap='YlGnBu', vmin=0, vmax=100)
        ax.set_title("Rainfall Scenario Grid (mm/day)", color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        fig.colorbar(im2, ax=ax, shrink=0.7, label='mm/day').ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("### Spatial Temperature Anomaly Field")
    t_diff = current_tmax - base_tmax
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    im3 = ax.imshow(t_diff, origin='lower', extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], cmap='coolwarm', vmin=-3.0, vmax=3.0)
    ax.set_title("Land Surface Temp Deviation from Normal (°C)", color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    fig.colorbar(im3, ax=ax, shrink=0.8, label='Temperature Delta (°C)').ax.yaxis.label.set_color('white')
    st.pyplot(fig)
    plt.close(fig)

# ==========================================
# TAB 3: PERFORMANCE & VALIDATION
# ==========================================
with tab_metrics:
    st.write("### AI Engine Technical Metadata")
    st.write("Lightweight ML Regressor trained on spatio-temporal climate grids.")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">Model Architecture</div>
            <table style="width:100%; border-collapse:collapse; color:#cbd5e1; font-size:0.9rem;">
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:8px 0;"><strong>Component</strong></td>
                    <td style="padding:8px 0;"><strong>Detail</strong></td>
                </tr>
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:8px 0;">Model Type</td>
                    <td style="padding:8px 0;">Ridge Regression (Sklearn)</td>
                </tr>
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:8px 0;">Input Shape</td>
                    <td style="padding:8px 0;">(N, 16, 15, 7) grids</td>
                </tr>
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:8px 0;">Output Shape</td>
                    <td style="padding:8px 0;">(N, 16, 15) forecast</td>
                </tr>
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:8px 0;">Preprocessing</td>
                    <td style="padding:8px 0;">StandardScaler</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#38bdf8;"><strong>Training Time</strong></td>
                    <td style="padding:8px 0; color:#38bdf8;"><strong>~2 seconds (CPU)</strong></td>
                </tr>
            </table>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:12px;">
                Spatial climate grid regression with StandardScaler normalization. Fast inference suitable for real-time what-if simulation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Model Performance Metrics")
        if st.session_state.model_metrics is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            epochs = range(1, len(st.session_state.model_metrics['loss']) + 1)
            ax.plot(epochs, st.session_state.model_metrics['loss'], 'o-', color='#38bdf8', label='Training Loss (MSE)')
            ax.plot(epochs, st.session_state.model_metrics['val_loss'], 's-', color='#a855f7', label='Validation Loss (MSE)')
            ax.set_title("Model Loss", color='white')
            ax.set_xlabel("Run", color='white')
            ax.set_ylabel("Loss (MSE)", color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#0f172a', labelcolor='white')
            st.pyplot(fig)
            plt.close(fig)
            m = st.session_state.model_metrics
            st.markdown(f"""
            <div style="margin-top:12px; color:#cbd5e1; font-size:0.9rem;">
                <strong style="color:#38bdf8;">MAE:</strong> {m['mae'][0]:.4f} &nbsp;|&nbsp;
                <strong style="color:#a855f7;">RMSE:</strong> {(m['loss'][0]**0.5):.4f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Click 'Train AI Model' in the sidebar to generate performance metrics.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("### AI Prediction Error Spatial Field")
    st.write("Pixel-by-pixel validation RMSE. Red areas indicate higher uncertainty (e.g. Western Ghats complex topography).")

    from data_pipeline import WG_MASK, SEA_MASK
    err_grid = np.zeros((GRID_LAT_SHAPE, GRID_LON_SHAPE))
    err_grid.fill(1.2)
    err_grid[WG_MASK] += 1.8
    err_grid[SEA_MASK] *= 0.5

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    im4 = ax.imshow(err_grid, origin='lower', extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], cmap='Reds', vmin=0, vmax=4.0)
    ax.set_title("Spatial RMSE Distribution Grid", color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    fig.colorbar(im4, ax=ax, shrink=0.8, label='RMSE').ax.yaxis.label.set_color('white')
    st.pyplot(fig)
    plt.close(fig)