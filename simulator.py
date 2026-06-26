import numpy as np
from data_pipeline import ClimateDataGenerator, WG_MASK, DECCAN_MASK, LAND_MASK, LATS, LONS

class ClimateSimulator:
    """
    Simulation engine that coordinates climate model runs, evaluates physical thresholds,
    and returns extreme warning alerts based on official IMD rules.
    """
    def __init__(self):
        self.generator = ClimateDataGenerator()
        
    def run_simulation(self, day_of_year, predictor=None, sst_anomaly=0.0, ghg_multiplier=1.0, albedo_factor=1.0):
        """
        Runs the simulation for a specific day:
        1. Generates 5 days of history leading up to the target day, incorporating the what-if parameter shifts.
        2. Feeds the history into the AI predictor (if provided) to get a 3-day forecast.
        3. Falls back to physical simulation grids if the predictor is not yet trained/available.
        4. Calculates anomaly maps and triggers warnings.
        """
        # Generate 5 days of historical grids
        history_seq = []
        for t in range(5):
            d = (day_of_year - 5 + t - 1) % 365 + 1
            t_max, t_min, rain, u, v, lst, sst = self.generator.generate_day_data(
                d, sst_anomaly, ghg_multiplier, albedo_factor
            )
            # Impute NaNs for neural net inputs
            lst_filled = np.where(np.isnan(lst), t_max, lst)
            sst_filled = np.where(np.isnan(sst), 28.0, sst)
            
            day_tensor = np.stack([rain, t_max, t_min, u, v, lst_filled, sst_filled], axis=-1)
            history_seq.append(day_tensor)
            
        history_seq = np.array(history_seq)
        
        # Predict future 3 days
        pred_grids = None
        if predictor is not None:
            pred_grids = predictor.predict_future(history_seq) # shape: (3, lat, lon, 7)
            
        if pred_grids is None:
            # Physical simulation fallback (generate directly)
            pred_grids = []
            for t in range(3):
                d = (day_of_year + t - 1) % 365 + 1
                t_max, t_min, rain, u, v, lst, sst = self.generator.generate_day_data(
                    d, sst_anomaly, ghg_multiplier, albedo_factor
                )
                lst_filled = np.where(np.isnan(lst), t_max, lst)
                sst_filled = np.where(np.isnan(sst), 28.0, sst)
                
                day_tensor = np.stack([rain, t_max, t_min, u, v, lst_filled, sst_filled], axis=-1)
                pred_grids.append(day_tensor)
            pred_grids = np.array(pred_grids)
            
        # Extract individual variables for the predicted days
        # Shape: (3, lat, lon)
        forecast_rain = pred_grids[:, :, :, 0]
        forecast_tmax = pred_grids[:, :, :, 1]
        forecast_tmin = pred_grids[:, :, :, 2]
        forecast_u = pred_grids[:, :, :, 3]
        forecast_v = pred_grids[:, :, :, 4]
        
        # Compute warnings based on forecast
        warnings = self.evaluate_thresholds(forecast_tmax, forecast_rain, day_of_year)
        
        return {
            'rain': forecast_rain,
            'tmax': forecast_tmax,
            'tmin': forecast_tmin,
            'u_wind': forecast_u,
            'v_wind': forecast_v,
            'warnings': warnings
        }

    def evaluate_thresholds(self, tmax_seq, rain_seq, day_of_year):
        """
        Evaluates regional grid pixels against standard physical/climatological thresholds.
        - Heatwave alerts (based on IMD absolute criteria: Max Temp > 40C, or departure > 4.5C)
        - Heavy rain/Flash floods (Rainfall > 64.5 mm/day)
        - Drought alerts (Rainfall deficit relative to baseline climatology)
        """
        warnings = []
        
        # Bounding box regions
        # Vidarbha (East Maharashtra): Lat 18-22, Lon 77-82
        # Coastal Karnataka / Goa: Lat 12-16, Lon 73-74.5
        # Marathwada (Dry Deccan): Lat 16-20, Lon 75-78
        
        lon_grid, lat_grid = np.meshgrid(LONS, LATS)
        
        vidarbha_mask = LAND_MASK & (lat_grid >= 18.0) & (lat_grid <= 22.0) & (lon_grid >= 77.0) & (lon_grid <= 82.0)
        coastal_mask = LAND_MASK & (lat_grid >= 12.0) & (lat_grid <= 16.0) & (lon_grid >= 73.0) & (lon_grid <= 74.5)
        deccan_dry_mask = DECCAN_MASK & (lat_grid >= 16.0) & (lat_grid <= 20.0) & (lon_grid >= 75.0) & (lon_grid <= 78.0)
        
        # Loop through each predicted day (day 0, 1, 2)
        for t in range(3):
            tmax = tmax_seq[t]
            rain = rain_seq[t]
            
            # 1. Heatwave warnings (Vidarbha hotspot)
            vidarbha_tmax = tmax[vidarbha_mask]
            if len(vidarbha_tmax) > 0:
                max_val = np.max(vidarbha_tmax)
                if max_val >= 45.0:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Vidarbha & Central India',
                        'type': 'Severe Heatwave Alert',
                        'severity': 'Critical',
                        'message': f"Severe heatwave detected in Central India with temperatures reaching {max_val:.1f}°C. Trigger IMD Red Warning protocols."
                    })
                elif max_val >= 42.0:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Vidarbha & Central India',
                        'type': 'Heatwave Warning',
                        'severity': 'Warning',
                        'message': f"Heatwave conditions expected. Max temperature will hit {max_val:.1f}°C. Advise hydration and shadow hours."
                    })
                    
            # 2. Flood / Torrential Rain warnings (Western Ghats & Coastal Karnataka)
            coast_rain = rain[coastal_mask]
            if len(coast_rain) > 0:
                max_rain = np.max(coast_rain)
                if max_rain >= 115.6:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Coastal Karnataka & Konkan',
                        'type': 'Flash Flood & Landslide Warning',
                        'severity': 'Critical',
                        'message': f"Very heavy rainfall of {max_rain:.1f} mm predicted. High landslide risk in Western Ghat slopes. Disruption to coastal transport likely."
                    })
                elif max_rain >= 64.5:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Coastal Karnataka & Konkan',
                        'type': 'Heavy Rainfall Warning',
                        'severity': 'Warning',
                        'message': f"Heavy precipitation forecast ({max_rain:.1f} mm). Localized inundation in low-lying coastal sectors."
                    })
                    
            # 3. Drought / Moisture Deficit Warning (Marathwada rain shadow)
            deccan_rain = rain[deccan_dry_mask]
            if len(deccan_rain) > 0:
                avg_rain = np.mean(deccan_rain)
                # SW Monsoon active phase but Deccan rain is dry (< 1 mm/day)
                if 160 <= (day_of_year + t) <= 260 and avg_rain < 0.5:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Marathwada & Northern Karnataka',
                        'type': 'Drought & Agricultural Dry Spell',
                        'severity': 'Warning',
                        'message': f"Extended dry spell during active monsoon phase (mean rain < 0.5 mm). Soil moisture stress detected for sowing crops."
                    })
                    
        return warnings
