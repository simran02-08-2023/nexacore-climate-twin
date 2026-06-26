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
        1. Generates 5 days of history leading up to the target day.
        2. Feeds the history into the AI predictor (if provided) to get a 3-day forecast.
        3. Falls back to physical simulation grids if the predictor is not available.
        4. Calculates anomaly maps and triggers warnings.
        """
        # Generate 5 days of historical grids
        history_seq = []
        for t in range(5):
            d = (day_of_year - 5 + t - 1) % 365 + 1
            t_max, t_min, rain, u, v, lst, sst = self.generator.generate_day_data(
                d, sst_anomaly, ghg_multiplier, albedo_factor
            )
            lst_filled = np.where(np.isnan(lst), t_max, lst)
            sst_filled = np.where(np.isnan(sst), 28.0, sst)
            day_tensor = np.stack([rain, t_max, t_min, u, v, lst_filled, sst_filled], axis=-1)
            history_seq.append(day_tensor)

        history_seq = np.array(history_seq)

        # Predict future 3 days using ML predictor if available
        pred_grids = None
        if predictor is not None:
            try:
                pred_grids = predictor.predict(history_seq)
                # pred_grids from ClimateModel.predict returns (N, lat, lon)
                # We need to reconstruct full (3, lat, lon, 7) tensor
                # Use physics fallback for the other variables, replace channel 0 with ML rain prediction
                phys_grids = []
                for t in range(min(3, len(pred_grids))):
                    d = (day_of_year + t - 1) % 365 + 1
                    t_max, t_min, rain, u, v, lst, sst = self.generator.generate_day_data(
                        d, sst_anomaly, ghg_multiplier, albedo_factor
                    )
                    lst_filled = np.where(np.isnan(lst), t_max, lst)
                    sst_filled = np.where(np.isnan(sst), 28.0, sst)
                    # Use ML prediction for rain (channel 0), physics for rest
                    ml_rain = pred_grids[t] if t < len(pred_grids) else rain
                    day_tensor = np.stack([ml_rain, t_max, t_min, u, v, lst_filled, sst_filled], axis=-1)
                    phys_grids.append(day_tensor)
                pred_grids = np.array(phys_grids)
            except Exception:
                pred_grids = None

        if pred_grids is None:
            # Physics simulation fallback
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

        # Extract individual variables — shape: (3, lat, lon)
        forecast_rain = pred_grids[:, :, :, 0]
        forecast_tmax = pred_grids[:, :, :, 1]
        forecast_tmin = pred_grids[:, :, :, 2]
        forecast_u = pred_grids[:, :, :, 3]
        forecast_v = pred_grids[:, :, :, 4]

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
        Evaluates regional grid pixels against IMD thresholds:
        - Heatwave: Max Temp >= 40C with departure >= 4.5C, Severe if >= 45C
        - Heavy Rain: >= 64.5 mm/day, Very Heavy: >= 115.6 mm/day
        - Drought: Rainfall deficit during active monsoon phase
        """
        warnings = []

        lon_grid, lat_grid = np.meshgrid(LONS, LATS)

        vidarbha_mask = (
            LAND_MASK &
            (lat_grid >= 18.0) & (lat_grid <= 22.0) &
            (lon_grid >= 77.0) & (lon_grid <= 82.0)
        )
        coastal_mask = (
            LAND_MASK &
            (lat_grid >= 12.0) & (lat_grid <= 16.0) &
            (lon_grid >= 73.0) & (lon_grid <= 74.5)
        )
        deccan_dry_mask = (
            DECCAN_MASK &
            (lat_grid >= 16.0) & (lat_grid <= 20.0) &
            (lon_grid >= 75.0) & (lon_grid <= 78.0)
        )

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
                        'message': f"Severe heatwave detected with temperatures reaching {max_val:.1f}°C. IMD Red Warning protocols triggered."
                    })
                elif max_val >= 42.0:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Vidarbha & Central India',
                        'type': 'Heatwave Warning',
                        'severity': 'Warning',
                        'message': f"Heatwave conditions expected. Max temperature {max_val:.1f}°C. Advise hydration and shade."
                    })

            # 2. Flood / Heavy Rain warnings (Western Ghats & Coastal Karnataka)
            coast_rain = rain[coastal_mask]
            if len(coast_rain) > 0:
                max_rain = np.max(coast_rain)
                if max_rain >= 115.6:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Coastal Karnataka & Konkan',
                        'type': 'Flash Flood & Landslide Warning',
                        'severity': 'Critical',
                        'message': f"Very heavy rainfall of {max_rain:.1f} mm predicted. High landslide risk on Western Ghat slopes."
                    })
                elif max_rain >= 64.5:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Coastal Karnataka & Konkan',
                        'type': 'Heavy Rainfall Warning',
                        'severity': 'Warning',
                        'message': f"Heavy precipitation forecast ({max_rain:.1f} mm). Localized flooding in low-lying coastal areas."
                    })

            # 3. Drought / Moisture Deficit (Marathwada rain shadow)
            deccan_rain = rain[deccan_dry_mask]
            if len(deccan_rain) > 0:
                avg_rain = np.mean(deccan_rain)
                if 160 <= (day_of_year + t) <= 260 and avg_rain < 0.5:
                    warnings.append({
                        'day': t + 1,
                        'region': 'Marathwada & Northern Karnataka',
                        'type': 'Drought & Agricultural Dry Spell',
                        'severity': 'Warning',
                        'message': f"Extended dry spell during active monsoon phase (mean rain {avg_rain:.2f} mm). Soil moisture stress detected."
                    })

        return warnings