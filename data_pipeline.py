import numpy as np
import pandas as pd
import os

# Pilot Region Configuration
LAT_MIN, LAT_MAX = 8.0, 24.0      # 16 degrees span
LON_MIN, LON_MAX = 70.0, 85.0     # 15 degrees span
GRID_RES = 0.25                   # 0.25 degree resolution
GRID_LAT_SHAPE = int((LAT_MAX - LAT_MIN) / GRID_RES)  # 64
GRID_LON_SHAPE = int((LON_MAX - LON_MIN) / GRID_RES)  # 60

# Coordinate arrays
LATS = np.linspace(LAT_MIN, LAT_MAX - GRID_RES, GRID_LAT_SHAPE)
LONS = np.linspace(LON_MIN, LON_MAX - GRID_RES, GRID_LON_SHAPE)

# Topography mask (simulating Western Ghats, Deccan Plateau, Sea)
# Western Ghats: Longitude around 73.5E, Latitude 8N to 18N
# Sea mask: Longitude < 72E or (Latitude < 15N and Longitude < 73E) or (Latitude < 10N and Longitude > 80E)
def get_masks():
    lon_grid, lat_grid = np.meshgrid(LONS, LATS)
    
    # Sea Mask: True for ocean grid points
    sea_mask = (lon_grid < 71.5) | \
               ((lat_grid < 14.0) & (lon_grid < 72.5)) | \
               ((lat_grid < 9.0) & (lon_grid > 79.0))
               
    # Western Ghats Mask (elevated mountain ridge along the west coast)
    wg_mask = (~sea_mask) & (lon_grid >= 73.0) & (lon_grid <= 74.2) & (lat_grid >= 8.5) & (lat_grid <= 18.0)
    
    # Deccan Plateau (rain-shadow region east of the Ghats)
    deccan_mask = (~sea_mask) & (~wg_mask) & (lon_grid > 74.2) & (lat_grid < 20.0)
    
    # Land Mask
    land_mask = ~sea_mask
    
    return land_mask, sea_mask, wg_mask, deccan_mask

LAND_MASK, SEA_MASK, WG_MASK, DECCAN_MASK = get_masks()

class ClimateDataGenerator:
    """
    Generates climatologically-accurate synthetic datasets for India's climate.
    Includes seasonal monsoonal shifts, spatial temperature gradients, 
    elevation/topography controls (Western Ghats orographic rain, Deccan rain shadow),
    SST/LST distinctions, and U/V wind vectors.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        
    def get_climatology(self, day_of_year):
        """
        Returns base climatology maps (no noise) for a specific day of the year (1 to 365).
        """
        # Define seasonal phases
        # SW Monsoon: June (152) to Sept (273). Peak in July/Aug (182-243)
        # NE Monsoon: Oct (274) to Dec (365)
        # Summer (Pre-monsoon): March (60) to May (151)
        # Winter: Jan (1) to Feb (59)
        
        d = day_of_year
        lon_grid, lat_grid = np.meshgrid(LONS, LATS)
        
        # 1. Temperature baseline: Latitude gradient (colder north, warmer south in winter)
        # In summer, central land heats up intensely.
        if 60 <= d < 152: # Summer
            base_temp = 36.0 + 3.0 * np.sin(np.pi * (lat_grid - 16) / 24) # Peak heat in central
            # Add desert heat towards northwest
            base_temp += 4.0 * (lon_grid - 70) / 15 * (lat_grid - 8) / 16
        elif 152 <= d < 274: # SW Monsoon
            base_temp = 28.0 + 2.0 * np.cos(np.pi * (lat_grid - 12) / 16) # Cool down due to rain
        elif 274 <= d < 335: # NE Monsoon / Autumn
            base_temp = 27.0 - 3.0 * (lat_grid - 16) / 16
        else: # Winter
            base_temp = 22.0 - 8.0 * (lat_grid - 16) / 16 # Cold north
            
        # Adjust temperature for elevation (Western Ghats are cooler)
        base_temp[WG_MASK] -= 4.5
        # Ocean (SST) has lower seasonal variation
        sst_base = 27.5 + 2.0 * np.sin(2 * np.pi * (d - 120) / 365)
        base_temp[SEA_MASK] = sst_base[SEA_MASK] if isinstance(sst_base, np.ndarray) else sst_base
        
        # Max and Min Temp separation
        t_max = base_temp + 4.5
        t_min = base_temp - 4.5
        t_max[SEA_MASK] = base_temp[SEA_MASK] + 1.5
        t_min[SEA_MASK] = base_temp[SEA_MASK] - 1.5
        
        # 2. Rainfall Baseline
        rainfall = np.zeros_like(lat_grid)
        if 152 <= d < 274: # Southwest Monsoon
            # Orographic rain on Western Ghats
            rainfall[WG_MASK] = 40.0 + 25.0 * np.sin(np.pi * (d - 152) / 122)
            # Coastal rain
            coastal_mask = (lon_grid < 73.5) & (~SEA_MASK)
            rainfall[coastal_mask] = 30.0 + 15.0 * np.sin(np.pi * (d - 152) / 122)
            # Inland Deccan (rain shadow - dry)
            rainfall[DECCAN_MASK] = 3.0 + 4.0 * np.sin(np.pi * (d - 152) / 122)
            # Other land (central/north)
            other_land = LAND_MASK & (~WG_MASK) & (~DECCAN_MASK) & (lon_grid > 74.0)
            rainfall[other_land] = 12.0 + 10.0 * np.sin(np.pi * (d - 152) / 122)
            # Ocean rain
            rainfall[SEA_MASK] = 8.0 + 6.0 * np.sin(np.pi * (d - 152) / 122)
            
        elif 274 <= d < 335: # Northeast Monsoon (Rain in South East)
            se_mask = LAND_MASK & (lon_grid > 78.0) & (lat_grid < 15.0)
            rainfall[se_mask] = 15.0 + 10.0 * np.sin(np.pi * (d - 274) / 61)
            # Rest of the country is dry
            other = LAND_MASK & (~se_mask)
            rainfall[other] = 0.2
        else: # Winter/Summer (dry except occasional pre-monsoon showers)
            rainfall.fill(0.1)
            if 120 <= d < 152: # Pre-monsoon showers
                rainfall[LAND_MASK] = self.rng.choice([0.1, 2.0, 5.0], size=np.sum(LAND_MASK), p=[0.9, 0.08, 0.02])

        # 3. Wind vectors (U: zonal, V: meridional)
        u_wind = np.zeros_like(lat_grid)
        v_wind = np.zeros_like(lat_grid)
        
        if 152 <= d < 274: # SW Monsoon (South-Westerly flow: positive U, positive V)
            u_wind.fill(8.0)
            v_wind.fill(6.0)
            # Increase wind speed over sea
            u_wind[SEA_MASK] += 3.0
            v_wind[SEA_MASK] += 2.0
        elif 274 <= d < 365: # NE Monsoon (North-Easterly flow: negative U, negative V)
            u_wind.fill(-4.0)
            v_wind.fill(-3.0)
        else: # Winter/Spring (Light easterly / variable winds)
            u_wind.fill(-2.0)
            v_wind.fill(1.0)
            
        # 4. INSAT Satellite LST and SST
        # LST equals land temperature, SST equals sea temperature
        lst = np.copy(t_max)
        lst[SEA_MASK] = np.nan # LST not defined on water
        
        sst = np.copy(t_max)
        sst[LAND_MASK] = np.nan # SST not defined on land
        
        return t_max, t_min, rainfall, u_wind, v_wind, lst, sst

    def generate_day_data(self, day_of_year, sst_anomaly=0.0, ghg_multiplier=1.0, albedo_factor=1.0):
        """
        Generates daily grids with spatial weather noise and applies physical what-if scaling parameters.
        - sst_anomaly: Ocean warming parameter (+/- Celsius)
        - ghg_multiplier: GHG radiative forcing scale (1.0 = baseline, >1.0 = warmer & more extreme rain)
        - albedo_factor: Urbanization/Deforestation (1.0 = base, <1.0 = lower albedo/drier/warmer land)
        """
        t_max, t_min, rain, u_wind, v_wind, lst, sst = self.get_climatology(day_of_year)
        
        # Apply What-if adjustments
        
        # A. Ocean warming (SST anomaly) directly adjusts SST and influences SW monsoon rainfall.
        # El Nino (+ SST anomaly) reduces monsoon rainfall by 20% per degree.
        # La Nina (- SST anomaly) increases monsoon rainfall by 15% per degree.
        sst_scale = 1.0
        if 152 <= day_of_year < 274: # SW Monsoon months
            if sst_anomaly > 0:
                sst_scale = max(0.4, 1.0 - 0.20 * sst_anomaly)
            else:
                sst_scale = min(1.5, 1.0 - 0.15 * sst_anomaly)
                
        # SST adjustment
        sst += sst_anomaly
        t_max[SEA_MASK] += sst_anomaly
        t_min[SEA_MASK] += sst_anomaly
        
        # B. GHG forcing increases air temp (LST/Max/Min Temp) and intensifies rainfall variance (Clausius-Clapeyron).
        temp_shift = 1.5 * (ghg_multiplier - 1.0)
        t_max[LAND_MASK] += temp_shift
        t_min[LAND_MASK] += temp_shift
        if lst is not None:
            lst[LAND_MASK] += temp_shift
            
        # Rainfall intensification (more heavy storms, or longer dry spells)
        rain_scale = 1.0 + 0.07 * temp_shift  # 7% increase in water holding capacity per degree C
        rain *= sst_scale * rain_scale
        
        # C. Albedo/Deforestation reduces land albedo, increasing sensible heat (warmer land, less transpiration/rain).
        albedo_temp_shift = 2.0 * (1.0 - albedo_factor)
        t_max[LAND_MASK] += albedo_temp_shift
        t_min[LAND_MASK] += albedo_temp_shift
        if lst is not None:
            lst[LAND_MASK] += albedo_temp_shift
        # Deforestation reduces local convective rainfall
        rain[LAND_MASK] *= (1.0 - 0.15 * (1.0 - albedo_factor))
        
        # Add spatial weather noise (moving fronts/noise)
        # Use simple spatial smoothing to make weather structures realistic
        noise_shape = (GRID_LAT_SHAPE, GRID_LON_SHAPE)
        
        # Temperature noise (correlated spatial fields)
        t_noise = self.smooth_grid(self.rng.normal(0, 1.5, size=noise_shape), sigma=2.0)
        t_max += t_noise
        t_min += t_noise
        if lst is not None:
            lst[LAND_MASK] += t_noise[LAND_MASK]
            
        # Rainfall noise (multiplicative gamma-like distribution to maintain positive values and sparseness)
        r_noise = self.smooth_grid(self.rng.uniform(0.5, 2.0, size=noise_shape), sigma=1.5)
        # Random storm events
        storm_trigger = self.smooth_grid(self.rng.choice([0.0, 5.0], size=noise_shape, p=[0.95, 0.05]), sigma=2.0)
        
        rain = rain * r_noise + storm_trigger
        rain = np.clip(rain, 0, 250.0) # Clip extreme rain
        
        # Light smoothing on winds
        u_wind += self.rng.normal(0, 1.0, size=noise_shape)
        v_wind += self.rng.normal(0, 1.0, size=noise_shape)
        
        return t_max, t_min, rain, u_wind, v_wind, lst, sst

    def smooth_grid(self, grid, sigma=1.5):
        """
        Applies a simple spatial Gaussian filter using numpy convolution (no external scipy needed).
        """
        # Create small Gaussian kernel
        k_size = int(2 * sigma) * 2 + 1
        x = np.arange(-k_size//2 + 1, k_size//2 + 1)
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()
        
        # Manual pad and 2D convolution
        pad = k_size // 2
        padded = np.pad(grid, pad, mode='edge')
        smoothed = np.zeros_like(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                smoothed[i, j] = np.sum(padded[i:i+k_size, j:j+k_size] * kernel_2d)
        return smoothed

    def generate_dataset(self, num_days=90, start_day=152, sst_anomaly=0.0, ghg_multiplier=1.0, albedo_factor=1.0):
        """
        Generates a continuous sequence of daily data tensors.
        Shape: (num_days, GRID_LAT_SHAPE, GRID_LON_SHAPE, 7)
        """
        dataset = []
        for t in range(num_days):
            d = (start_day + t - 1) % 365 + 1
            t_max, t_min, rain, u_wind, v_wind, lst, sst = self.generate_day_data(
                d, sst_anomaly, ghg_multiplier, albedo_factor
            )
            # Impute NaNs in LST and SST for NN tensor input (LST filled with max temp, SST filled with average SST)
            lst_filled = np.where(np.isnan(lst), t_max, lst)
            sst_filled = np.where(np.isnan(sst), 28.0, sst)
            
            day_tensor = np.stack([
                rain,          # ch 0: Rainfall
                t_max,         # ch 1: Max Temp
                t_min,         # ch 2: Min Temp
                u_wind,        # ch 3: U Wind
                v_wind,        # ch 4: V Wind
                lst_filled,    # ch 5: LST
                sst_filled     # ch 6: SST
            ], axis=-1)
            dataset.append(day_tensor)
            
        return np.array(dataset)

# IMD Binary Parser Implementations (provided for compliance with national datasets)
def read_imd_gridded_rainfall(filepath, day_of_year, year):
    """
    Reads IMD binary gridded daily rainfall data (0.25 x 0.25 degree resolution).
    India grid size is 135 rows (latitudes 6.5N to 38.5N) by 129 columns (longitudes 66.5E to 100.0E).
    Each day contains 135 * 129 float32 values (54,810 bytes).
    """
    # Calculate offset
    # IMD files store 365 or 366 days depending on leap year
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_in_file = 366 if is_leap else 365
    
    if day_of_year < 1 or day_of_year > days_in_file:
        raise ValueError(f"Day of year {day_of_year} invalid for year {year}")
        
    num_grid_points = 135 * 129
    bytes_per_day = num_grid_points * 4 # float32 is 4 bytes
    offset = (day_of_year - 1) * bytes_per_day
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IMD Rainfall binary file not found: {filepath}")
        
    with open(filepath, 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=np.float32, count=num_grid_points)
        
    # Reshape and flip latitude so it starts from North to South, or match coordinates
    # IMD grid starts from 6.5N (index 0) to 38.5N (index 134)
    # Longitude starts from 66.5E (index 0) to 100.0E (index 128)
    grid = data.reshape((135, 129))
    
    # Extract our pilot region bounding box (8N-24N, 70E-85E)
    # Mapping coordinates to indices:
    # lat_index = int((lat - 6.5) / 0.25)
    # lon_index = int((lon - 66.5) / 0.25)
    lat_start_idx = int((LAT_MIN - 6.5) / 0.25)
    lat_end_idx = int((LAT_MAX - 6.5) / 0.25)
    lon_start_idx = int((LON_MIN - 66.5) / 0.25)
    lon_end_idx = int((LON_MAX - 66.5) / 0.25)
    
    pilot_grid = grid[lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]
    # Replace negative fill values (IMD uses -99.9 or similar for ocean/no-data)
    pilot_grid = np.where(pilot_grid < 0, 0.0, pilot_grid)
    return pilot_grid

def read_imd_gridded_temperature(filepath, day_of_year, year):
    """
    Reads IMD binary gridded daily Max/Min temperature data (1.0 x 1.0 degree resolution).
    India grid size is 31 rows (latitudes 7.5N to 37.5N) by 31 columns (longitudes 67.5E to 97.5E).
    Each day contains 31 * 31 float32 values (3844 bytes).
    """
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_in_file = 366 if is_leap else 365
    
    num_grid_points = 31 * 31
    bytes_per_day = num_grid_points * 4
    offset = (day_of_year - 1) * bytes_per_day
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IMD Temp binary file not found: {filepath}")
        
    with open(filepath, 'rb') as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=np.float32, count=num_grid_points)
        
    grid = data.reshape((31, 31))
    # Fill missing values (99.9 or similar) with NaN or climatology
    grid = np.where(grid > 99.0, np.nan, grid)
    
    # Extract pilot region (8N-24N, 70E-85E)
    # lat_index = int(lat - 7.5)
    # lon_index = int(lon - 67.5)
    lat_start_idx = int(LAT_MIN - 7.5)
    lat_end_idx = int(LAT_MAX - 7.5)
    lon_start_idx = int(LON_MIN - 67.5)
    lon_end_idx = int(LON_MAX - 67.5)
    
    pilot_grid_1deg = grid[lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]
    
    # Interpolate from 1.0 degree to 0.25 degree to match the rainfall grid size (64 x 60)
    # Simple bilinear zoom interpolation
    pilot_grid_25deg = np.repeat(np.repeat(pilot_grid_1deg, 4, axis=0), 4, axis=1)
    
    # Handle NaNs by interpolating/filling
    if np.any(np.isnan(pilot_grid_25deg)):
        mean_val = np.nanmean(pilot_grid_25deg)
        pilot_grid_25deg = np.where(np.isnan(pilot_grid_25deg), mean_val if not np.isnan(mean_val) else 30.0, pilot_grid_25deg)
        
    return pilot_grid_25deg

if __name__ == "__main__":
    # Test execution
    print("Testing Climatology Generator...")
    gen = ClimateDataGenerator()
    dataset = gen.generate_dataset(num_days=10)
    print("Generated dataset shape:", dataset.shape)
    assert dataset.shape == (10, GRID_LAT_SHAPE, GRID_LON_SHAPE, 7)
    print("SUCCESS: Spatio-temporal climate dataset generator verified.")
