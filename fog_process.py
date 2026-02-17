import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from herbie import Herbie
from datetime import datetime, timedelta
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
LOOKBACK_HOURS = 6
FORECAST_HOURS = 18
OUTPUT_DIR = "images"

# --- DOMAIN SETTINGS (NC/VA/TN/SC) ---
# Format: [West Longitude, East Longitude, South Latitude, North Latitude]
PLOT_EXTENT = [-85, -75, 32, 38] 

def get_crossover_temp():
    """
    Calculates TXover: Minimum dewpoint observed during warmest daytime hours.
    """
    print("--- Calculating Crossover Temperature (TXover) ---")
    dps = []
    
    # Round current time down to nearest hour
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    for i in range(LOOKBACK_HOURS):
        time_step = now - timedelta(hours=i)
        try:
            # Fetch HRRR Analysis (fxx=0) for Surface Dewpoint
            H = Herbie(time_step, model="hrrr", product="sfc", fxx=0)
            ds = H.xarray(":(DPT):2 m above ground")
            
            # Convert Kelvin to Fahrenheit
            dpt_f = (ds['d2m'] - 273.15) * 9/5 + 32
            dps.append(dpt_f)
            print(f"Loaded Dewpoint for {time_step}")
        except Exception as e:
            print(f"Skipping {time_step} (Data likely unavailable): {e}")

    if not dps:
        print("CRITICAL: No dewpoint data found. Check Herbie/HRRR availability.")
        sys.exit(1)

    # Stack and find minimum Dewpoint per pixel
    concat_da = xr.concat(dps, dim='time')
    txover = concat_da.min(dim='time')
    return txover

def process_forecast(txover):
    print("--- Processing Forecast ---")
    
    # Use a safe run time (2 hours ago) to ensure data availability
    model_init_time = (datetime.utcnow() - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    print(f"Using Model Run: {model_init_time} UTC")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0

    for fxx in range(1, FORECAST_HOURS + 1):
        try:
            # Fetch HRRR Forecast Temp
            H = Herbie(model_init_time, model="hrrr", product="sfc", fxx=fxx)
            ds = H.xarray(":(TMP):2 m above ground")
            
            t_sfc_f = (ds['t2m'] - 273.15) * 9/5 + 32
            
            # --- ALGORITHM ---
            # 1 = Mist (Yellow)
            # 2 = Dense Fog (Purple)
            fog_mask = np.zeros_like(t_sfc_f)
            
            # Logic: If T <= Txover, it's at least Mist.
            fog_mask[t_sfc_f <= txover] = 1        
            
            # Logic: If T is MORE than 3 degrees below Txover, it upgrades to Dense Fog.
            fog_mask[t_sfc_f <= (txover - 3.0)] = 2 
            
            # --- PLOTTING ---
            fig = plt.figure(figsize=(12, 10)) # Slightly larger for zoom
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
            
            # SET THE ZOOM (NC/VA/TN/SC)
            ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
            
            # Add Features
            ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='black')
            ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black')
            
            # Custom Colormap: Transparent -> Yellow -> Purple
            from matplotlib.colors import ListedColormap
            # index 0 (masked) = transparent
            # index 1 (Mist) = Bright Yellow
            # index 2 (Dense) = Deep Purple
            cmap = ListedColormap(['none', '#FFEB3B', '#8E24AA']) 
            
            data_to_plot = np.ma.masked_where(fog_mask == 0, fog_mask)
            
            ax.pcolormesh(ds.longitude, ds.latitude, data_to_plot, 
                               transform=ccrs.PlateCarree(), 
                               cmap=cmap, 
                               shading='auto')
            
            # Titles
            valid_time = model_init_time + timedelta(hours=fxx)
            plt.title(f"Crossover Fog Forecast (NC/VA/TN/SC)\nInit: {model_init_time.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", loc='left', fontsize=12, fontweight='bold')
            plt.title("Yellow: Mist (T < Tx)\nPurple: Dense Fog (T < Tx-3)", loc='right', fontsize=9, color='purple')
            
            # Save
            filename = f"fog_{datetime.utcnow().strftime('%Y%m%d')}_23z_f{fxx:02d}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            if os.path.exists(save_path):
                print(f"Generated {filename}")
                success_count += 1
            
        except Exception as e:
            print(f"Failed to generate frame {fxx}: {e}")

    if success_count == 0:
        print("ERROR: 0 images were generated. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        txover_grid = get_crossover_temp()
        process_forecast(txover_grid)
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)
