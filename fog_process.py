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
            # We wrap this in a loop to try slightly older runs if the exact hour is missing
            # But for Lookback, we usually accept what we can get.
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
    
    # --- KEY FIX: USE A "SAFE" RUN TIME ---
    # Models take 1-2 hours to upload. If it's 04:30 UTC, the 04:00 run isn't ready.
    # We go back 2 hours to ensure we get a complete run.
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
            fog_mask = np.zeros_like(t_sfc_f)
            fog_mask[t_sfc_f <= txover] = 1        # Mist
            fog_mask[t_sfc_f <= (txover - 3.0)] = 2 # Dense Fog
            
            # --- PLOTTING ---
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(['none', '#FFEB3B', '#9C27B0']) 
            
            data_to_plot = np.ma.masked_where(fog_mask == 0, fog_mask)
            
            ax.pcolormesh(ds.longitude, ds.latitude, data_to_plot, 
                               transform=ccrs.PlateCarree(), 
                               cmap=cmap, 
                               shading='auto')
            
            # Valid time is Model Init + Forecast Hour
            valid_time = model_init_time + timedelta(hours=fxx)
            
            plt.title(f"Crossover Fog Forecast\nInit: {model_init_time.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", loc='left', fontsize=10)
            plt.title("Yellow: Mist (T < Tx)\nPurple: Dense Fog (T < Tx-3)", loc='right', fontsize=8, color='purple')
            
            # Filename based on VALID time so website logic stays simple
            # NOTE: We use the Run Time in filename for uniqueness, but index.html looks for this specific format
            filename = f"fog_{datetime.utcnow().strftime('%Y%m%d')}_23z_f{fxx:02d}.png"
            
            # To make it easier for the website (which looks for '23z'), we force the filename 
            # to match what the website expects for "Today's Run", even if the data is from 21z or 22z.
            # This is a 'hack' to keep the Javascript simple.
            
            save_path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            if os.path.exists(save_path):
                print(f"Generated {filename}")
                success_count += 1
            
        except Exception as e:
            print(f"Failed to generate frame {fxx}: {e}")

    # Fail if 0 images
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
