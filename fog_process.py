import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from herbie import Herbie
from datetime import datetime, timedelta
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# The paper defines Txover as the min dewpoint during warmest daytime hours.
# Running at 6PM (approx 23Z), we look back at the peak heating of the day.
LOOKBACK_HOURS = 6   # Look at past 6 hours for min Dewpoint
FORECAST_HOURS = 18  # Forecast through the next morning
OUTPUT_DIR = "images"

def get_crossover_temp():
    """
    Calculates TXover: Minimum dewpoint observed during warmest daytime hours.
    """
    print("--- Calculating Crossover Temperature (TXover) ---")
    dps = []
    
    # Current time (runtime)
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
            print(f"Skipping {time_step}: {e}")

    if not dps:
        raise ValueError("No data found for Crossover calculation.")

    # Stack and find minimum Dewpoint per pixel over the time window
    concat_da = xr.concat(dps, dim='time')
    txover = concat_da.min(dim='time')
    return txover

def process_forecast(txover):
    """
    Compare forecasted T to TXover to assign Fog Risk.
    """
    print("--- Processing Forecast ---")
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean up old images to keep repo size down (optional)
    # for f in os.listdir(OUTPUT_DIR):
    #     os.remove(os.path.join(OUTPUT_DIR, f))

    for fxx in range(1, FORECAST_HOURS + 1):
        try:
            # Fetch HRRR Forecast Temp
            H = Herbie(now, model="hrrr", product="sfc", fxx=fxx)
            ds = H.xarray(":(TMP):2 m above ground")
            
            # Convert to Fahrenheit
            t_sfc_f = (ds['t2m'] - 273.15) * 9/5 + 32
            
            # --- APPLY PAPER LOGIC ---
            # 0 = No Fog
            # 1 = Mist (T <= Txover) [Source: Baker et al.]
            # 2 = Dense Fog (T <= Txover - 3) [Source: Baker et al.]
            
            fog_mask = np.zeros_like(t_sfc_f)
            
            # Mist condition
            fog_mask[t_sfc_f <= txover] = 1
            
            # Dense Fog condition (overwrites Mist where applicable)
            fog_mask[t_sfc_f <= (txover - 3.0)] = 2
            
            # --- PLOTTING ---
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            
            # Colors: Transparent, Yellow (Mist), Purple (Dense Fog)
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(['none', '#FFEB3B', '#9C27B0']) 
            
            # Mask the "0" (No fog) values so they don't block the map
            data_to_plot = np.ma.masked_where(fog_mask == 0, fog_mask)
            
            cs = ax.pcolormesh(ds.longitude, ds.latitude, data_to_plot, 
                               transform=ccrs.PlateCarree(), 
                               cmap=cmap, 
                               shading='auto')
            
            valid_time = now + timedelta(hours=fxx)
            
            # Title
            plt.title(f"Crossover Temp Fog Forecast\nInit: {now.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", loc='left', fontsize=10)
            plt.title("Yellow: Mist (T < Tx)\nPurple: Dense Fog (T < Tx-3)", loc='right', fontsize=8, color='purple')
            
            # Filename: fog_YYYYMMDD_RunHour_ForecastHour.png
            filename = f"fog_{now.strftime('%Y%m%d')}_{now.strftime('%H')}z_f{fxx:02d}.png"
            plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches='tight', dpi=100)
            plt.close()
            print(f"Generated {filename}")
            
        except Exception as e:
            print(f"Failed to generate frame {fxx}: {e}")

if __name__ == "__main__":
    try:
        # 1. Calculate Crossover Temp (Min Dewpoint from afternoon)
        txover_grid = get_crossover_temp()
        
        # 2. Run Forecast against Crossover Temp
        process_forecast(txover_grid)
    except Exception as e:
        print(f"Critical Error: {e}")
