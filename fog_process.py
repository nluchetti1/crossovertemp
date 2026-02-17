import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
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
            print(f"Skipping {time_step}: {e}")

    if not dps:
        print("CRITICAL: No dewpoint data found.")
        sys.exit(1)

    # Stack and find minimum Dewpoint per pixel
    concat_da = xr.concat(dps, dim='time')
    txover = concat_da.min(dim='time')
    
    # Preserve coordinates for plotting
    return txover

def plot_crossover_map(txover, ds_sample):
    """
    Generates a static map of the calculated Crossover Temp (Min Afternoon Dewpoint).
    """
    print("--- Generating Crossover Analysis Map ---")
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
        ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())

        # Features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=1)
        
        # Add Counties
        reader = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural', name='admin_2_counties_lakes_north_america'))
        counties = list(reader.geometries())
        COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
        ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray', linewidth=0.3)

        # Plot Data
        cs = ax.pcolormesh(ds_sample.longitude, ds_sample.latitude, txover, 
                           transform=ccrs.PlateCarree(), 
                           cmap='BrBG', vmin=20, vmax=70, shading='auto')
        
        cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05, aspect=50)
        cbar.set_label("Dewpoint (°F)")

        plt.title(f"Calculated Crossover Temp (Min Afternoon Dewpoint)\nBased on lowest Td from last {LOOKBACK_HOURS} hours", fontweight='bold')
        
        filename = "crossover_analysis.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Generated {filename}")
        
    except Exception as e:
        print(f"Failed to generate crossover map: {e}")

def process_forecast(txover):
    print("--- Processing Forecast ---")
    
    # Safe run time
    model_init_time = (datetime.utcnow() - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    print(f"Using Model Run: {model_init_time} UTC")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0

    # Load Counties Reader once to save time in loop
    print("Loading County Shapefiles...")
    reader = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural', name='admin_2_counties_lakes_north_america'))
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    for fxx in range(1, FORECAST_HOURS + 1):
        try:
            H = Herbie(model_init_time, model="hrrr", product="sfc", fxx=fxx)
            ds = H.xarray(":(TMP):2 m above ground")
            t_sfc_f = (ds['t2m'] - 273.15) * 9/5 + 32
            
            # --- ALGORITHM ---
            fog_mask = np.zeros_like(t_sfc_f)
            fog_mask[t_sfc_f <= txover] = 1        # Mist
            fog_mask[t_sfc_f <= (txover - 3.0)] = 2 # Dense Fog
            
            # --- PLOTTING ---
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
            ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
            
            # Features
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linewidth=1)
            ax.add_feature(cfeature.STATES, linewidth=0.8)
            ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray', linewidth=0.3, alpha=0.5)
            
            # Colormap
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(['none', '#FFEB3B', '#8E24AA']) 
            data_to_plot = np.ma.masked_where(fog_mask == 0, fog_mask)
            
            ax.pcolormesh(ds.longitude, ds.latitude, data_to_plot, 
                               transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
            
            valid_time = model_init_time + timedelta(hours=fxx)
            
            # Updated Title (Removed Location String)
            plt.title(f"Crossover Fog Forecast\nInit: {model_init_time.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", loc='left', fontsize=12, fontweight='bold')
            plt.title("Yellow: Mist (T < Tx)\nPurple: Dense Fog (T < Tx-3)", loc='right', fontsize=9, color='purple')
            
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
        # Get Txover
        txover_grid = get_crossover_temp()
        
        # Get a sample DS for coordinate plotting in the static map
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        H_sample = Herbie(now, model="hrrr", product="sfc", fxx=0)
        ds_sample = H_sample.xarray(":(DPT):2 m above ground")
        
        # Plot the Crossover Map
        plot_crossover_map(txover_grid, ds_sample)
        
        # Run Forecast
        process_forecast(txover_grid)
        
    except Exception as e:
        print(f"Critical Error: {e}")
        sys.exit(1)
