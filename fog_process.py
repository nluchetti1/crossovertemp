import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import os
import json
from datetime import datetime, timedelta
from herbie import Herbie  # pip install herbie-data

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the region (Mid-Atlantic based on your image)
EXTENT = [-84, -75, 33, 41] # [West, East, South, North]

# ================= DATE LOGIC =================
# We need to find the latest *completed* run.
# HRRR takes about 1-1.5 hours to upload. We look back 2 hours to be safe.
current_utc = datetime.utcnow()
model_init_time = current_utc - timedelta(hours=2)

# Round down to the nearest hour (e.g., 23:15 -> 23:00)
model_init_time = model_init_time.replace(minute=0, second=0, microsecond=0)

# Generate the ID strings
date_str = model_init_time.strftime("%Y%m%d") # 20260217
hour_str = model_init_time.strftime("%H")     # 23
run_id = f"{date_str}_{hour_str}z"            # 20260217_23z

print(f"=======================================")
print(f"Script Execution Time: {current_utc}")
print(f"Target Model Run:      {run_id}")
print(f"=======================================")

# ================= PROCESSING LOOP =================
# HRRR Forecast hours 1 through 18
forecast_hours = range(1, 19)

for fxx in forecast_hours:
    print(f"Processing Forecast Hour: {fxx:02d}...")

    try:
        # 1. Initialize Herbie to fetch HRRR data
        # We look for Surface fields (Temp and Dewpoint)
        H = Herbie(
            model_init_time,
            model='hrrr',
            product='sfc',
            fxx=fxx
        )

        # 2. Download/Open the specific variables we need
        # We need 2m Temperature (t2m) and 2m Dewpoint (d2m)
        # SearchString uses regex to grab only necessary layers to save bandwidth
        ds = H.xarray(":(TMP|DPT):2 m above ground")

        # 3. Extract Data & Apply Crossover Logic
        # Note: 't2m' and 'd2m' variable names might vary slightly by GRIB source
        # Usually in Herbie/cfgrib they map to 't2m' and 'd2m' or 't' and 'dpt'
        
        # Verify variable names in your specific environment if this fails
        temp_k = ds['t2m'] 
        dew_k = ds['d2m']

        # Convert Kelvin to Fahrenheit
        temp_f = (temp_k - 273.15) * 9/5 + 32
        dew_f = (dew_k - 273.15) * 9/5 + 32
        
        # --- CROSSOVER LOGIC ---
        # NOTE: True Crossover logic requires the previous afternoon's Min Dewpoint Depression.
        # Since this is a simple hourly run, we are approximating based on your image legend:
        # "Yellow = Mist (T < Tx)" and "Purple = Dense Fog (T < Tx-3)"
        
        # Simple Fog Approximation (T - Td spread)
        # Modify this math to match your exact Crossover formula
        dep = temp_f - dew_f
        
        # Mask: 0 = Clear, 1 = Mist (Yellow), 2 = Dense Fog (Purple)
        fog_mask = np.zeros_like(temp_f)
        
        # If Depression < 3.0 -> Mist (Yellow)
        fog_mask[dep < 3.0] = 1 
        # If Depression < 1.0 -> Dense Fog (Purple)
        fog_mask[dep < 1.0] = 2 

        # ================= PLOTTING =================
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT)

        # Add Features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        # Plot the Fog Mask
        # We use a custom colormap: Transparent, Yellow, Purple
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['none', 'gold', 'purple'])
        
        # Plot data (using pcolormesh for speed)
        # Using a masked array to hide "Clear" areas
        masked_data = np.ma.masked_where(fog_mask == 0, fog_mask)
        
        mesh = ax.pcolormesh(
            ds.longitude, ds.latitude, masked_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=0, vmax=2
        )

        # Titles
        valid_time = model_init_time + timedelta(hours=fxx)
        plt.title(f"Crossover Fog Forecast\nInit: {hour_str}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", loc='left', fontsize=10, fontweight='bold')
        plt.title("Yellow: Mist | Purple: Dense Fog", loc='right', fontsize=8, color='purple')

        # Save Image
        filename = f"fog_{run_id}_f{fxx:02d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        print(f" -> Saved {filename}")

    except Exception as e:
        print(f"Failed to process hour {fxx}: {e}")

# ================= STATUS UPDATE =================
# This is the crucial handshake for your website
status_data = {
    "run_id": run_id,               # e.g. "20260217_23z"
    "model_init": f"{hour_str}Z",   # e.g. "23Z"
    "generated_at": current_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
}

json_path = os.path.join(OUTPUT_DIR, "current_status.json")
with open(json_path, "w") as f:
    json.dump(status_data, f)

print("Process Complete. Status updated.")
