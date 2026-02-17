import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="herbie")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import json
import glob
import imageio.v2 as imageio
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from herbie import Herbie
import matplotlib.colors as mcolors

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXTENT = [-83, -75, 33.5, 40.5] 

CITIES = [
    [-80.22, 36.13, 'KINT'], [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8)
    # High-resolution County Boundaries
    counties = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_2_counties',
        scale='10m', facecolor='none'
    )
    ax.add_feature(counties, edgecolor='gray', linewidth=0.3)
    
    for lon, lat, name in CITIES:
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
        ax.text(lon + 0.05, lat + 0.05, name, transform=ccrs.PlateCarree(), 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# ================= DYNAMIC CROSSOVER LOGIC =================
print("Determining Crossover Temp at Peak Heating...")
current_utc = datetime.utcnow()
# Look back over the last 24 hours to find peak heating
search_hours = 24
times = [current_utc - timedelta(hours=i) for i in range(search_hours)]

try:
    # Use a representative hour (21Z) to initialize the grid size
    ref_time = current_utc.replace(hour=21, minute=0, second=0, microsecond=0)
    if ref_time > current_utc: ref_time -= timedelta(days=1)
    
    H_ref = Herbie(ref_time, model='rtma', product='anl')
    ds_ref = H_ref.xarray(":(TMP|DPT):2 m")
    if isinstance(ds_ref, list): ds_ref = ds_ref[0]
    if 'nav_lon' in ds_ref.coords: ds_ref = ds_ref.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})

    # Arrays to track Max T and the corresponding Dew Point
    max_t_grid = np.full(ds_ref['t2m'].shape, -999.0)
    crossover_grid = np.zeros_like(max_t_grid)

    # Scrape last 24 hours for Peak Heating
    for t in [ref_time - timedelta(hours=i) for i in range(12)]: # Looking at afternoon window
        try:
            H = Herbie(t, model='rtma', product='anl')
            ds = H.xarray(":(TMP|DPT):2 m")
            if isinstance(ds, list): ds = ds[0]
            
            temp_f = (ds['t2m'] - 273.15) * 9/5 + 32
            dwpt_f = (ds['d2m'] - 273.15) * 9/5 + 32
            
            # Update crossover where current T is higher than previous Max T
            mask = temp_f.values > max_t_grid
            max_t_grid[mask] = temp_f.values[mask]
            crossover_grid[mask] = dwpt_f.values[mask]
        except: continue

    crossover_f = crossover_grid
    longitude = ds_ref.longitude.values
    latitude = ds_ref.latitude.values

    # Plot the Dynamic Crossover Map
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    add_map_features(ax)
    levels = np.arange(20, 78, 2)
    norm = mcolors.BoundaryNorm(levels, plt.get_cmap('turbo').N)
    mesh = ax.pcolormesh(longitude, latitude, crossover_f, cmap='turbo', norm=norm, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, label='Crossover Temp (at Peak Heating) °F', shrink=0.8, ticks=levels[::2])
    plt.title(f"Dynamic Analysis: Crossover Threshold\nBased on Last 24h Peak Heating", loc='left', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=120)
    plt.close()

except Exception as e:
    print(f"Dynamic peak heating logic failed: {e}")
    exit(1)

# ================= 2. FORECAST LOOP =================
hrrr_init_time = (current_utc - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
run_id = hrrr_init_time.strftime("%Y%m%d_%Hz")

# Cleanup
for f in glob.glob(os.path.join(OUTPUT_DIR, "fog_*.png")):
    if run_id not in f: os.remove(f)

gif_frames = []
hourly_winds = {}
rtma_points = np.array([longitude.ravel(), latitude.ravel()]).T
rtma_values = crossover_f.ravel()

for fxx in range(1, 19):
    try:
        H_fcst = Herbie(hrrr_init_time, model='hrrr', product='sfc', fxx=fxx)
        ds_list = H_fcst.xarray(":(TMP):2 m|:(UGRD|VGRD):925 mb")
        ds_fcst = ds_list[0].merge(ds_list[1], compat='override') if isinstance(ds_list, list) else ds_list
        if 'nav_lon' in ds_fcst.coords: ds_fcst = ds_fcst.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        
        temp_f = (ds_fcst['t2m'] - 273.15) * 9/5 + 32
        thresh_on_grid = griddata(rtma_points, rtma_values, (ds_fcst.longitude.values, ds_fcst.latitude.values), method='linear')
        
        u_var, v_var = ('u925', 'v925') if 'u925' in ds_fcst else ('u', 'v')
        wind_kt = np.sqrt(ds_fcst[u_var]**2 + ds_fcst[v_var]**2) * 1.94384
        hourly_winds[fxx] = round(float(wind_kt.mean().values), 1)

        fog_mask = np.zeros_like(temp_f)
        fog_mask[(temp_f <= thresh_on_grid) & (wind_kt <= 15.0)] = 1
        fog_mask[(temp_f <= (thresh_on_grid - 3.0)) & (wind_kt <= 15.0)] = 2

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        add_map_features(ax)
        cmap = mcolors.ListedColormap(['none', 'gold', 'purple'])
        ax.pcolormesh(ds_fcst.longitude, ds_fcst.latitude, np.ma.masked_where(fog_mask == 0, fog_mask), 
                      transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=2)

        plt.title(f"Crossover Fog Forecast | Init: {hrrr_init_time.strftime('%H')}Z | Valid: +{fxx}h", loc='left', fontweight='bold', fontsize=12)
        ax.text(0.98, 1.05, "Dense Fog (< 1/2 SM)", color='purple', transform=ax.transAxes, ha='right', fontweight='bold')
        ax.text(0.98, 1.02, "Mist (1-3 SM)", color='orange', transform=ax.transAxes, ha='right', fontweight='bold')
        
        fname = os.path.join(OUTPUT_DIR, f"fog_{run_id}_f{fxx:02d}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        gif_frames.append(imageio.imread(fname))
        plt.close()
    except Exception as e: print(f"F{fxx} failed: {e}")

if gif_frames: imageio.mimsave(os.path.join(OUTPUT_DIR, "fog_animation.gif"), gif_frames, fps=2)

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{hrrr_init_time.strftime('%H')}Z", 
               "generated_at": current_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), "avg_winds": hourly_winds}, f)
