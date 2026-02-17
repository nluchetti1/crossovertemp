import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import json
import glob
import imageio.v2 as imageio
from datetime import datetime, timedelta
from herbie import Herbie
import matplotlib.colors as mcolors

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXTENT = [-81.5, -76.5, 33.8, 37.0] 

CITIES = [
    [-80.22, 36.13, 'KINT'], [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8)
    for lon, lat, name in CITIES:
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
        ax.text(lon + 0.03, lat + 0.03, name, transform=ccrs.PlateCarree(), 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# ================= DATE LOGIC =================
current_utc = datetime.utcnow()
if current_utc.hour < 22:
    target_date = current_utc - timedelta(days=1)
else:
    target_date = current_utc

rtma_time = target_date.replace(hour=21, minute=0, second=0, microsecond=0)
hrrr_init_time = (current_utc - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
run_id = hrrr_init_time.strftime("%Y%m%d_%Hz")

# Cleanup
for f in glob.glob(os.path.join(OUTPUT_DIR, "fog_*.png")):
    if run_id not in f: os.remove(f)

# ================= 1. FETCH RTMA CROSSOVER =================
print(f"Fetching RTMA 21Z Threshold...")
try:
    H_rtma = Herbie(rtma_time, model='rtma', product='anl')
    ds_rtma = H_rtma.xarray(":(DPT):2 m")
    if isinstance(ds_rtma, list): ds_rtma = ds_rtma[0]
    
    # Force coordinate names
    if 'nav_lon' in ds_rtma.coords: ds_rtma = ds_rtma.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
    
    # Convert to DataArray and ensure lat/lon are coords
    crossover_f = (ds_rtma['d2m'] - 273.15) * 9/5 + 32

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    add_map_features(ax)
    levels = np.arange(20, 78, 2)
    norm = mcolors.BoundaryNorm(levels, plt.get_cmap('turbo').N)
    mesh = ax.pcolormesh(ds_rtma.longitude, ds_rtma.latitude, crossover_f, cmap='turbo', norm=norm, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, label='RTMA 21Z Crossover Threshold °F', shrink=0.8, ticks=levels[::2])
    plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=120)
    plt.close()
except Exception as e:
    print(f"RTMA Failed: {e}")
    exit(1)

# ================= 2. FORECAST LOOP =================
gif_frames = []
hourly_winds = {}

for fxx in range(1, 19):
    try:
        H_fcst = Herbie(hrrr_init_time, model='hrrr', product='sfc', fxx=fxx)
        ds_list = H_fcst.xarray(":(TMP):2 m|:(UGRD|VGRD):925 mb")
        
        # Merge datasets if list returned
        if isinstance(ds_list, list):
            ds_fcst = ds_list[0].merge(ds_list[1], compat='override')
        else:
            ds_fcst = ds_list
        
        # Force coordinate names for HRRR
        if 'nav_lon' in ds_fcst.coords: ds_fcst = ds_fcst.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        
        temp_f = (ds_fcst['t2m'] - 273.15) * 9/5 + 32
        
        # INTERPOLATION: Use the .interp() method with lat/lon directly
        # This bypasses the x/y dimension size mismatch
        thresh_on_grid = crossover_f.interp(
            longitude=ds_fcst.longitude, 
            latitude=ds_fcst.latitude, 
            method='linear'
        )
        
        u_var = 'u925' if 'u925' in ds_fcst else 'u'
        v_var = 'v925' if 'v925' in ds_fcst else 'v'
        wind_kt = np.sqrt(ds_fcst[u_var]**2 + ds_fcst[v_var]**2) * 1.94384
        avg_wind = float(wind_kt.mean().values)
        hourly_winds[fxx] = round(avg_wind, 1)

        fog_mask = np.zeros_like(temp_f)
        fog_mask[(temp_f <= thresh_on_grid) & (wind_kt <= 15.0)] = 1
        fog_mask[(temp_f <= (thresh_on_grid - 3.0)) & (wind_kt <= 15.0)] = 2

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        add_map_features(ax)
        cmap = mcolors.ListedColormap(['none', 'gold', 'purple'])
        ax.pcolormesh(ds_fcst.longitude, ds_fcst.latitude, np.ma.masked_where(fog_mask == 0, fog_mask), 
                      transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=2)

        plt.title(f"Combo Forecast | Init: {hrrr_init_time.strftime('%H')}Z | Valid: +{fxx}h\nAvg 925mb Wind: {hourly_winds[fxx]} kt", loc='left', fontweight='bold')
        
        fname = os.path.join(OUTPUT_DIR, f"fog_{run_id}_f{fxx:02d}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        gif_frames.append(imageio.imread(fname))
        plt.close()
        print(f"✅ F{fxx} done")
    except Exception as e: 
        print(f"❌ F{fxx} failed: {e}")

if gif_frames: imageio.mimsave(os.path.join(OUTPUT_DIR, "fog_animation.gif"), gif_frames, fps=2)

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{hrrr_init_time.strftime('%H')}Z", 
               "generated_at": current_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), "avg_winds": hourly_winds}, f)
