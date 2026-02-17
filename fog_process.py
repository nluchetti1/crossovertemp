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

# Target NC Airports
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

# Cleanup old PNGs
old_pngs = glob.glob(os.path.join(OUTPUT_DIR, "fog_*.png"))
for f in old_pngs:
    if run_id not in f:
        os.remove(f)

# ================= 1. FETCH RTMA CROSSOVER THRESHOLD =================
print(f"Fetching RTMA 21Z Threshold for {rtma_time}...")
try:
    H_rtma = Herbie(rtma_time, model='rtma', product='anl')
    ds_rtma = H_rtma.xarray(":(DPT):2 m")
    if isinstance(ds_rtma, list): ds_rtma = ds_rtma[0]
    
    crossover_f = (ds_rtma['d2m'] - 273.15) * 9/5 + 32
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    add_map_features(ax)
    
    # Precise 2-degree bins for the Analysis Plot
    levels = np.arange(20, 78, 2)
    cmap = plt.get_cmap('turbo', len(levels) - 1)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    
    mesh = ax.pcolormesh(ds_rtma.longitude, ds_rtma.latitude, crossover_f, 
                          cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    # Create the "Stepped" Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, ticks=levels[::2])
    cbar.set_label('Crossover Threshold (RTMA 21Z Dewpoint) °F', fontweight='bold')
    
    plt.title(f"Input Analysis: Derived Crossover Temp\nRef: {rtma_time.strftime('%Y-%m-%d %H')}Z", loc='left', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=120)
    plt.close()
except Exception as e:
    print(f"RTMA Fetch Failed: {e}. Fallback to HRRR f00.")
    H_fallback = Herbie(hrrr_init_time, model='hrrr', product='sfc', fxx=0)
    ds_fallback = H_fallback.xarray(":(DPT):2 m")
    if isinstance(ds_fallback, list): ds_fallback = ds_fallback[0]
    crossover_f = (ds_fallback['d2m'] - 273.15) * 9/5 + 32

# ================= 2. GENERATE FORECAST LOOP =================
gif_frames = []
for fxx in range(1, 19):
    try:
        H_fcst = Herbie(hrrr_init_time, model='hrrr', product='sfc', fxx=fxx)
        ds_list = H_fcst.xarray(":(TMP):2 m|:(UGRD|VGRD):925 mb")
        
        if isinstance(ds_list, list):
            ds_fcst = ds_list[0]
            for extra_ds in ds_list[1:]:
                ds_fcst = ds_fcst.merge(extra_ds, compat='override')
        else:
            ds_fcst = ds_list
        
        temp_f = (ds_fcst['t2m'] - 273.15) * 9/5 + 32
        thresh_on_grid = crossover_f.interp_like(temp_f)
        
        u_var = 'u925' if 'u925' in ds_fcst else 'u'
        v_var = 'v925' if 'v925' in ds_fcst else 'v'
        wind_kt = np.sqrt(ds_fcst[u_var]**2 + ds_fcst[v_var]**2) * 1.94384

        fog_mask = np.zeros_like(temp_f)
        # Combo Logic: T <= Tx AND 925mb Winds <= 15kts
        fog_mask[(temp_f <= thresh_on_grid) & (wind_kt <= 15.0)] = 1
        fog_mask[(temp_f <= (thresh_on_grid - 3.0)) & (wind_kt <= 15.0)] = 2

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        add_map_features(ax)
        cmap_fog = mcolors.ListedColormap(['none', 'gold', 'purple'])
        ax.pcolormesh(ds_fcst.longitude, ds_fcst.latitude, np.ma.masked_where(fog_mask == 0, fog_mask), 
                      transform=ccrs.PlateCarree(), cmap=cmap_fog, vmin=0, vmax=2)

        valid_time = hrrr_init_time + timedelta(hours=fxx)
        plt.title(f"Combo Fog Forecast | Init: {hrrr_init_time.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z", loc='left', fontweight='bold')
        ax.text(0.98, 1.05, "Dense (< 1/2 SM)", color='purple', transform=ax.transAxes, ha='right', fontweight='bold')
        ax.text(0.98, 1.02, "Mist (1-3 SM)", color='orange', transform=ax.transAxes, ha='right', fontweight='bold')
        
        fname = os.path.join(OUTPUT_DIR, f"fog_{run_id}_f{fxx:02d}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        gif_frames.append(imageio.imread(fname))
        plt.close()
    except Exception as e: print(f"Forecast hour {fxx} failed: {e}")

if gif_frames:
    imageio.mimsave(os.path.join(OUTPUT_DIR, "fog_animation.gif"), gif_frames, fps=2)

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{hrrr_init_time.strftime('%H')}Z", "generated_at": current_utc.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
