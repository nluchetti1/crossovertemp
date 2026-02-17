import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import json
import imageio.v2 as imageio
from datetime import datetime, timedelta
from herbie import Herbie
import matplotlib.colors as mcolors

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Focused extent for North Carolina region
EXTENT = [-81.5, -76.5, 33.8, 37.0] 

# Strictly the requested NC Airport List
CITIES = [
    [-80.22, 36.13, 'KINT'], 
    [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], 
    [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

def add_map_features(ax):
    """Adds standard geographical features and specific NC airport markers."""
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    for lon, lat, name in CITIES:
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
        ax.text(lon + 0.03, lat + 0.03, name, transform=ccrs.PlateCarree(), 
                fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# ================= DATE LOGIC =================
current_utc = datetime.utcnow()
# 2-hour lag for HRRR availability
model_init_time = (current_utc - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
run_id = model_init_time.strftime("%Y%m%d_%Hz")

# ================= 1. HIGH-PRECISION BINNED ANALYSIS =================
print("Generating 2-Degree Binned Analysis...")
try:
    H_obs = Herbie(model_init_time, model='hrrr', product='sfc', fxx=0)
    ds_obs = H_obs.xarray(":(DPT):2 m")
    dpt_f = (ds_obs['d2m'] - 273.15) * 9/5 + 32

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    add_map_features(ax)

    # 2-degree bins for high-precision readout
    levels = np.arange(20, 76, 2)
    cmap = plt.get_cmap('turbo', len(levels) - 1)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    mesh = ax.pcolormesh(ds_obs.longitude, ds_obs.latitude, dpt_f, 
                          cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    plt.colorbar(mesh, ax=ax, orientation='vertical', label='Crossover Temp (Dewpoint) °F', shrink=0.8, pad=0.02)
    plt.title(f"Input Analysis: Derived Crossover Temp\nValid: {model_init_time.strftime('%Y-%m-%d %H')}Z", loc='left', fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=120)
    plt.close()
except Exception as e:
    print(f"Analysis plot failed: {e}")

# ================= 2. FORECAST LOOP & GIF =================
gif_frames = []
for fxx in range(1, 19):
    try:
        H = Herbie(model_init_time, model='hrrr', product='sfc', fxx=fxx)
        ds = H.xarray(":(TMP|DPT):2 m")
        temp_f = (ds['t2m'] - 273.15) * 9/5 + 32
        dew_f = (ds['d2m'] - 273.15) * 9/5 + 32
        dep = temp_f - dew_f 
        
        fog_mask = np.zeros_like(temp_f)
        fog_mask[dep <= 3.0] = 1   # Mist
        fog_mask[dep <= 0.0] = 2   # Dense Fog

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        add_map_features(ax)
        cmap_fog = mcolors.ListedColormap(['none', 'gold', 'purple'])
        ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog_mask == 0, fog_mask), 
                      transform=ccrs.PlateCarree(), cmap=cmap_fog, vmin=0, vmax=2)

        valid_time = model_init_time + timedelta(hours=fxx)
        plt.title(f"Fog Forecast | Init: {model_init_time.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", 
                  loc='left', fontweight='bold', fontsize=12)
        
        # Visibility Legend in title
        ax.text(0.98, 1.02, "Dense Fog (< 1/2 SM)", color='purple', transform=ax.transAxes, ha='right', fontweight='bold')
        ax.text(0.70, 1.02, "Mist (1-3 SM)", color='orange', transform=ax.transAxes, ha='right', fontweight='bold')

        fname = os.path.join(OUTPUT_DIR, f"fog_{run_id}_f{fxx:02d}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        gif_frames.append(imageio.imread(fname))
        plt.close()
        
    except Exception as e:
        print(f"F{fxx} failed: {e}")

if gif_frames:
    imageio.mimsave(os.path.join(OUTPUT_DIR, "fog_animation.gif"), gif_frames, fps=2)

# Save status for HTML
with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{model_init_time.strftime('%H')}Z", 
               "generated_at": current_utc.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
