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
# Map bounds for the Mid-Atlantic
EXTENT = [-83, -75, 33.5, 40.5] 

# Major Cities for spatial reference
CITIES = [
    [-77.43, 37.54, 'RIC'], [-78.64, 35.77, 'RDU'], 
    [-76.28, 36.85, 'ORF'], [-79.94, 37.27, 'ROA'],
    [-80.84, 35.22, 'CLT'], [-77.03, 38.90, 'DCA'],
    [-81.63, 38.35, 'CRW'], [-76.61, 39.29, 'BWI']
]

def add_map_features(ax):
    """Adds standard geographical features and city markers."""
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    for lon, lat, name in CITIES:
        ax.plot(lon, lat, 'ko', markersize=3, transform=ccrs.PlateCarree())
        ax.text(lon + 0.05, lat + 0.05, name, transform=ccrs.PlateCarree(), 
                fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# ================= DATE LOGIC =================
current_utc = datetime.utcnow()
# Look back 2 hours to ensure the model run has finished uploading
model_init_time = (current_utc - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
run_id = model_init_time.strftime("%Y%m%d_%Hz")

# ================= 1. GENERATE BINNED ANALYSIS PLOT =================
print("Generating Binned Input Analysis...")
try:
    H_obs = Herbie(model_init_time, model='hrrr', product='sfc', fxx=0)
    ds_obs = H_obs.xarray(":(DPT):2 m")
    dpt_f = (ds_obs['d2m'] - 273.15) * 9/5 + 32

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    add_map_features(ax)

    # Create 5-degree bins for the Crossover Temp (Dewpoint)
    levels = np.arange(20, 75, 5)
    cmap = plt.get_cmap('viridis', len(levels) - 1)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    mesh = ax.pcolormesh(ds_obs.longitude, ds_obs.latitude, dpt_f, 
                          cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    
    plt.colorbar(mesh, ax=ax, orientation='vertical', label='Crossover Temp (Dewpoint) °F', shrink=0.7, pad=0.02)
    plt.title(f"Input Analysis: Derived Crossover Temp\nValid: {model_init_time.strftime('%Y-%m-%d %H')}Z", loc='left', fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=120)
    plt.close()
except Exception as e:
    print(f"Analysis plot failed: {e}")

# ================= 2. GENERATE FORECAST LOOP & GIF =================
gif_frames = []
print(f"Generating Forecast Loop for {run_id}...")

for fxx in range(1, 19):
    try:
        H = Herbie(model_init_time, model='hrrr', product='sfc', fxx=fxx)
        ds = H.xarray(":(TMP|DPT):2 m")

        temp_f = (ds['t2m'] - 273.15) * 9/5 + 32
        dew_f = (ds['d2m'] - 273.15) * 9/5 + 32
        dep = temp_f - dew_f 
        
        # 0=Clear, 1=Mist (Yellow), 2=Dense Fog (Purple)
        fog_mask = np.zeros_like(temp_f)
        fog_mask[dep <= 3.0] = 1   # T <= Txover (1-3 SM)
        fog_mask[dep <= 0.0] = 2   # T <= Txover-3 (approx) (< 1/2 SM)

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        add_map_features(ax)

        cmap_fog = mcolors.ListedColormap(['none', 'gold', 'purple'])
        masked_data = np.ma.masked_where(fog_mask == 0, fog_mask)
        
        ax.pcolormesh(ds.longitude, ds.latitude, masked_data, transform=ccrs.PlateCarree(), cmap=cmap_fog, vmin=0, vmax=2)

        # Labels & Legend
        valid_time = model_init_time + timedelta(hours=fxx)
        plt.title(f"Fog Forecast | Init: {model_init_time.strftime('%H')}Z | Valid: {valid_time.strftime('%a %H')}Z (+{fxx})", 
                  loc='left', fontweight='bold', fontsize=12)
        
        # Color-coded Title Legend
        ax.text(0.98, 1.02, "Dense Fog (< 1/2 SM)", color='purple', transform=ax.transAxes, ha='right', fontweight='bold')
        ax.text(0.70, 1.02, "Mist (1-3 SM)", color='orange', transform=ax.transAxes, ha='right', fontweight='bold')

        fname = os.path.join(OUTPUT_DIR, f"fog_{run_id}_f{fxx:02d}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=100)
        
        # Add frame to GIF list
        gif_frames.append(imageio.imread(fname))
        plt.close()
        print(f" -> Processed f{fxx:02d}")
        
    except Exception as e:
        print(f"Forecast hour {fxx} failed: {e}")

# Save GIF
if gif_frames:
    print("Saving GIF Animation...")
    imageio.mimsave(os.path.join(OUTPUT_DIR, "fog_animation.gif"), gif_frames, fps=2)

# ================= STATUS UPDATE =================
with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({
        "run_id": run_id, 
        "model_init": f"{model_init_time.strftime('%H')}Z",
        "generated_at": current_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    }, f)

print("Process Complete.")
