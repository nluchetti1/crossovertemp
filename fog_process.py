import warnings
# Suppress the Herbie/Pandas regex warnings to keep logs clean
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

# Centered Domain on North Carolina
EXTENT = [-83, -75, 31.5, 38.5] 

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
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    counties = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_2_counties',
        scale='10m', facecolor='none'
    )
    ax.add_feature(counties, edgecolor='gray', linewidth=0.3)

# ================= 1. DYNAMIC PEAK HEATING LOGIC =================
print("Starting Dynamic Crossover Analysis...")
now = datetime.utcnow()

try:
    ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if ref_time > now: ref_time -= timedelta(days=1)
    
    H_init = Herbie(ref_time, model='rtma', product='anl')
    ds_init = H_init.xarray(":(TMP|DPT):2 m")
    if isinstance(ds_init, list): ds_init = ds_init[0]
    
    if 'nav_lon' in ds_init.coords:
        ds_init = ds_init.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
    
    lons_rtma = ds_init.longitude.values
    lats_rtma = ds_init.latitude.values
    max_t_grid = np.full(lons_rtma.shape, -999.0)
    xover_grid = np.full(lons_rtma.shape, -999.0)

    for i in range(12):
        check_time = ref_time - timedelta(hours=i)
        try:
            H = Herbie(check_time, model='rtma', product='anl', verbose=False)
            ds = H.xarray(":(TMP|DPT):2 m")
            if isinstance(ds, list): ds = ds[0]
            t_vals = (ds['t2m'].values - 273.15) * 9/5 + 32
            d_vals = (ds['d2m'].values - 273.15) * 9/5 + 32
            mask = t_vals > max_t_grid
            max_t_grid[mask] = t_vals[mask]
            xover_grid[mask] = d_vals[mask]
        except: continue

    # Analysis Map
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    add_map_features(ax)
    levels = np.arange(20, 78, 2)
    cmap = plt.get_cmap('turbo', len(levels) - 1)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    mesh = ax.pcolormesh(lons_rtma, lats_rtma, xover_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    plt.colorbar(mesh, ax=ax, shrink=0.8, ticks=levels, label='°F')
    
    # Robust coordinate lookup for airport labels
    rtma_points = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
    for lon, lat, name in CITIES:
        # Using griddata to interpolate at the exact site
        val_max_t = griddata(rtma_points, max_t_grid.ravel(), (lon, lat), method='linear')
        val_xover = griddata(rtma_points, xover_grid.ravel(), (lon, lat), method='linear')
        
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
        
        # Shift KINT label to the left to avoid overlap with KGSO
        x_offset = -0.12 if name == 'KINT' else 0.08
        ha_val = 'right' if name == 'KINT' else 'left'
        
        # Display readout; fallback to "N/A" if griddata returns NaN
        t_str = f"{val_max_t:.0f}°" if not np.isnan(val_max_t) else "N/A"
        c_str = f"{val_xover:.0f}°" if not np.isnan(val_xover) else "N/A"
        
        ax.text(lon + x_offset, lat, f"{name}\nMaxT: {t_str}\nCovr: {c_str}", 
                transform=ccrs.PlateCarree(), fontsize=8, fontweight='bold', va='center', ha=ha_val,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', pad=2))

    plt.title(f"Dynamic Crossover Analysis | Ref: {ref_time.strftime('%Y-%m-%d')}", loc='left', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=130)
    plt.close()
except Exception as e:
    print(f"Analysis Failed: {e}"); exit(1)

# ================= 2. FORECAST LOOP =================
hrrr_init = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
run_id = hrrr_init.strftime("%Y%m%d_%Hz")

# Flatten points for HRRR grid mapping
rtma_flat_vals = xover_grid.ravel()

gif_frames = []
for fxx in range(1, 2):
    try:
        H_fcst = Herbie(hrrr_init, model='hrrr', product='sfc', fxx=fxx, verbose=False)
        ds_list = H_fcst.xarray(":(TMP):2 m|:(UGRD|VGRD):925 mb")
        ds_f = ds_list[0].merge(ds_list[1], compat='override') if isinstance(ds_list, list) else ds_list
        if 'nav_lon' in ds_f.coords: ds_f = ds_f.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        
        f_temp = (ds_f['t2m'].values - 273.15) * 9/5 + 32
        u_925, v_925 = (ds_f['u925'].values, ds_f['v925'].values) if 'u925' in ds_f else (ds_f['u'].values, ds_f['v'].values)
        f_wind = np.sqrt(u_925**2 + v_925**2) * 1.94384
        
        # Mapping RTMA Threshold onto HRRR grid
        f_thresh = griddata(rtma_points, rtma_flat_vals, (ds_f.longitude.values, ds_f.latitude.values), method='linear')

        fog_layer = np.zeros_like(f_temp)
        fog_layer[(f_temp <= f_thresh) & (f_wind <= 15.0)] = 1
        fog_layer[(f_temp <= (f_thresh - 3.0)) & (f_wind <= 15.0)] = 2

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        add_map_features(ax)
        for lon, lat, name in CITIES:
            x_off = -0.05 if name == 'KINT' else 0.05
            h_align = 'right' if name == 'KINT' else 'left'
            ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
            ax.text(lon + x_off, lat + 0.05, name, transform=ccrs.PlateCarree(), 
                    fontsize=9, fontweight='bold', ha=h_align, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax.pcolormesh(ds_f.longitude, ds_f.latitude, np.ma.masked_where(fog_layer == 0, fog_layer), 
                      transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['none', 'gold', 'purple']), vmin=0, vmax=2)

        v_z = (hrrr_init + timedelta(hours=fxx)).strftime('%HZ')
        plt.title(f"Crossover Fog Forecast | Init: {hrrr_init.strftime('%H')}Z | Valid: {v_z}", loc='left', fontweight='bold')
        ax.text(0.98, 1.05, "Dense Fog (< 1/2 SM)", color='purple', transform=ax.transAxes, ha='right', fontweight='bold')
        ax.text(0.98, 1.02, "Mist (1-3 SM)", color='orange', transform=ax.transAxes, ha='right', fontweight='bold')
        
        save_p = os.path.join(OUTPUT_DIR, f"fog_{run_id}_f{fxx:02d}.png")
        plt.savefig(save_p, bbox_inches='tight', dpi=100)
        gif_frames.append(imageio.imread(save_p)); plt.close()
    except: continue

if gif_frames: imageio.mimsave(os.path.join(OUTPUT_DIR, "fog_animation.gif"), gif_frames, fps=2)
with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{hrrr_init.strftime('%H')}Z", "generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
